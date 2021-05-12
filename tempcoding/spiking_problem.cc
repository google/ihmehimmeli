// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tempcoding/spiking_problem.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "common/file_passthrough.h"
#include "common/mnist_loader.h"
#include "common/util.h"
#include "tempcoding/tempcoder.h"

namespace ihmehimmeli {

namespace internal {

CsvData LoadCsvData(const std::string& file_path) {
  file::IhmFile file = file::OpenOrDie(file_path, "r");
  std::string file_content;
  file.ReadWholeFileToString(&file_content);
  file.Close();
  const std::vector<absl::string_view> lines =
      absl::StrSplit(file_content, '\n');
  CsvData data;
  data.reserve(lines.size());
  for (absl::string_view line : lines) {
    data.push_back(absl::StrSplit(line, ','));
  }
  return data;
}

std::tuple<VectorXd, VectorXd> FindMinMaxPerColumn(const CsvData& data,
                                                   const int n_inputs) {
  VectorXd lower_bounds(n_inputs, std::numeric_limits<double>::max());
  VectorXd upper_bounds(n_inputs, std::numeric_limits<double>::lowest());
  for (const auto& record : data) {
    IHM_CHECK(record.size() > n_inputs, "Too few features per row.");
    for (int i = 0; i < n_inputs; ++i) {
      if (record[i].empty()) continue;  // missing datapoint
      double value;
      IHM_CHECK(absl::SimpleAtod(record[i], &value));
      lower_bounds[i] = std::min(lower_bounds[i], value);
      upper_bounds[i] = std::max(upper_bounds[i], value);
    }
  }
  return {lower_bounds, upper_bounds};
}

std::set<std::string> GetClasses(const CsvData& data) {
  std::set<std::string> classes;
  for (const auto& row : data) {
    std::string value =
        absl::AsciiStrToLower(absl::StripAsciiWhitespace(row.back()));
    classes.insert(std::move(value));
  }
  return classes;
}

void ChangeRandomElements(VectorXd* v, int n_changes, double new_value) {
  for (const auto& value : *v) {
    IHM_CHECK(value != new_value, "v already contains new_value");
  }
  IHM_CHECK(n_changes >= 0, "n_changes must be nonnegative.");
  IHM_CHECK(n_changes <= v->size(), "n_changes must not exceed vector size.");

  // Floyd's algorithm for sampling indices of elems to change to `base_value`.
  std::default_random_engine generator(std::random_device{}());
  for (int j = v->size() - n_changes; j < v->size(); ++j) {
    std::uniform_int_distribution<> dist(0, j);
    int k = dist(generator);
    if ((*v)[k] != new_value) {
      (*v)[k] = new_value;
    } else {
      (*v)[j] = new_value;
    }
  }
}

}  // namespace internal

absl::string_view ProblemTypeToString(ProblemType problem_type) {
  switch (problem_type) {
    case ProblemType::XOR:
      return "xor";
    case ProblemType::OR:
      return "or";
    case ProblemType::AND:
      return "and";
    case ProblemType::CIRCLE:
      return "circle";
    case ProblemType::CSV:
      return "csv";
    case ProblemType::MNIST:
      return "mnist";
    case ProblemType::MNIST_BIPOLAR:
      return "mnist2";
    case ProblemType::MNIST_INV:
      return "mnist_inv";
    case ProblemType::MNIST_AE:
      return "mnist_ae";
    case ProblemType::MNIST_AE_INV:
      return "mnist_ae_inv";
    case ProblemType::CIFAR10:
      return "cifar10";
  }
}

ProblemType ParseProblemType(absl::string_view problem_string) {
  std::string trimmed_lowercase_problem =
      absl::AsciiStrToLower(absl::StripAsciiWhitespace(problem_string));
  if (trimmed_lowercase_problem == "and") return ProblemType::AND;
  if (trimmed_lowercase_problem == "or") return ProblemType::OR;
  if (trimmed_lowercase_problem == "xor") return ProblemType::XOR;
  if (trimmed_lowercase_problem == "circle") return ProblemType::CIRCLE;
  if (trimmed_lowercase_problem == "csv") return ProblemType::CSV;
  if (trimmed_lowercase_problem == "mnist") return ProblemType::MNIST;
  if (trimmed_lowercase_problem == "mnist2") return ProblemType::MNIST_BIPOLAR;
  if (trimmed_lowercase_problem == "mnist_inv") return ProblemType::MNIST_INV;
  if (trimmed_lowercase_problem == "mnist_ae") return ProblemType::MNIST_AE;
  if (trimmed_lowercase_problem == "mnist_ae_inv")
    return ProblemType::MNIST_AE_INV;
  if (trimmed_lowercase_problem == "cifar10") return ProblemType::CIFAR10;
  IHM_LOG(LogSeverity::FATAL,
          "Unknown problem type (allowed: and, or, xor, circle, csv, "
          "mnist, mnist2, mnist_ae, cifar10).");
}

std::vector<Example> SpikingProblem::GenerateSpikingProblemCircleExamples(
    const int n_examples) {
  IHM_CHECK(n_examples >= 0, "Number of examples must be nonnegative");

  // Generate examples from class 1 inside a circle with radius `inner_radius`,
  // and from class 2 on a circular boundary between `outer_radius_min` and
  // `outer_radius_max`. Both circles are centered at `circle_centre`.
  // The distribution of points is Gaussian in the inner circle with
  // s.d. = `inner_sd` and uniform in the outer circle.
  // All points will have coordinates between 0 and 1.
  const std::tuple<double, double> circle_centre(0.5, 0.5);
  const double inner_radius = 0.3;
  const double outer_radius_min = 0.4;
  const double outer_radius_max = 0.5;

  std::default_random_engine generator(std::random_device{}());
  std::vector<Example> examples(n_examples);

  // To sample uniformly from the circle area, we use points with coordinates of
  // the form (sqrt(u) * cos(angle), sqrt(u) * sin(angle)), with uniformly
  // distributed u = r^2.
  std::bernoulli_distribution generate_on_inner_dist(0.5);
  std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
  std::uniform_real_distribution<double> inner_circle_length_dist(
      0.0, inner_radius * inner_radius);
  std::uniform_real_distribution<double> outer_circle_length_dist(
      outer_radius_min * outer_radius_min, outer_radius_max * outer_radius_max);

  for (auto& example : examples) {
    const bool generate_on_inner = generate_on_inner_dist(generator);

    const double angle = angle_dist(generator);
    double length = 0;
    if (generate_on_inner) {  // generate a point in the inner circle
      length = std::sqrt(inner_circle_length_dist(generator));
      example.targets = {1.0, 0.0};
    } else {  // generate a point in the outer circle
      length = std::sqrt(outer_circle_length_dist(generator));
      example.targets = {0.0, 1.0};
    }
    example.inputs = {length * cos(angle) + std::get<0>(circle_centre),
                      length * sin(angle) + std::get<1>(circle_centre)};
  }
  return ScaleExamplesFrom01ToInputRange(examples, input_range_);
}

std::vector<Example> SpikingProblem::GenerateSpikingProblemLogicExamples(
    const int n_examples, const ProblemType problem, const int n_inputs) {
  IHM_CHECK((problem == ProblemType::AND || problem == ProblemType::OR ||
             problem == ProblemType::XOR),
            "Invalid problem type");
  IHM_CHECK(n_inputs > 1, "Number of inputs per example must be at least 2");

  std::default_random_engine generator(std::random_device{}());
  std::vector<Example> examples(n_examples);
  std::bernoulli_distribution class_distribution(0.5);
  for (auto& example : examples) {
    // Inputs consist of true and false elements before adding noise.
    // The numbers are chosen so that the final elements are all between 0
    // and 1.
    double true_val = 0.55;
    double false_val = 0.0;
    double base_value = (problem == ProblemType::AND) ? true_val : false_val;
    example.inputs.assign(n_inputs, base_value);
    int n_changes;  // number of elements to change in the inputs
    // Generate data with equal class probabilities.
    if (class_distribution(generator)) {
      // Class 1:
      // AND: all true.
      // OR: all false.
      // XOR: even number of non-base_inputs.
      if (problem == ProblemType::XOR) {
        std::uniform_int_distribution<> changes_dist(0, n_inputs / 2);
        n_changes = changes_dist(generator) * 2;
      } else {
        n_changes = 0;
      }

      example.targets = {0.0, 1.0};
    } else {
      // Class 2:
      // AND: at least one false.
      // OR: at least one true.
      // (with equal probability for the number of non-base_value inputs).
      // XOR: odd number of non-base_inputs.
      if (problem == ProblemType::XOR) {
        std::uniform_int_distribution<> changes_dist(0, (n_inputs + 1) / 2 - 1);
        n_changes = changes_dist(generator) * 2 + 1;
      } else {
        std::uniform_int_distribution<> changes_dist(1, n_inputs);
        n_changes = changes_dist(generator);
      }
      example.targets = {1.0, 0.0};
    }
    internal::ChangeRandomElements(&example.inputs, n_changes,
                                   true_val - base_value);
    // Add noise.
    std::uniform_real_distribution<double> noise_dist(0.0, 0.45);
    for (int i = 0; i < example.inputs.size(); ++i) {
      example.inputs[i] += noise_dist(generator);
    }
  }
  return ScaleExamplesFrom01ToInputRange(examples, input_range_);
}

VectorXd ValueToOneHot(int index, int max_index) {
  VectorXd result(max_index, 0.0);
  result[index] = 1.0;
  return result;
}

VectorXd ValueToInvertedOneHot(int index, int max_index) {
  VectorXd result(max_index, 1.0);
  result[index] = 0.0;
  return result;
}

int SpikingProblem::InitialiseNInputs(const int n_inputs) {
  IHM_CHECK(n_inputs > 1);
  int expected_n_inputs;
  switch (problem_type_) {
    case ProblemType::AND:
    case ProblemType::OR:
    case ProblemType::XOR:
      IHM_CHECK(n_inputs > 1,
                "Number of inputs per example must be at least 2");
      return n_inputs;
    case ProblemType::CIRCLE:
      expected_n_inputs = 2;
      if (n_inputs != expected_n_inputs)
        IHM_LOG(
            LogSeverity::WARNING,
            absl::StrFormat(
                "Ignoring n_inputs flag for circle problem. Using %d inputs.",
                expected_n_inputs));
      return expected_n_inputs;
    case ProblemType::CSV:
      IHM_CHECK(n_inputs >= 1);
      return n_inputs;
    case ProblemType::MNIST:
    case ProblemType::MNIST_INV:
    case ProblemType::MNIST_AE:
    case ProblemType::MNIST_AE_INV:
      expected_n_inputs =
          (28 - conv_edge_length_ + 1) * (28 - conv_edge_length_ + 1);
      if (n_inputs != expected_n_inputs)
        IHM_LOG(
            LogSeverity::WARNING,
            absl::StrFormat(
                "Ignoring n_inputs flag for MNIST problem. Using %d inputs.",
                expected_n_inputs));
      return expected_n_inputs;
    case ProblemType::MNIST_BIPOLAR:
      expected_n_inputs =
          2 * (28 - conv_edge_length_ + 1) * (28 - conv_edge_length_ + 1);
      if (n_inputs != expected_n_inputs)
        IHM_LOG(LogSeverity::WARNING,
                absl::StrFormat("Ignoring n_inputs flag for MNIST "
                                "bipolar problem. Using %d inputs.",
                                expected_n_inputs));
      return expected_n_inputs;
    case ProblemType::CIFAR10:
      expected_n_inputs =
          (32 - conv_edge_length_ + 1) * (32 - conv_edge_length_ + 1);
      if (n_inputs != expected_n_inputs)
        IHM_LOG(
            LogSeverity::WARNING,
            absl::StrFormat(
                "Ignoring n_inputs flag for CIFAR10 problem. Using %d inputs.",
                expected_n_inputs));
      return expected_n_inputs;
  }
}

int SpikingProblem::InitialiseNOutputs(int n_outputs) {
  switch (problem_type_) {
    case ProblemType::AND:
    case ProblemType::OR:
    case ProblemType::XOR:
    case ProblemType::CIRCLE:
      if (n_outputs != 2)
        IHM_LOG(LogSeverity::WARNING,
                "Ignoring n_outputs for and/or/xor/circle problem.");
      return 2;
    case ProblemType::CSV:
      IHM_CHECK(n_outputs > 1);
      return n_outputs;
    case ProblemType::MNIST:
    case ProblemType::MNIST_BIPOLAR:
    case ProblemType::MNIST_INV:
      IHM_CHECK(n_outputs >= 2 && n_outputs <= 10,
                "Number of outputs for MNIST must be between 2 and 10.");
      return n_outputs;
    case ProblemType::MNIST_AE:
    case ProblemType::MNIST_AE_INV:
      return 28 * 28;
    case ProblemType::CIFAR10:
      IHM_CHECK(n_outputs == 10, "Number of outputs for CIFAR10 must be 10.");
      return n_outputs;
  }
}

int SpikingProblem::InitialiseNDigits(int n_outputs) {
  switch (problem_type_) {
    case ProblemType::AND:
    case ProblemType::OR:
    case ProblemType::XOR:
    case ProblemType::CIRCLE:
    case ProblemType::CSV:
    case ProblemType::CIFAR10:
      return -1;
    case ProblemType::MNIST:
    case ProblemType::MNIST_BIPOLAR:
    case ProblemType::MNIST_INV:
      IHM_CHECK(n_outputs >= 2 && n_outputs <= 10,
                "Number of outputs for MNIST must be between 2 and 10.");
      return n_outputs;
    case ProblemType::MNIST_AE:
    case ProblemType::MNIST_AE_INV:
      IHM_CHECK(n_outputs >= 1 && n_outputs <= 10,
                "Number of outputs for MNIST_AE* must be between 0 and 10.");
      return n_outputs;
  }
}

void SpikingProblem::LoadExamplesIfRequired(const bool should_shuffle) {
  if (loaded_train_examples_.empty()) {
    switch (problem_type_) {
      case ProblemType::CIFAR10:
        loaded_train_examples_ = LoadCifar10InSpikingFormat(
            cifar10_data_path_,
            {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
             "data_batch_4.bin", "data_batch_5.bin"});
        loaded_test_examples_ =
            LoadCifar10InSpikingFormat(cifar10_data_path_, {"test_batch.bin"});
        return;
      case ProblemType::MNIST:
      case ProblemType::MNIST_BIPOLAR:
      case ProblemType::MNIST_INV:
      case ProblemType::MNIST_AE:
      case ProblemType::MNIST_AE_INV:
        loaded_train_examples_ = LoadMNistInSpikingFormat(
            file::JoinPath(mnist_data_path_, mnist_train_images_filename_),
            file::JoinPath(mnist_data_path_, mnist_train_labels_filename_));
        loaded_test_examples_ = LoadMNistInSpikingFormat(
            file::JoinPath(mnist_data_path_, mnist_test_images_filename_),
            file::JoinPath(mnist_data_path_, mnist_test_labels_filename_));
        break;
      case ProblemType::CSV:
        loaded_train_examples_ = LoadCsvInSpikingFormat();
        loaded_test_examples_.clear();
        break;
      case ProblemType::AND:
      case ProblemType::OR:
      case ProblemType::XOR:
      case ProblemType::CIRCLE:
        // Do nothing, for these problems we're only generating data on the fly.
        break;
    }
  }
  if (should_shuffle) {
    std::default_random_engine generator(std::random_device{}());
    std::shuffle(loaded_train_examples_.begin(), loaded_train_examples_.end(),
                 generator);
  }
}

void SpikingProblem::GenerateExamplesWithSeparateTest(
    const int n_train, const int n_validation, const int n_test,
    const bool should_shuffle) {
  LoadExamplesIfRequired(should_shuffle);
  IHM_CHECK(n_train + n_validation <= loaded_train_examples_.size());
  IHM_CHECK(n_test <= loaded_test_examples_.size());

  train_examples_.assign(loaded_train_examples_.begin(),
                         loaded_train_examples_.begin() + n_train);
  validation_examples_.assign(
      loaded_train_examples_.begin() + n_train,
      loaded_train_examples_.begin() + n_train + n_validation);
  test_examples_.assign(loaded_test_examples_.begin(),
                        loaded_test_examples_.begin() + n_test);
}

void SpikingProblem::GenerateExamplesFromSingleSet(const int n_train,
                                                   const int n_validation,
                                                   const int n_test,
                                                   const bool should_shuffle) {
  LoadExamplesIfRequired(should_shuffle);
  IHM_CHECK(n_train + n_validation + n_test <= loaded_train_examples_.size());
  train_examples_.assign(loaded_train_examples_.begin(),
                         loaded_train_examples_.begin() + n_train);
  validation_examples_.assign(
      loaded_train_examples_.begin() + n_train,
      loaded_train_examples_.begin() + n_train + n_validation);
  test_examples_.assign(
      loaded_train_examples_.begin() + n_train + n_validation,
      loaded_train_examples_.begin() + n_train + n_validation + n_test);
}

void SpikingProblem::GenerateExamples(const int n_train, const int n_validation,
                                      const int n_test,
                                      const bool should_shuffle) {
  IHM_CHECK(n_train >= 0);
  IHM_CHECK(n_validation >= 0);
  IHM_CHECK(n_test >= 0);

  switch (problem_type_) {
    case ProblemType::AND:
    case ProblemType::OR:
    case ProblemType::XOR:
      train_examples_ = GenerateSpikingProblemLogicExamples(
          n_train, problem_type_, n_inputs_);
      validation_examples_ = GenerateSpikingProblemLogicExamples(
          n_validation, problem_type_, n_inputs_);
      test_examples_ =
          GenerateSpikingProblemLogicExamples(n_test, problem_type_, n_inputs_);
      break;
    case ProblemType::CIRCLE:
      train_examples_ = GenerateSpikingProblemCircleExamples(n_train);
      validation_examples_ = GenerateSpikingProblemCircleExamples(n_validation);
      test_examples_ = GenerateSpikingProblemCircleExamples(n_test);
      break;
    case ProblemType::CSV:
      GenerateExamplesFromSingleSet(n_train, n_validation, n_test,
                                    should_shuffle);
      break;
    case ProblemType::MNIST:
    case ProblemType::MNIST_BIPOLAR:
    case ProblemType::MNIST_INV:
    case ProblemType::MNIST_AE:
    case ProblemType::MNIST_AE_INV:
      GenerateExamplesWithSeparateTest(n_train, n_validation, n_test,
                                       should_shuffle);
      break;
    case ProblemType::CIFAR10:
      GenerateExamplesWithSeparateTest(n_train, n_validation, n_test,
                                       should_shuffle);
      break;
  }
}

void SpikingProblem::GeneratePartitionedExamples(
    const int n_train, const int n_test, const int validation_index_low,
    const int validation_index_high, const bool should_shuffle) {
  IHM_CHECK(n_train >= 0);
  IHM_CHECK(n_test >= 0);
  IHM_CHECK(validation_index_low >= 0);
  IHM_CHECK(validation_index_low <= validation_index_high);
  IHM_CHECK(validation_index_high <= n_train);
  IHM_CHECK(problem_type_ == ProblemType::MNIST ||
                problem_type_ == ProblemType::MNIST_BIPOLAR ||
                problem_type_ == ProblemType::MNIST_INV ||
                problem_type_ == ProblemType::CSV ||
                problem_type_ == ProblemType::CIFAR10,
            "Example partitioning only works with MNIST, MNIST2, CSV, "
            "CIFAR10.");

  LoadExamplesIfRequired(should_shuffle);

  IHM_CHECK(n_train <= loaded_train_examples_.size());
  IHM_CHECK(n_test <= loaded_test_examples_.size());

  train_examples_.assign(loaded_train_examples_.begin(),
                         loaded_train_examples_.begin() + validation_index_low);
  validation_examples_.assign(
      loaded_train_examples_.begin() + validation_index_low,
      loaded_train_examples_.begin() + validation_index_high);
  train_examples_.insert(train_examples().end(),
                         loaded_train_examples_.begin() + validation_index_high,
                         loaded_train_examples_.begin() + n_train);
  test_examples_.assign(loaded_test_examples_.begin(),
                        loaded_test_examples_.begin() + n_test);
}

std::vector<Example> SpikingProblem::LoadCsvInSpikingFormat() {
  std::vector<Example> examples;
  const auto data = internal::LoadCsvData(csv_data_path_);

  // Automatically find classes from the last element of each row.
  const std::set<std::string> target_classes = internal::GetClasses(data);
  IHM_CHECK(target_classes.size() == n_outputs_);

  // Automatically find values for scaling features between 0 and 1.
  const auto [lower_bounds, upper_bounds] =
      internal::FindMinMaxPerColumn(data, n_inputs_);

  for (const auto& record : data) {
    IHM_CHECK(record.size() == (n_inputs_ + 1), "Wrong number of features.");
    Example example;

    // The first elements of a record are input features.
    example.inputs.assign(n_inputs_, 0.0);
    for (int i = 0; i < n_inputs_; ++i) {
      if (record[i].empty()) {  // missing datapoint
        example.inputs[i] = Tempcoder::kNoSpike;
        continue;
      }
      double value;
      IHM_CHECK(absl::SimpleAtod(record[i], &value));
      example.inputs[i] =
          (value - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i]);
    }

    // Create one-hot-encoded target vector.
    const std::string class_string =
        absl::AsciiStrToLower(absl::StripAsciiWhitespace(record.back()));
    if (target_classes.count(class_string) == 0) {
      IHM_LOG(LogSeverity::FATAL, "Wrong class match.");
    }
    example.targets.reserve(target_classes.size());
    for (const std::string& target : target_classes) {
      example.targets.push_back(target == class_string ? 1 : 0);
    }
    examples.push_back(example);
  }
  return ScaleExamplesFrom01ToInputRange(examples, input_range_);
}

void AppendConvLikeInputs(const std::vector<float>& data,
                          const ConvWindowParameters& params,
                          std::vector<Example>* examples) {
  IHM_CHECK(data.size() == params.input_edge_length * params.input_edge_length *
                               examples->size());
  IHM_CHECK(params.input_edge_length >= 0);
  IHM_CHECK(params.conv_edge_length >= 0);
  IHM_CHECK(params.conv_edge_length <= params.input_edge_length);
  IHM_CHECK(params.sliding_step > 0);

  const int input_size = params.input_edge_length * params.input_edge_length;

  std::vector<float> window;
  window.reserve(params.conv_edge_length * params.conv_edge_length);

  for (int example_ind = 0; example_ind < examples->size(); ++example_ind) {
    // For each possible start of a 2D window...
    for (int conv_row_start = 0;
         conv_row_start <
         params.input_edge_length - params.conv_edge_length + 1;
         conv_row_start += params.sliding_step) {
      for (int conv_col_start = 0;
           conv_col_start <
           params.input_edge_length - params.conv_edge_length + 1;
           conv_col_start += params.sliding_step) {
        // ...take a square window with length `conv_edge_length`...
        for (int row = 0; row < params.conv_edge_length; ++row) {
          for (int col = 0; col < params.conv_edge_length; ++col) {
            int absolute_index =
                example_ind * input_size +
                (conv_row_start + row) * params.input_edge_length +
                conv_col_start + col;
            IHM_CHECK(data[absolute_index] >= 0.0);
            IHM_CHECK(data[absolute_index] <= 1.0);
            window.push_back(params.invert ? 1.0 - data[absolute_index]
                                           : data[absolute_index]);
          }
        }
        // ...and append the min/max over this window to this example's inputs.
        double input = params.invert
                           ? *std::min_element(window.begin(), window.end())
                           : *std::max_element(window.begin(), window.end());
        if (input == 1.0) input = Tempcoder::kNoSpike;
        (*examples)[example_ind].inputs.push_back(input);
        window.clear();
      }
    }
  }
}

std::vector<Example> SpikingProblem::LoadMNistInSpikingFormat(
    const std::string& data_path, const std::string& labels_path) {
  int n_examples;
  int n_inputs;
  std::vector<float> mnist_all_data_float;
  std::vector<int> mnist_labels;
  IHM_LOG(LogSeverity::INFO, "Loading MNIST data...");
  LoadMNISTDataAndLabels(data_path, labels_path, &n_inputs, &n_examples,
                         &mnist_all_data_float, &mnist_labels);
  constexpr int kMnistEdgeLength = 28;
  IHM_CHECK(n_inputs == kMnistEdgeLength * kMnistEdgeLength);

  std::vector<Example> examples;
  std::vector<float> mnist_data_float;  // stores only the examples we want
  for (int i = 0; i < n_examples; ++i) {
    // Only keep examples labelled up to `n_digits_` digits.
    if (mnist_labels[i] >= n_digits_) continue;
    Example example;
    example.original_class = mnist_labels[i];
    example.targets = ValueToOneHot(mnist_labels[i], n_digits_);
    examples.push_back(example);
    mnist_data_float.insert(mnist_data_float.end(),
                            mnist_all_data_float.begin() + i * n_inputs,
                            mnist_all_data_float.begin() + (i + 1) * n_inputs);
  }

  ConvWindowParameters params;
  params.input_edge_length = kMnistEdgeLength;
  params.conv_edge_length = conv_edge_length_;
  params.sliding_step = 1;
  params.invert = true;
  if (problem_type_ != ProblemType::MNIST_INV &&
      problem_type_ != ProblemType::MNIST_AE_INV) {
    AppendConvLikeInputs(mnist_data_float, params, &examples);
  }
  if (problem_type_ == ProblemType::MNIST_BIPOLAR ||
      problem_type_ == ProblemType::MNIST_INV ||
      problem_type_ == ProblemType::MNIST_AE_INV) {
    params.invert = false;
    AppendConvLikeInputs(mnist_data_float, params, &examples);
  }

  if (problem_type_ == ProblemType::MNIST_AE ||
      problem_type_ == ProblemType::MNIST_AE_INV) {
    for (Example& example : examples) example.targets = example.inputs;
  }

  IHM_LOG(LogSeverity::INFO, "Done loading MNIST data.");
  return ScaleExamplesFrom01ToInputRange(
      AddNoise(examples, noise_factor_, noisy_targets_), input_range_,
      /*also_scale_targets=*/problem_type_ == ProblemType::MNIST_AE ||
          problem_type_ == ProblemType::MNIST_AE_INV);
}

constexpr size_t kCifar10RecordLen = 3073;
constexpr size_t kCifarEdgeLen = 32;

void LoadSingleCifar10File(absl::string_view file_path,
                           std::vector<float>* inputs,
                           std::vector<int>* labels) {
  constexpr size_t kSamplesPerRead = 1000;
  constexpr size_t kChannelLen = kCifarEdgeLen * kCifarEdgeLen;

  file::IhmFile fp;
  IHM_CHECK(file::Open(file_path, "r", &fp));

  std::string buffer;

  bool status = fp.Read(kSamplesPerRead * kCifar10RecordLen, &buffer);

  while (status) {
    if (!buffer.empty()) {
      const uint8_t* sample_bytes =
          reinterpret_cast<const uint8_t*>(buffer.c_str());
      IHM_CHECK((buffer.size() % kCifar10RecordLen) == 0);
      const int num_samples = buffer.size() / kCifar10RecordLen;

      for (int i = 0; i < num_samples; ++i) {
        const uint8_t* example = sample_bytes + i * kCifar10RecordLen;
        const int label = example[0];
        labels->push_back(label);

        for (size_t j = 0; j < kChannelLen; ++j) {
          // Read RGB values from input
          float r = example[1 + j];
          float g = example[1 + kChannelLen + j];
          float b = example[1 + 2 * kChannelLen + j];

          // Standard linear RGB to grayscale conversion, with scaling to 0-1
          // range.
          // Y = 0.2126 * R + 0.7152 * G + 0.0722 * B.
          const float intensity =
              (0.2126f * r + 0.7152f * g + 0.0722 * b) / 255.0f;
          inputs->push_back(intensity);
        }
      }

      status = fp.Read(kSamplesPerRead * kCifar10RecordLen, &buffer);
    }
  }

  IHM_CHECK(fp.Close());
}

std::vector<Example> SpikingProblem::LoadCifar10InSpikingFormat(
    absl::string_view cifar_base_path,
    const std::vector<absl::string_view>& files_to_load) {
  constexpr size_t kChannelLen = kCifarEdgeLen * kCifarEdgeLen;
  std::vector<float> data_float;
  std::vector<int> labels;

  IHM_LOG(LogSeverity::INFO, "Loading CIFAR10 data...");

  for (absl::string_view file : files_to_load) {
    LoadSingleCifar10File(file::JoinPath(cifar_base_path, file), &data_float,
                          &labels);
  }

  IHM_CHECK(data_float.size() == kChannelLen * labels.size());

  std::vector<Example> examples(labels.size());
  for (int i = 0; i < labels.size(); i++) {
    examples[i].original_class = labels[i];
    examples[i].targets = ValueToOneHot(labels[i], /*max_index=*/10);
  }

  ConvWindowParameters params;
  params.input_edge_length = kCifarEdgeLen;
  params.conv_edge_length = conv_edge_length_;
  params.sliding_step = 1;
  params.invert = true;
  AppendConvLikeInputs(data_float, params, &examples);

  IHM_LOG(LogSeverity::INFO, "Done loading CIFAR10 data.");
  return ScaleExamplesFrom01ToInputRange(examples, input_range_);
}

std::vector<Example> ScaleExamplesFrom01ToInputRange(
    std::vector<Example> examples, const std::pair<double, double>& range,
    bool also_scale_targets) {
  for (auto& example : examples) {
    for (auto& input : example.inputs) {
      if (input == Tempcoder::kNoSpike) continue;
      input = (input * (range.second - range.first) + range.first);
    }
    if (also_scale_targets) {
      for (auto& target : example.targets) {
        if (target == Tempcoder::kNoSpike) continue;
        target = (target * (range.second - range.first) + range.first);
      }
    }
  }
  return examples;
}

std::vector<Example> AddNoise(std::vector<Example> examples,
                              const double noise_factor,
                              const bool noisy_targets) {
  std::default_random_engine generator(std::random_device{}());
  std::normal_distribution<double> distribution(1.0, 1.0);
  for (auto& example : examples) {
    for (auto& input : example.inputs) {
      double original_input = input == Tempcoder::kNoSpike ? 1.0 : input;
      double noisy_input = std::clamp(
          original_input + noise_factor * distribution(generator), 0.0, 1.0);
      input = noisy_input >= 1.0 ? Tempcoder::kNoSpike : noisy_input;
    }
    if (noisy_targets) example.targets = example.inputs;
  }
  return examples;
}

}  // namespace ihmehimmeli
