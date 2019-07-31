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
#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "tempcoding/file_passthrough.h"
#include "tempcoding/mnist_loader.h"
#include "tempcoding/tempcoder.h"
#include "tempcoding/util.h"

namespace ihmehimmeli {

namespace {
// All Iris examples.
std::vector<std::vector<std::string>> iris_data_ = {
    {"5.1", "3.5", "1.4", "0.2", "Iris-setosa"},
    {"4.9", "3.0", "1.4", "0.2", "Iris-setosa"},
    {"4.7", "3.2", "1.3", "0.2", "Iris-setosa"},
    {"4.6", "3.1", "1.5", "0.2", "Iris-setosa"},
    {"5.0", "3.6", "1.4", "0.2", "Iris-setosa"},
    {"5.4", "3.9", "1.7", "0.4", "Iris-setosa"},
    {"4.6", "3.4", "1.4", "0.3", "Iris-setosa"},
    {"5.0", "3.4", "1.5", "0.2", "Iris-setosa"},
    {"4.4", "2.9", "1.4", "0.2", "Iris-setosa"},
    {"4.9", "3.1", "1.5", "0.1", "Iris-setosa"},
    {"5.4", "3.7", "1.5", "0.2", "Iris-setosa"},
    {"4.8", "3.4", "1.6", "0.2", "Iris-setosa"},
    {"4.8", "3.0", "1.4", "0.1", "Iris-setosa"},
    {"4.3", "3.0", "1.1", "0.1", "Iris-setosa"},
    {"5.8", "4.0", "1.2", "0.2", "Iris-setosa"},
    {"5.7", "4.4", "1.5", "0.4", "Iris-setosa"},
    {"5.4", "3.9", "1.3", "0.4", "Iris-setosa"},
    {"5.1", "3.5", "1.4", "0.3", "Iris-setosa"},
    {"5.7", "3.8", "1.7", "0.3", "Iris-setosa"},
    {"5.1", "3.8", "1.5", "0.3", "Iris-setosa"},
    {"5.4", "3.4", "1.7", "0.2", "Iris-setosa"},
    {"5.1", "3.7", "1.5", "0.4", "Iris-setosa"},
    {"4.6", "3.6", "1.0", "0.2", "Iris-setosa"},
    {"5.1", "3.3", "1.7", "0.5", "Iris-setosa"},
    {"4.8", "3.4", "1.9", "0.2", "Iris-setosa"},
    {"5.0", "3.0", "1.6", "0.2", "Iris-setosa"},
    {"5.0", "3.4", "1.6", "0.4", "Iris-setosa"},
    {"5.2", "3.5", "1.5", "0.2", "Iris-setosa"},
    {"5.2", "3.4", "1.4", "0.2", "Iris-setosa"},
    {"4.7", "3.2", "1.6", "0.2", "Iris-setosa"},
    {"4.8", "3.1", "1.6", "0.2", "Iris-setosa"},
    {"5.4", "3.4", "1.5", "0.4", "Iris-setosa"},
    {"5.2", "4.1", "1.5", "0.1", "Iris-setosa"},
    {"5.5", "4.2", "1.4", "0.2", "Iris-setosa"},
    {"4.9", "3.1", "1.5", "0.1", "Iris-setosa"},
    {"5.0", "3.2", "1.2", "0.2", "Iris-setosa"},
    {"5.5", "3.5", "1.3", "0.2", "Iris-setosa"},
    {"4.9", "3.1", "1.5", "0.1", "Iris-setosa"},
    {"4.4", "3.0", "1.3", "0.2", "Iris-setosa"},
    {"5.1", "3.4", "1.5", "0.2", "Iris-setosa"},
    {"5.0", "3.5", "1.3", "0.3", "Iris-setosa"},
    {"4.5", "2.3", "1.3", "0.3", "Iris-setosa"},
    {"4.4", "3.2", "1.3", "0.2", "Iris-setosa"},
    {"5.0", "3.5", "1.6", "0.6", "Iris-setosa"},
    {"5.1", "3.8", "1.9", "0.4", "Iris-setosa"},
    {"4.8", "3.0", "1.4", "0.3", "Iris-setosa"},
    {"5.1", "3.8", "1.6", "0.2", "Iris-setosa"},
    {"4.6", "3.2", "1.4", "0.2", "Iris-setosa"},
    {"5.3", "3.7", "1.5", "0.2", "Iris-setosa"},
    {"5.0", "3.3", "1.4", "0.2", "Iris-setosa"},
    {"7.0", "3.2", "4.7", "1.4", "Iris-versicolor"},
    {"6.4", "3.2", "4.5", "1.5", "Iris-versicolor"},
    {"6.9", "3.1", "4.9", "1.5", "Iris-versicolor"},
    {"5.5", "2.3", "4.0", "1.3", "Iris-versicolor"},
    {"6.5", "2.8", "4.6", "1.5", "Iris-versicolor"},
    {"5.7", "2.8", "4.5", "1.3", "Iris-versicolor"},
    {"6.3", "3.3", "4.7", "1.6", "Iris-versicolor"},
    {"4.9", "2.4", "3.3", "1.0", "Iris-versicolor"},
    {"6.6", "2.9", "4.6", "1.3", "Iris-versicolor"},
    {"5.2", "2.7", "3.9", "1.4", "Iris-versicolor"},
    {"5.0", "2.0", "3.5", "1.0", "Iris-versicolor"},
    {"5.9", "3.0", "4.2", "1.5", "Iris-versicolor"},
    {"6.0", "2.2", "4.0", "1.0", "Iris-versicolor"},
    {"6.1", "2.9", "4.7", "1.4", "Iris-versicolor"},
    {"5.6", "2.9", "3.6", "1.3", "Iris-versicolor"},
    {"6.7", "3.1", "4.4", "1.4", "Iris-versicolor"},
    {"5.6", "3.0", "4.5", "1.5", "Iris-versicolor"},
    {"5.8", "2.7", "4.1", "1.0", "Iris-versicolor"},
    {"6.2", "2.2", "4.5", "1.5", "Iris-versicolor"},
    {"5.6", "2.5", "3.9", "1.1", "Iris-versicolor"},
    {"5.9", "3.2", "4.8", "1.8", "Iris-versicolor"},
    {"6.1", "2.8", "4.0", "1.3", "Iris-versicolor"},
    {"6.3", "2.5", "4.9", "1.5", "Iris-versicolor"},
    {"6.1", "2.8", "4.7", "1.2", "Iris-versicolor"},
    {"6.4", "2.9", "4.3", "1.3", "Iris-versicolor"},
    {"6.6", "3.0", "4.4", "1.4", "Iris-versicolor"},
    {"6.8", "2.8", "4.8", "1.4", "Iris-versicolor"},
    {"6.7", "3.0", "5.0", "1.7", "Iris-versicolor"},
    {"6.0", "2.9", "4.5", "1.5", "Iris-versicolor"},
    {"5.7", "2.6", "3.5", "1.0", "Iris-versicolor"},
    {"5.5", "2.4", "3.8", "1.1", "Iris-versicolor"},
    {"5.5", "2.4", "3.7", "1.0", "Iris-versicolor"},
    {"5.8", "2.7", "3.9", "1.2", "Iris-versicolor"},
    {"6.0", "2.7", "5.1", "1.6", "Iris-versicolor"},
    {"5.4", "3.0", "4.5", "1.5", "Iris-versicolor"},
    {"6.0", "3.4", "4.5", "1.6", "Iris-versicolor"},
    {"6.7", "3.1", "4.7", "1.5", "Iris-versicolor"},
    {"6.3", "2.3", "4.4", "1.3", "Iris-versicolor"},
    {"5.6", "3.0", "4.1", "1.3", "Iris-versicolor"},
    {"5.5", "2.5", "4.0", "1.3", "Iris-versicolor"},
    {"5.5", "2.6", "4.4", "1.2", "Iris-versicolor"},
    {"6.1", "3.0", "4.6", "1.4", "Iris-versicolor"},
    {"5.8", "2.6", "4.0", "1.2", "Iris-versicolor"},
    {"5.0", "2.3", "3.3", "1.0", "Iris-versicolor"},
    {"5.6", "2.7", "4.2", "1.3", "Iris-versicolor"},
    {"5.7", "3.0", "4.2", "1.2", "Iris-versicolor"},
    {"5.7", "2.9", "4.2", "1.3", "Iris-versicolor"},
    {"6.2", "2.9", "4.3", "1.3", "Iris-versicolor"},
    {"5.1", "2.5", "3.0", "1.1", "Iris-versicolor"},
    {"5.7", "2.8", "4.1", "1.3", "Iris-versicolor"},
    {"6.3", "3.3", "6.0", "2.5", "Iris-virginica"},
    {"5.8", "2.7", "5.1", "1.9", "Iris-virginica"},
    {"7.1", "3.0", "5.9", "2.1", "Iris-virginica"},
    {"6.3", "2.9", "5.6", "1.8", "Iris-virginica"},
    {"6.5", "3.0", "5.8", "2.2", "Iris-virginica"},
    {"7.6", "3.0", "6.6", "2.1", "Iris-virginica"},
    {"4.9", "2.5", "4.5", "1.7", "Iris-virginica"},
    {"7.3", "2.9", "6.3", "1.8", "Iris-virginica"},
    {"6.7", "2.5", "5.8", "1.8", "Iris-virginica"},
    {"7.2", "3.6", "6.1", "2.5", "Iris-virginica"},
    {"6.5", "3.2", "5.1", "2.0", "Iris-virginica"},
    {"6.4", "2.7", "5.3", "1.9", "Iris-virginica"},
    {"6.8", "3.0", "5.5", "2.1", "Iris-virginica"},
    {"5.7", "2.5", "5.0", "2.0", "Iris-virginica"},
    {"5.8", "2.8", "5.1", "2.4", "Iris-virginica"},
    {"6.4", "3.2", "5.3", "2.3", "Iris-virginica"},
    {"6.5", "3.0", "5.5", "1.8", "Iris-virginica"},
    {"7.7", "3.8", "6.7", "2.2", "Iris-virginica"},
    {"7.7", "2.6", "6.9", "2.3", "Iris-virginica"},
    {"6.0", "2.2", "5.0", "1.5", "Iris-virginica"},
    {"6.9", "3.2", "5.7", "2.3", "Iris-virginica"},
    {"5.6", "2.8", "4.9", "2.0", "Iris-virginica"},
    {"7.7", "2.8", "6.7", "2.0", "Iris-virginica"},
    {"6.3", "2.7", "4.9", "1.8", "Iris-virginica"},
    {"6.7", "3.3", "5.7", "2.1", "Iris-virginica"},
    {"7.2", "3.2", "6.0", "1.8", "Iris-virginica"},
    {"6.2", "2.8", "4.8", "1.8", "Iris-virginica"},
    {"6.1", "3.0", "4.9", "1.8", "Iris-virginica"},
    {"6.4", "2.8", "5.6", "2.1", "Iris-virginica"},
    {"7.2", "3.0", "5.8", "1.6", "Iris-virginica"},
    {"7.4", "2.8", "6.1", "1.9", "Iris-virginica"},
    {"7.9", "3.8", "6.4", "2.0", "Iris-virginica"},
    {"6.4", "2.8", "5.6", "2.2", "Iris-virginica"},
    {"6.3", "2.8", "5.1", "1.5", "Iris-virginica"},
    {"6.1", "2.6", "5.6", "1.4", "Iris-virginica"},
    {"7.7", "3.0", "6.1", "2.3", "Iris-virginica"},
    {"6.3", "3.4", "5.6", "2.4", "Iris-virginica"},
    {"6.4", "3.1", "5.5", "1.8", "Iris-virginica"},
    {"6.0", "3.0", "4.8", "1.8", "Iris-virginica"},
    {"6.9", "3.1", "5.4", "2.1", "Iris-virginica"},
    {"6.7", "3.1", "5.6", "2.4", "Iris-virginica"},
    {"6.9", "3.1", "5.1", "2.3", "Iris-virginica"},
    {"5.8", "2.7", "5.1", "1.9", "Iris-virginica"},
    {"6.8", "3.2", "5.9", "2.3", "Iris-virginica"},
    {"6.7", "3.3", "5.7", "2.5", "Iris-virginica"},
    {"6.7", "3.0", "5.2", "2.3", "Iris-virginica"},
    {"6.3", "2.5", "5.0", "1.9", "Iris-virginica"},
    {"6.5", "3.0", "5.2", "2.0", "Iris-virginica"},
    {"6.2", "3.4", "5.4", "2.3", "Iris-virginica"},
    {"5.9", "3.0", "5.1", "1.8", "Iris-virginica"}};

}  // namespace

namespace internal {

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
    case ProblemType::IRIS:
      return "iris";
    case ProblemType::MNIST:
      return "mnist";
    case ProblemType::MNIST_BIPOLAR:
      return "mnist2";
    case ProblemType::CIFAR10:
      return "cifar10";
  }
}

ProblemType ParseProblemType(absl::string_view problem_string) {
  std::string trimmed_lowercase_problem_string =
      absl::AsciiStrToLower(absl::StripAsciiWhitespace(problem_string));
  if (trimmed_lowercase_problem_string == "and") {
    return ProblemType::AND;
  } else if (trimmed_lowercase_problem_string == "or") {
    return ProblemType::OR;
  } else if (trimmed_lowercase_problem_string == "xor") {
    return ProblemType::XOR;
  } else if (trimmed_lowercase_problem_string == "circle") {
    return ProblemType::CIRCLE;
  } else if (trimmed_lowercase_problem_string == "iris") {
    return ProblemType::IRIS;
  } else if (trimmed_lowercase_problem_string == "mnist") {
    return ProblemType::MNIST;
  } else if (trimmed_lowercase_problem_string == "mnist2") {
    return ProblemType::MNIST_BIPOLAR;
  } else if (trimmed_lowercase_problem_string == "cifar10") {
    return ProblemType::CIFAR10;
  }
  IHM_LOG(LogSeverity::FATAL,
          "Unknown problem type (allowed: and, or, xor, circle, iris, "
          "mnist, mnist2, cifar10).");
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
    case ProblemType::IRIS:
      expected_n_inputs = 4;
      if (n_inputs != expected_n_inputs)
        IHM_LOG(LogSeverity::WARNING,
                absl::StrFormat(
                    "Ignoring n_inputs flag for Iris problem. Using %d inputs.",
                    expected_n_inputs));
      return expected_n_inputs;
    case ProblemType::MNIST:
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
    case ProblemType::IRIS:
      if (n_outputs != 2)
        IHM_LOG(LogSeverity::WARNING, "Ignoring n_outputs for Iris problem.");
      return 3;
    case ProblemType::MNIST:
    case ProblemType::MNIST_BIPOLAR:
      IHM_CHECK(n_outputs >= 2 && n_outputs <= 10,
                "Number of outputs for MNIST must be between 2 and 10.");
      return n_outputs;
    case ProblemType::CIFAR10:
      IHM_CHECK(n_outputs == 10, "Number of outputs for CIFAR10 must be 10.");
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
        loaded_train_examples_ = LoadMNistInSpikingFormat(
            file::JoinPath(mnist_data_path_, mnist_train_images_filename_),
            file::JoinPath(mnist_data_path_, mnist_train_labels_filename_));
        loaded_test_examples_ = LoadMNistInSpikingFormat(
            file::JoinPath(mnist_data_path_, mnist_test_images_filename_),
            file::JoinPath(mnist_data_path_, mnist_test_labels_filename_));
        break;
      case ProblemType::IRIS:
        loaded_train_examples_ = LoadIrisInSpikingFormat();
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

void SpikingProblem::GenerateExamplesIris(const int n_train,
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
    case ProblemType::IRIS:
      GenerateExamplesIris(n_train, n_validation, n_test, should_shuffle);
      break;
    case ProblemType::MNIST:
    case ProblemType::MNIST_BIPOLAR:
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
  IHM_CHECK(
      problem_type_ == ProblemType::MNIST ||
          problem_type_ == ProblemType::MNIST_BIPOLAR ||
          problem_type_ == ProblemType::IRIS ||
          problem_type_ == ProblemType::CIFAR10,
      "Example partitioning only works with MNIST, MNIST2, IRIS, CIFAR10.");

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

std::vector<Example> SpikingProblem::LoadIrisInSpikingFormat() {
  std::vector<Example> examples;
  IHM_CHECK(iris_data_.size() == 150, "Wrong number of iris examples");

  // Predefined values for scaling each Iris feature between 0 and 1.
  const VectorXd lower_bounds = {4.3, 2.0, 1.0, 0.1};
  const VectorXd upper_bounds = {7.9, 4.4, 6.9, 2.5};

  for (const auto& record : iris_data_) {
    IHM_CHECK(record.size() == (n_inputs_ + 1),
              "Wrong number of iris features.");
    Example example;
    // The first 4 elements of a record are input features.
    example.inputs.assign(n_inputs_, 0.0);
    for (int i = 0; i < n_inputs_; ++i) {
      IHM_CHECK(absl::SimpleAtod(record[i], &example.inputs[i]));
      example.inputs[i] = (example.inputs[i] - lower_bounds[i]) /
                          (upper_bounds[i] - lower_bounds[i]);
    }
    // The last element gives the class.
    const std::string class_string =
        absl::AsciiStrToLower(absl::StripAsciiWhitespace(record.back()));
    if (class_string == "iris-setosa") {
      example.targets = {1.0, 0.0, 0.0};
    } else if (class_string == "iris-versicolor") {
      example.targets = {0.0, 1.0, 0.0};
    } else if (class_string == "iris-virginica") {
      example.targets = {0.0, 0.0, 1.0};
    } else {
      IHM_LOG(LogSeverity::FATAL,
              absl::StrFormat("Unknown iris class: %s", record.back()));
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
    // Only keep examples labelled up to `n_outputs_` digits.
    if (mnist_labels[i] >= n_outputs_) continue;
    Example example;
    example.targets = ValueToOneHot(mnist_labels[i], n_outputs_);
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
  AppendConvLikeInputs(mnist_data_float, params, &examples);
  if (problem_type_ == ProblemType::MNIST_BIPOLAR) {
    params.invert = false;
    AppendConvLikeInputs(mnist_data_float, params, &examples);
  }

  IHM_LOG(LogSeverity::INFO, "Done loading MNIST data.");
  return ScaleExamplesFrom01ToInputRange(examples, input_range_);
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
    std::vector<Example> examples,
    const std::pair<double, double>& input_range) {
  for (auto& example : examples) {
    for (auto& input : example.inputs) {
      if (input == Tempcoder::kNoSpike) continue;
      input = (input * (input_range.second - input_range.first) +
               input_range.first);
    }
  }
  return examples;
}

}  // namespace ihmehimmeli
