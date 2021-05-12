/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Provides problem examples in spiking form and functions to assess results.

#ifndef IHMEHIMMELI_TEMPCODING_SPIKING_PROBLEM_H_
#define IHMEHIMMELI_TEMPCODING_SPIKING_PROBLEM_H_

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "common/util.h"
#include "tempcoding/tempcoder.h"

namespace ihmehimmeli {
namespace internal {

using CsvRow = std::vector<std::string>;
using CsvData = std::vector<CsvRow>;

// Loads csv data from `file_path` as a vector of rows. A row is a vector of
// strings.
CsvData LoadCsvData(const std::string& file_path);

// Computes the lowest and highest value in the given `data` per column (2nd
// dimension) for the first `n_inputs` columns. Each row must contain
// at least `n_inputs` + 1 elements (i.e. features followed by class label).
std::tuple<VectorXd, VectorXd> FindMinMaxPerColumn(const CsvData& data,
                                                   int n_inputs);

// Returns a vector of unique strings occuring as the last element of each
// vector.
std::set<std::string> GetClasses(const CsvData& data);

// Modifies `v` in-place by changing `n_changes` elements to `new_value`.
// Assumes no values in `v` are already equal to `new_value`.
void ChangeRandomElements(VectorXd* v, int n_changes, double new_value);

}  // namespace internal

// Defines a training, validation or testing example.
struct Example {
  VectorXd inputs;
  VectorXd targets;
  Prediction prediction;
  int original_class;
};

// Specifies a square convolution window applied on square inputs with edge
// length `input_edge_length`. The length of the window edge is
// `conv_edge_length` (by default 1). The window is slid over the 2D input
// with the given `sliding_step` (by default 1). The `invert` parameter
// specifies whether larger input values are more salient (spike first,
// therefore they should be inverted); by default this is true.
struct ConvWindowParameters {
  int input_edge_length = 28;  // default value for MNIST
  int conv_edge_length = 1;
  int sliding_step = 1;
  bool invert = true;
};

enum class ProblemType {
  XOR,
  OR,
  AND,
  CIRCLE,
  CSV,
  MNIST,
  MNIST_BIPOLAR,
  MNIST_INV,
  CIFAR10,
  MNIST_AE,
  MNIST_AE_INV
};

// Returns a lowercase string that represents (and can be used to parse) a
// ProblemType.
absl::string_view ProblemTypeToString(ProblemType problem_type);

// Parse a ProblemType from a string.
ProblemType ParseProblemType(absl::string_view problem_string);

// Returns a vector of `max_index` doubles with all elements set to 0, except
// for the element at position `index`, which is set to 1.
VectorXd ValueToOneHot(int index, int max_index);

// Returns a vector of `max_index` doubles with all elements set to 1, except
// for the element at position `index`, which is set to 0. This inverted one-hot
// encoding is because the network aims for the correct neuron to spike fastest.
// Example: ValueToInvertedOneHot(1, 3) = {1, 0, 1, 1}
VectorXd ValueToInvertedOneHot(int index, int max_index);

// Converts a one-hot vector to a single value representing the size of the 1.
int OneHotToValue(const VectorXd targets);

// Scale `examples` from [0, 1] to `input_range`. No-spikes are left unchanged.
std::vector<Example> ScaleExamplesFrom01ToInputRange(
    const std::vector<Example> examples, const std::pair<double, double>& range,
    bool also_scale_targets = false);

// Adds random Gaussian noise (mean=1.0, std=1.0, i.e. later spikes)
// with strength `noise_factor` to pixel values of the `inputs` field of
// `examples`, then clamps between 0.0 and 1.0 (with 1.0 turned into
// kNoSpike). If `noisy_targets` is true, replaces the `targets` field of
// `examples` with the `inputs` field. Assumes `inputs` are between 0.0 and 1.0;
// kNoSpike is interpreted as 1.0.
std::vector<Example> AddNoise(std::vector<Example> examples,
                              double noise_factor = 0.0,
                              bool noisy_targets = false);

// Appends to each of the ordered `examples` a sequence of inputs obtained by
// sliding a square window as specified by `parameters` over the 2D inputs
// stored in `data`. If the `invert` parameter is true, the minimum value within
// the window is taken; else, the maximum is taken. For example, for MNIST, this
// means that when  `invert` is true, the blackest pixel within the window
// matters, otherwise the whitest pixel matters. `data` is a sequence of values
// representing a concatenated sequence of 2D square inputs with edge length
// `parameters.input_edge_length`, flattened row by row. The order of elements
// stored in `examples` and `data` should be the same.
void AppendConvLikeInputs(const std::vector<float>& data,
                          const ConvWindowParameters& params,
                          std::vector<Example>* examples);

class SpikingProblem {
 public:
  explicit SpikingProblem(const ProblemType problem_type,
                          const int n_inputs = 2, const int n_outputs = 10,
                          const std::pair<double, double>& input_range = {0, 1},
                          const double noise_factor = 0.0,
                          const bool noisy_targets = false,
                          int conv_edge_length = 1)
      : problem_type_(problem_type),
        conv_edge_length_(conv_edge_length),
        n_inputs_(InitialiseNInputs(n_inputs)),
        n_outputs_(InitialiseNOutputs(n_outputs)),
        n_digits_(InitialiseNDigits(n_outputs)),
        input_range_(input_range),
        noise_factor_(noise_factor),
        noisy_targets_(noisy_targets) {
    IHM_CHECK(input_range_.first < input_range_.second);
    IHM_CHECK(input_range_.first >= 0.0);
    IHM_CHECK(input_range_.second > 0.0);
  }

  // Generates the requested number of examples, according to the problem type.
  // All generated values are scaled in the pre-defined input range. If the data
  // is loaded from an existing set (MNIST/CSV), `should_shuffle`
  // specifies if the loaded training data should be first shuffled. Then,
  // according to the problem type:
  // * AND, OR, XOR, CIRCLE: the examples are randomly generated.
  // * MNIST, MNIST2, MNIST_INV, CIFAR10: the train examples are the first
  // `n_train` examples, and the validation examples are the next `n_validation`
  // examples from the loaded train data. The test examples are the first
  // `n_test` examples from the loaded test data.
  // * CSV: the train, validation and test examples are the first
  // `n_train`, `n_validation` and `n_test` from the training data, respectively
  // (there is no separate test data). The number of requested examples must not
  // exceed the available data size.
  void GenerateExamples(int n_train, int n_validation, int n_test,
                        bool should_shuffle);

  // Generates the requested number of examples, according to the problem type.
  // All generated values are scaled in the pre-defined input range. If the data
  // is loaded from an existing set (MNIST/CIFAR10/CSV), `should_shuffle`
  // specifies if the loaded training data should be first shuffled. Then, the
  // first `n_train` examples go into the train examples, except for the
  // examples between [validation_index_low, validation_index_high), which go
  // into the validation set. The first `n_test` examples from the loaded test
  // data go into the test examples. For CSV, `n_test` must be 0, as there
  // is no separate test data. This method only works for problems with loaded
  // data (MNIST, MNIST2, MNIST_INV, CSV, CIFAR10).
  void GeneratePartitionedExamples(int n_train, int n_test,
                                   int validation_index_low,
                                   int validation_index_high,
                                   bool should_shuffle);

  ProblemType problem_type() const { return problem_type_; }
  int conv_edge_length() const { return conv_edge_length_; }
  int n_inputs() const { return n_inputs_; }
  int n_outputs() const { return n_outputs_; }
  int n_digits() const { return n_digits_; }
  std::pair<double, double> input_range() const { return input_range_; }

  std::vector<Example>& train_examples() { return train_examples_; }
  std::vector<Example>& validation_examples() { return validation_examples_; }
  std::vector<Example>& test_examples() { return test_examples_; }

  std::string& cifar10_data_path() { return cifar10_data_path_; }
  std::string& mnist_data_path() { return mnist_data_path_; }
  std::string& mnist_train_images_filename() {
    return mnist_train_images_filename_;
  }
  std::string& mnist_train_labels_filename() {
    return mnist_train_labels_filename_;
  }
  std::string& mnist_test_images_filename() {
    return mnist_test_images_filename_;
  }
  std::string& mnist_test_labels_filename() {
    return mnist_test_labels_filename_;
  }

  // Stores a full path to a CSV file that stores a dataset. On each row, the
  // dataset must contain elements separated by commas. The last element
  // indicates the class and it will be whitespace-trimmed and lowercased. Rows
  // must be separated by newline. Missing values will be converted to kNoSpike.
  std::string csv_data_path_ = "";

 private:
  const ProblemType problem_type_;
  const int conv_edge_length_;
  const int n_inputs_;
  const int n_outputs_;
  const int n_digits_;
  const std::pair<double, double> input_range_;
  const double noise_factor_;
  const double noisy_targets_;

  int InitialiseNInputs(int n_inputs);
  int InitialiseNOutputs(int n_outputs);
  int InitialiseNDigits(int n_outputs);

  // Loads the data, if not already loaded, from MNIST, MNIST2, MNIST_INV, CSV.
  // Does not load anything for other problem types.
  // If `should_shuffle` is true, shuffles the loaded data (not the actual
  // train, validation or test data) regardless of whether the data was already
  // loaded.
  void LoadExamplesIfRequired(bool should_shuffle);

  // Loads data from the csv file. Missing values are converted to kNoSpike.
  std::vector<Example> LoadCsvInSpikingFormat();

  std::vector<Example> LoadMNistInSpikingFormat(const std::string& data_path,
                                                const std::string& labels_path);

  std::vector<Example> LoadCifar10InSpikingFormat(
      absl::string_view cifar_base_path,
      const std::vector<absl::string_view>& files_to_load);

  void GenerateExamplesWithSeparateTest(int n_train, int n_validation,
                                        int n_test, bool should_shuffle);

  // Loads (if necessary) and splits the dataset into train, validation, test.
  void GenerateExamplesFromSingleSet(int n_train, int n_validation, int n_test,
                                     bool should_shuffle);

  std::vector<Example> GenerateSpikingProblemCircleExamples(int n_examples);
  std::vector<Example> GenerateSpikingProblemLogicExamples(int n_examples,
                                                           ProblemType problem,
                                                           int n_inputs);

  std::vector<Example> train_examples_;
  std::vector<Example> validation_examples_;
  std::vector<Example> test_examples_;

  std::string mnist_data_path_ = "";
  std::string mnist_train_images_filename_ = "train-images-idx3-ubyte";
  std::string mnist_train_labels_filename_ = "train-labels-idx1-ubyte";
  std::string mnist_test_images_filename_ = "t10k-images-idx3-ubyte";
  std::string mnist_test_labels_filename_ = "t10k-labels-idx1-ubyte";

  // Must be set explicitly by the caller. The given folder must contain files
  // named "data_batch_1.bin" through "data_batch_5.bin" for training data, and
  // "test_batch.bin" for test data.
  std::string cifar10_data_path_ = "";

  // These are currently only used for MNIST, to save loading time.
  std::vector<Example> loaded_train_examples_;
  std::vector<Example> loaded_test_examples_;
};

}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_TEMPCODING_SPIKING_PROBLEM_H_
