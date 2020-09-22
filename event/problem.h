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

#ifndef IHMEHIMMELI_EVENT_PROBLEM_H
#define IHMEHIMMELI_EVENT_PROBLEM_H
#include <stdint.h>

#include <limits>
#include <string>
#include <vector>

namespace ihmehimmeli {

class Problem {
 public:
  virtual ~Problem() = default;

  virtual size_t NumInputs() const = 0;

  virtual std::string Name() const = 0;

  // Number of outputs *of a network solving this problem*. This needs not be
  // the length of the `output` vector returned by Example.
  virtual size_t NumOutputs() const = 0;

  // Produces the example (`input`/`output` pair) corresponding to a given
  // `id`. `id` must be a value returned in one of the output parameters of a
  // call to `Split`; all calls with the same value of `id` will produce the
  // same example.
  virtual void Example(uint32_t id, std::vector<std::vector<double>> *input,
                       std::vector<std::vector<double>> *output) const = 0;

  // Produces indentifiers of training/validation/test examples. For problems
  // with a fixed dataset, most implementations will return identifiers that
  // correspond to indexes of the available examples. For problems with a
  // generated dataset, in most implementations identifiers will correspond to
  // seeds to some random number generator. The `seed` parameter is used to
  // control the split: calls with the same seed are guaranteed to produce the
  // same split.
  virtual void Split(size_t seed, std::vector<uint32_t> *training,
                     std::vector<uint32_t> *validation,
                     std::vector<uint32_t> *test) = 0;
};

class MNIST : public Problem {
 public:
  // Loads MNIST data from the given path.
  explicit MNIST(std::string path);

  virtual size_t NumInputs() const override final { return 28 * 28; }
  virtual size_t NumOutputs() const override final { return 10; }
  virtual std::string Name() const override final { return "MNIST"; }

  // Returns an `input` vector that contains one value in the [0, 1] range (with
  // 0 representing the brightest pixels) per input neuron. The `output` vector
  // will contain a single `0` value for the neuron representing the correct
  // digit, and a single `PotentialFunction::kNoSpike` for all others.
  void Example(uint32_t id, std::vector<std::vector<double>> *input,
               std::vector<std::vector<double>> *output) const override final;

  // Splits the training dataset into training and validation, using a 90%
  // split, and returns the test dataset as-is.
  void Split(size_t seed, std::vector<uint32_t> *training,
             std::vector<uint32_t> *validation,
             std::vector<uint32_t> *test) override final;

 private:
  std::vector<float> mnist_all_data_float_;
  std::vector<int> mnist_labels_;
  int n_train_;
  int n_test_;
};

// Repeat the last input value until the next input arrives.
class RepeatInput : public Problem {
 public:
  explicit RepeatInput(size_t num_inputs, size_t n_train, size_t n_validation,
                       size_t n_test, double cycle_length)
      : num_inputs_(num_inputs),
        n_train_(n_train),
        n_validation_(n_validation),
        n_test_(n_test),
        cycle_length_(cycle_length) {}

  virtual size_t NumInputs() const override final { return num_inputs_; }
  virtual size_t NumOutputs() const override final {
    // 1 sync signal + num_inputs_ outputs.
    return 1 + num_inputs_;
  }
  virtual std::string Name() const override final { return "RepeatInput"; }

  // Returns an `input` vector that contains spikes on each input neuron in
  // regular intervals (with some spikes potentially skipped). The `output`
  // vector contains the same spikes as the input vector: the network should
  // produce output spikes corresponding to the previously-received input.
  void Example(uint32_t id, std::vector<std::vector<double>> *input,
               std::vector<std::vector<double>> *output) const override final;

  // Produces RNG seeds for the training/validation/test datasets, in amounts
  // corresponding to the respective `n_train`/`n_validation`/`n_test`
  // constructor params.
  void Split(size_t seed, std::vector<uint32_t> *training,
             std::vector<uint32_t> *validation,
             std::vector<uint32_t> *test) override final;

 private:
  size_t num_inputs_;
  size_t n_train_;
  size_t n_validation_;
  size_t n_test_;
  double cycle_length_;
};

// Alternate cyclically between the output values when an input is received.
class FlipFlop : public Problem {
 public:
  explicit FlipFlop(size_t num_outputs, size_t n_train, size_t n_validation,
                    size_t n_test, double cycle_length)
      : num_outputs_(num_outputs),
        n_train_(n_train),
        n_validation_(n_validation),
        n_test_(n_test),
        cycle_length_(cycle_length) {}

  virtual size_t NumInputs() const override final { return 1; }
  virtual size_t NumOutputs() const override final {
    // 1 sync signal + num_outputs_ outputs.
    return 1 + num_outputs_;
  }
  virtual std::string Name() const override final { return "FlipFlop"; }

  // Returns an `input` vector that contains spikes on the single input neuron
  // in regular intervals (with some spikes potentially skipped). The `output`
  // vector contains spikes that alternate between the outputs, starting from 0
  // and switching to the next output (cyclically) whenever an input spike is
  // produced.
  void Example(uint32_t id, std::vector<std::vector<double>> *input,
               std::vector<std::vector<double>> *output) const override final;

  // Produces RNG seeds for the training/validation/test datasets, in amounts
  // corresponding to the respective `n_train`/`n_validation`/`n_test`
  // constructor params.
  void Split(size_t seed, std::vector<uint32_t> *training,
             std::vector<uint32_t> *validation,
             std::vector<uint32_t> *test) override final;

 private:
  size_t num_outputs_;
  size_t n_train_;
  size_t n_validation_;
  size_t n_test_;
  double cycle_length_;
};
}  // namespace ihmehimmeli

#endif
