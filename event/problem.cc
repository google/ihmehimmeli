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

#include "event/problem.h"

#include <algorithm>
#include <random>

#include "common/mnist_loader.h"
#include "event/potential.h"

namespace ihmehimmeli {

MNIST::MNIST(std::string path) {
  int n_inputs;
  // Load training data
  LoadMNISTDataAndLabels(path + "/train-images-idx3-ubyte",
                         path + "/train-labels-idx1-ubyte", &n_inputs,
                         &n_train_, &mnist_all_data_float_, &mnist_labels_);
  // Load test data
  LoadMNISTDataAndLabels(path + "/t10k-images-idx3-ubyte",
                         path + "/t10k-labels-idx1-ubyte", &n_inputs, &n_test_,
                         &mnist_all_data_float_, &mnist_labels_);
}

void MNIST::Example(uint32_t id, std::vector<std::vector<double>> *input,
                    std::vector<std::vector<double>> *output) const {
  const float *data_in = mnist_all_data_float_.data() + id * NumInputs();
  const size_t label = mnist_labels_[id];
  input->resize(NumInputs());
  output->resize(NumOutputs());
  for (size_t i = 0; i < NumInputs(); i++) {
    input->data()[i].resize(1);
    input->data()[i][0] = 1.0 - data_in[i];
  }
  for (size_t i = 0; i < NumOutputs(); i++) {
    output->data()[i].resize(1);
    output->data()[i][0] = i == label ? 0 : PotentialFunction::kNoSpike;
  }
}

void MNIST::Split(size_t seed, std::vector<uint32_t> *training,
                  std::vector<uint32_t> *validation,
                  std::vector<uint32_t> *test) {
  std::mt19937 rng(seed);
  test->resize(n_test_);
  std::iota(test->begin(), test->end(), n_train_);
  training->resize(n_train_);
  std::iota(training->begin(), training->end(), 0);
  size_t n_training = 0.9 * n_train_;
  std::shuffle(training->begin(), training->end(), rng);
  validation->assign(training->begin() + n_training, training->end());
  training->resize(n_training);
}

void RepeatInput::Example(uint32_t id, std::vector<std::vector<double>> *input,
                          std::vector<std::vector<double>> *output) const {
  std::mt19937 rng(id);
  size_t num_events = std::uniform_int_distribution<size_t>(2, 64)(rng);
  std::uniform_int_distribution<size_t> input_dist(0, num_inputs_ - 1);
  input->clear();
  input->resize(num_inputs_);
  output->clear();
  output->resize(num_inputs_);
  double last_event = 0;
  for (size_t i = 0; i < num_events; i++) {
    size_t in = input_dist(rng);
    (*input)[in].push_back(last_event);
    (*output)[in].push_back(last_event);
    last_event += 4.0 * cycle_length_;
  }
}

void RepeatInput::Split(size_t seed, std::vector<uint32_t> *training,
                        std::vector<uint32_t> *validation,
                        std::vector<uint32_t> *test) {
  std::mt19937 rng(seed);
  for (size_t i = 0; i < n_train_; i++) {
    training->push_back(rng());
  }
  for (size_t i = 0; i < n_validation_; i++) {
    validation->push_back(rng());
  }
  for (size_t i = 0; i < n_test_; i++) {
    test->push_back(rng());
  }
}

void FlipFlop::Example(uint32_t id, std::vector<std::vector<double>> *input,
                       std::vector<std::vector<double>> *output) const {
  std::mt19937 rng(id);
  size_t num_events = std::uniform_int_distribution<size_t>(2, 64)(rng);
  // Every four `cycle_length`s, either don't do anything or change the state.
  std::uniform_int_distribution<int> event_interval(1, 2);
  input->clear();
  input->resize(1);
  output->clear();
  output->resize(num_outputs_);
  double last_event =
      cycle_length_;  //  Give the network some time to initialize.
  size_t cur = 0;
  for (size_t i = 0; i < num_events; i++) {
    cur = (cur + 1) % num_outputs_;
    (*input)[0].push_back(last_event);
    (*output)[cur].push_back(last_event);
    last_event += 4.0 * cycle_length_ * event_interval(rng);
  }
}

void FlipFlop::Split(size_t seed, std::vector<uint32_t> *training,
                     std::vector<uint32_t> *validation,
                     std::vector<uint32_t> *test) {
  std::mt19937 rng(seed);
  for (size_t i = 0; i < n_train_; i++) {
    training->push_back(rng());
  }
  for (size_t i = 0; i < n_validation_; i++) {
    validation->push_back(rng());
  }
  for (size_t i = 0; i < n_test_; i++) {
    test->push_back(rng());
  }
}
}  // namespace ihmehimmeli
