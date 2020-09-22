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

#include <random>

#include "common/util.h"
#include "event/problem.h"
#include "event/runner.h"

namespace ihmehimmeli {

// Test code to run the event-based tempcoder on MNIST.
void MnistMain(std::string mnist_path) {
  // Load MNIST data.
  MNIST mnist(mnist_path);
  std::vector<uint32_t> training, validation, test;
  mnist.Split(0, &training, &validation, &test);

  // TODO: come up with some interface to define networks.

  // Set up a single-hidden-layer network.
  size_t hidden = 100;
  size_t pulses_per_layer = 5;
  size_t pulses = pulses_per_layer * 2;
  const double decay_rate = 0.1;

  // Input + pulses.
  AlphaPotential alpha_potential(decay_rate);
  Network network(&alpha_potential);
  network.n_inputs = mnist.NumInputs();
  network.pulses_t0.resize(pulses);
  network.pulses_interval.resize(pulses);
  // Ensure each neuron spikes only once.
  network.refractory_period = 1e9;
  for (size_t i = 0; i < pulses; i++) {
    // Non-periodic pulses.
    network.pulses_t0[i] =
        1.0f * (i % pulses_per_layer) / (pulses_per_layer - 1);
    network.pulses_interval[i] = std::numeric_limits<double>::max();
  }

  // Hidden layer.
  std::mt19937 rng(42);
  std::normal_distribution<double> hidden_weight_dist(
      0.0f, std::sqrt(2.0f / (hidden + pulses_per_layer + network.n_inputs)));
  network.connections.resize(network.connections.size() + hidden);
  for (size_t i = 0; i < hidden; i++) {
    network.connections[i].resize(mnist.NumInputs() + pulses_per_layer);
    for (size_t j = 0; j < mnist.NumInputs() + pulses_per_layer; j++) {
      network.connections[i][j].weight_id = network.weights.size();
      network.weights.push_back(hidden_weight_dist(rng));
      network.connections[i][j].source = j;
    }
  }

  // Output layer.
  std::normal_distribution<double> output_weight_dist(
      0.0f, std::sqrt(2.0f / (hidden + pulses_per_layer + mnist.NumOutputs())));
  network.outputs.resize(mnist.NumOutputs());
  std::iota(network.outputs.begin(), network.outputs.end(),
            network.n_inputs + pulses + network.connections.size());
  network.connections.resize(network.connections.size() + mnist.NumOutputs());
  for (size_t i = hidden; i < hidden + mnist.NumOutputs(); i++) {
    network.connections[i].resize(hidden + pulses_per_layer);
    // Connections to hidden layer.
    for (size_t j = 0; j < hidden; j++) {
      network.connections[i][j].weight_id = network.weights.size();
      network.weights.push_back(output_weight_dist(rng));
      network.connections[i][j].source = mnist.NumInputs() + pulses + j;
    }
    // Connections to pulses.
    for (size_t j = 0; j < pulses_per_layer; j++) {
      network.connections[i][j + hidden].weight_id = network.weights.size();
      network.weights.push_back(output_weight_dist(rng));
      network.connections[i][j + hidden].source =
          mnist.NumInputs() + pulses_per_layer + j;
    }
  }

  // Training parameters.
  TrainingParams training_params;
  training_params.loss.reset(new CrossEntropyLoss());
  TerminationParams termination_params;  // defaults are OK.
  LearningParams learning_params;
  StderrTrainCallback train_callback;

  TrainNetwork(learning_params, training_params, termination_params, mnist,
               training, validation, &network, &train_callback);

  StderrTestCallback test_callback;
  TestNetwork(termination_params, mnist, test, network,
              training_params.loss.get(), &test_callback);
}

}  // namespace ihmehimmeli

int main(int argc, char **argv) {
  IHM_CHECK(argc >= 2);
  ihmehimmeli::MnistMain(argv[1]);
}
