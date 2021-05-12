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

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "common/util.h"
#include "event/event.h"
#include "event/loss.h"
#include "event/potential.h"
#include "event/problem.h"
#include "event/runner.h"

ABSL_FLAG(double, decay_rate, 1.0, "decay rate");
ABSL_FLAG(double, lr, 0.01, "learning rate (non-pulses)");
ABSL_FLAG(double, lr_pulses, 0.001, "learning rate for pulses");
ABSL_FLAG(int64_t, n_epochs, 10, "number of training epochs");
ABSL_FLAG(int64_t, batch_size, 2, "batch size");
ABSL_FLAG(int64_t, n_train, 10000, "number of training examples");
ABSL_FLAG(int64_t, n_validation, 1000, "number of validation examples");
ABSL_FLAG(int64_t, n_test, 10000, "number of test examples");
ABSL_FLAG(int64_t, n_layers, 2, "number of layers");
ABSL_FLAG(
    int64_t, layer_size_multiplier, 4,
    "each layer will have this times more neurons than the number of inputs");
ABSL_FLAG(int64_t, n_inputs, 2, "number of inputs");

namespace ihmehimmeli {

void RepeatMain() {
  uint32_t num_inputs = absl::GetFlag(FLAGS_n_inputs);
  double decay_rate = absl::GetFlag(FLAGS_decay_rate);

  // Set up a cyclic network with one pulse and two neurons per toroidal
  // layer. Each layer is fully connected to the previous and the following
  // one. There is one more pulse for the output layer.
  uint32_t num_layers = absl::GetFlag(FLAGS_n_layers);
  uint32_t neurons_per_layer =
      num_inputs * absl::GetFlag(FLAGS_layer_size_multiplier);
  uint32_t pulses = num_layers + 1;
  double initial_cycle_length = num_layers / decay_rate;

  // Input + pulses.
  AlphaPotential alpha_potential(decay_rate);
  Network network(&alpha_potential);
  network.pulses_t0.resize(pulses);
  network.pulses_interval.resize(pulses);

  RepeatInput repeat_input(num_inputs, /*n_train=*/absl::GetFlag(FLAGS_n_train),
                           /*n_validation=*/absl::GetFlag(FLAGS_n_validation),
                           /*n_test=*/absl::GetFlag(FLAGS_n_test),
                           initial_cycle_length);
  std::vector<uint32_t> training, validation, test;
  repeat_input.Split(0, &training, &validation, &test);
  network.n_inputs = repeat_input.NumInputs();

  // Initialize network with a given cycle length.
  for (uint32_t i = 0; i < num_layers; i++) {
    network.pulses_t0[i] = initial_cycle_length * i / num_layers;
    network.pulses_interval[i] = initial_cycle_length;
  }
  // Output pulse.
  network.pulses_t0[num_layers] = initial_cycle_length;
  network.pulses_interval[num_layers] = initial_cycle_length;
  network.outputs.push_back(network.n_inputs + num_layers);

  // No refractory period, to allow neurons to spike as often as needed.
  network.refractory_period = 0;

  std::mt19937 rng(0);
  std::normal_distribution<double> dist(0, 1);

  auto add_connection = [&](size_t neuron, size_t source) {
    network.connections[neuron].emplace_back();
    network.connections[neuron].back().source = source;
    network.connections[neuron].back().weight_id = network.weights.size();
    network.weights.push_back(dist(rng));
  };

  // Set up cycle.
  for (uint32_t i = 0; i < num_layers; i++) {
    uint32_t previous_layer = i == 0 ? num_layers - 1 : i - 1;
    for (uint32_t j = 0; j < neurons_per_layer; j++) {
      // Add neuron.
      network.connections.emplace_back();
      // Pulse connection.
      add_connection(network.connections.size() - 1, network.n_inputs + i);
      for (uint32_t k = 0; k < neurons_per_layer; k++) {
        add_connection(
            network.connections.size() - 1,
            network.n_inputs + pulses + previous_layer * neurons_per_layer + k);
      }
    }
  }

  // Connections to inputs.
  for (uint32_t j = 0; j < neurons_per_layer; j++) {
    for (uint32_t in = 0; in < num_inputs; in++) {
      add_connection(j, in);
    }
  }

  // Output neurons.
  for (uint32_t out = 0; out < num_inputs; out++) {
    // Add neuron.
    network.outputs.push_back(num_inputs + pulses + network.connections.size());
    network.connections.emplace_back();
    // Pulse connection.
    add_connection(network.connections.size() - 1, num_inputs + num_layers);
    for (uint32_t j = 0; j < neurons_per_layer; j++) {
      add_connection(
          network.connections.size() - 1,
          network.n_inputs + pulses + (num_layers - 1) * neurons_per_layer + j);
    }
  }

  // Training parameters.
  TrainingParams training_params;
  training_params.loss.reset(new CyclicLoss(initial_cycle_length));
  TerminationParams termination_params;
  // Up to two cycles after the last input. Higher values seem to help the
  // network find a reasonable initial condition.
  termination_params.max_time_after_last_input = 4 * initial_cycle_length;
  LearningParams learning_params;
  learning_params.learning_rate = absl::GetFlag(FLAGS_lr);
  learning_params.learning_rate_pulses = absl::GetFlag(FLAGS_lr_pulses);
  learning_params.num_epochs = absl::GetFlag(FLAGS_n_epochs);
  learning_params.batch_size = absl::GetFlag(FLAGS_batch_size);
  StderrTrainCallback train_callback;

  // Debug options: save outputs and expected outputs to files with this
  // probability.
  DebugInfo debug_info;
  debug_info.print_spike_times_probability = 0.0;
  debug_info.print_outputs_and_expected_probability = 0.0;
  network.debug_info = &debug_info;

  Network best_network =
      TrainNetwork(learning_params, training_params, termination_params,
                   repeat_input, training, validation, &network,
                   &train_callback)
          .best_network;

  StderrTestCallback test_callback;
  TestNetwork(termination_params, repeat_input, test, best_network,
              training_params.loss.get(), &test_callback);
}

}  // namespace ihmehimmeli

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  ihmehimmeli::RepeatMain();
}
