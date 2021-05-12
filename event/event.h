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

#ifndef IHMEHIMMELI_EVENT_EVENT_H
#define IHMEHIMMELI_EVENT_EVENT_H
#include <stdint.h>

#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/kahan.h"
#include "common/util.h"
#include "event/loss.h"
#include "event/potential.h"

namespace ihmehimmeli {

struct DebugInfo {
  // Debugging/diagnosing parameters.
  std::string debug_dir = "/tmp/ihm-debug";
  // Print all the firing events with a timestamp to file with this probability
  // (decided per single network evaluation).
  double print_spike_times_probability = 0.0;
  // Print all the outputs and expected outputs of the network with this
  // probability.
  double print_outputs_and_expected_probability = 0.0;
  mutable std::atomic<size_t> current_debug_file{0};
};

// Network definition parameters.
// TODO: serialization/deserialization support (visitor?).
// TODO: allow multiple sources of loss (i.e. for L2 regularization).
struct Network {
  Network(PotentialFunction *potential_function) {
    this->potential = potential_function;
  }

  // Number of inputs.
  uint32_t n_inputs;

  // Synchronization pulses. `pulses_t0` and `pulses_interval` should have
  // the same size. Setting an interval different from
  // std::numeric_limits<double>::max() will cause the pulse to periodically
  // repeat.
  std::vector<double> pulses_t0;
  std::vector<double> pulses_interval;

  // Weights
  std::vector<double> weights;

  struct Connection {
    uint32_t source;
    uint32_t weight_id;
  };

  // For each neuron, specify the neurons it gets an input from.
  // Neurons are numbered as follows:
  //  - input neurons are [0, n_inputs)
  //  - the next pulses.size() neurons are sync pulses
  //  - all other neurons (connections.size() in total) come after.
  std::vector<std::vector<Connection>> connections;

  // Potential function to use. The owner of the Network must ensure this is
  // valid whenever the Network is used.
  PotentialFunction *potential;

  // Firing threshold and length of refractory period.
  double firing_threshold = 1.0;
  double refractory_period = 0.0;

  // Indices of output neurons.
  std::vector<uint32_t> outputs;

  DebugInfo *debug_info = nullptr;
};

// Derivatives of loss function wrt network parameters and inputs.
struct NetworkGradient {
  std::vector<std::vector<Kahan<double>>> d_inputs;
  std::vector<Kahan<double>> d_pulses_t0;
  std::vector<Kahan<double>> d_pulses_interval;
  std::vector<Kahan<double>> d_weights;
  std::mutex mt;
  void Assimilate(NetworkGradient *other);
  void Clear();
};

// Parameters for training steps.
struct TrainingParams {
  double clip_derivatives = 100;
  double no_spike_penalty = 1;
  std::unique_ptr<LossFunction> loss;
};

// Defines conditions for evaluation termination.
struct TerminationParams {
  // Stop when this number of output spikes have been produced by all output
  // neurons (disabled if 0).
  uint32_t num_full_outputs = 0;

  // Stop when this number of output spikes have been produced by at least one
  // output neuron (disabled if 0).
  uint32_t num_single_outputs = 0;

  // Maximum amount of time to run the network for.
  double max_time = std::numeric_limits<double>::max();

  // How much time to run the network for after the last input was fed in.
  double max_time_after_last_input = std::numeric_limits<double>::max();

  // If no termination conditions are set, the network runs until all spikes
  // drain. This might never happen if the network contains loops, or periodic
  // pulses.
};

// Runs the network on single example, and produces the output spikes.
void Run(const Network &network, const TerminationParams &termination_params,
         const std::vector<std::vector<double>> &inputs,
         std::vector<std::vector<double>> *outputs);

// Same as `Run`, but also computes the value of the loss function.
double RunAndComputeLoss(const Network &network,
                         const TerminationParams &termination_params,
                         const std::vector<std::vector<double>> &inputs,
                         LossFunction *loss,
                         const std::vector<std::vector<double>> &expected,
                         std::vector<std::vector<double>> *outputs);

// Runs the network on a single example, produces the outputs and the gradients
// according to the given TrainingParams.
double RunAndBackpropagate(const Network &network,
                           const TerminationParams &termination_params,
                           const std::vector<std::vector<double>> &inputs,
                           const TrainingParams &training_params,
                           const std::vector<std::vector<double>> &expected,
                           std::vector<std::vector<double>> *outputs,
                           NetworkGradient *gradient,
                           NetworkGradient *global_gradient);

}  // namespace ihmehimmeli

#endif
