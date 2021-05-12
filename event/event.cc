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

#include "event/event.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <queue>
#include <random>

#include "common/util.h"
#include "event/spike_queue.h"

namespace ihmehimmeli {

namespace {

struct FwdConnection {
  uint32_t dest_neuron_id;
  uint32_t connection_idx;
  uint32_t weight_id;
};

struct FireEvent {
  uint32_t source_neuron_id;
  uint32_t source_spike_id;
  uint32_t dest_neuron_id;
  uint32_t dest_spike_id;
  uint32_t weight_id;
};

// TODO: all the setup in this function is rather expensive and should
// likely only be done once. In particular, a lot of time is spent in
// allocations.
template <bool backprop>
double RunImpl(const Network &network,
               const TerminationParams &termination_params,
               const std::vector<std::vector<double>> &inputs,
               const TrainingParams &training_params, LossFunction *loss,
               const std::vector<std::vector<double>> &expected,
               std::vector<std::vector<double>> *outputs,
               NetworkGradient *gradient, NetworkGradient *global_gradient) {
  IHM_CHECK(network.n_inputs == inputs.size());
  IHM_CHECK(network.pulses_t0.size() == network.pulses_interval.size());
  if (backprop) {
    // Ensure gradients have the correct size.
    gradient->d_weights.resize(network.weights.size());
    gradient->d_pulses_t0.resize(network.pulses_t0.size());
    gradient->d_pulses_interval.resize(network.pulses_interval.size());
    gradient->d_inputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      gradient->d_inputs[i].resize(inputs[i].size());
    }
  }
  outputs->resize(network.outputs.size());
  for (size_t i = 0; i < network.outputs.size(); i++) {
    outputs->data()[i].clear();
  }

  uint32_t first_noninput_neuron_id =
      network.n_inputs + network.pulses_t0.size();

  // Compute network forward links.
  std::vector<std::vector<FwdConnection>> fwd_connections(
      network.n_inputs + network.pulses_t0.size() + network.connections.size());
  for (uint32_t i = 0; i < network.connections.size(); i++) {
    for (uint32_t j = 0; j < network.connections[i].size(); j++) {
      const Network::Connection &c = network.connections[i][j];
      IHM_CHECK(c.source < fwd_connections.size());
      fwd_connections[c.source].push_back(
          FwdConnection{first_noninput_neuron_id + i, j, c.weight_id});
    }
  }

  // For each neuron, index of its corresponding output, if any.
  size_t num_outputs = network.outputs.size();

  // Map from neuron ID to output number.
  std::vector<uint32_t> output_idx(fwd_connections.size(), num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    output_idx[network.outputs[i]] = i;
  }

  // List of spike propagations for backprop.
  std::vector<FireEvent> spike_propagations;

  // Bookkeeping information for potential function, one per output spike (plus
  // the pending one). Not used for input and pulse spikes.
  size_t potential_state_size = network.potential->StateSize();
  std::vector<std::vector<double>> potential_state(
      fwd_connections.size(), std::vector<double>(potential_state_size));

  // Timing information of each spike.
  std::vector<std::vector<double>> spikes(fwd_connections.size());

  // Spike index of the last processed spike for each input neuron.
  std::vector<uint32_t> cur_input_spike(network.n_inputs);

  SpikeQueue queue;
  queue.Init(fwd_connections.size());

  for (uint32_t i = 0; i < network.n_inputs; i++) {
    const std::vector<double> &in = inputs[i];
    IHM_CHECK(is_sorted(in.begin(), in.end()));
    if (!in.empty()) {
      queue.Add({in[0], i});
    }
  }
  for (uint32_t i = 0; i < network.pulses_t0.size(); i++) {
    queue.Add({network.pulses_t0[i], network.n_inputs + i});
  }

  double max_time = termination_params.max_time;
  if (termination_params.max_time_after_last_input !=
      std::numeric_limits<double>::max()) {
    double last_input = 0;
    for (uint32_t i = 0; i < network.n_inputs; i++) {
      if (!inputs[i].empty()) {
        last_input = std::max(last_input, inputs[i].back());
      }
    }
    max_time = std::min(
        last_input + termination_params.max_time_after_last_input, max_time);
  }

  std::mt19937 rng(std::random_device{}());

  FILE *f = nullptr;
  if (network.debug_info &&
      std::uniform_real_distribution<double>(0, 1)(rng) <
          network.debug_info->print_spike_times_probability) {
    size_t current_debug_file = network.debug_info->current_debug_file++;
    f = fopen((network.debug_info->debug_dir + "/times_" +
               std::to_string(current_debug_file) + ".txt")
                  .c_str(),
              "w");
    IHM_CHECK(f != nullptr);
  }

  // Evaluation loop. Also populates `outputs`.
  Spike spike;
  while (queue.Pop(&spike)) {
    // Early exit condition: run time exceeded.
    if (max_time < spike.timestamp) {
      break;
    }

    if (f != nullptr) {
      fprintf(f, "%u %f\n", spike.source_neuron_id, spike.timestamp);
    }

    // Record spike time for backprop.
    spikes[spike.source_neuron_id].push_back(spike.timestamp);

    // If this spike is an input, enqueue the next input.
    if (spike.source_neuron_id < network.n_inputs) {
      uint32_t id = spike.source_neuron_id;
      cur_input_spike[id]++;
      if (cur_input_spike[id] < inputs[id].size()) {
        queue.Add({inputs[id][cur_input_spike[id]], id});
      }
    }

    // If this spike is a pulse, enqueue the next pulse.
    if (spike.source_neuron_id >= network.n_inputs &&
        spike.source_neuron_id < first_noninput_neuron_id) {
      uint32_t pulse_id = spike.source_neuron_id - network.n_inputs;
      if (network.pulses_interval[pulse_id] !=
          std::numeric_limits<double>::max()) {
        queue.Add({spike.timestamp + network.pulses_interval[pulse_id],
                   spike.source_neuron_id});
      }
    }

    // Copy output spike (if output neuron).
    if (output_idx[spike.source_neuron_id] != outputs->size()) {
      outputs->data()[output_idx[spike.source_neuron_id]].push_back(
          spike.timestamp);
    }

    // Check early exit conditions.
    if (termination_params.num_full_outputs != 0) {
      bool early_exit = true;
      for (const std::vector<double> &out : *outputs) {
        if (out.size() < termination_params.num_full_outputs) {
          early_exit = false;
          break;
        }
      }
      if (early_exit) break;
    }
    if (termination_params.num_single_outputs != 0) {
      bool early_exit = false;
      for (const std::vector<double> &out : *outputs) {
        if (out.size() >= termination_params.num_full_outputs) {
          early_exit = true;
          break;
        }
      }
      if (early_exit) break;
    }

    // Reset spike state (i.e. reset potential to 0) for future spikes.
    size_t old_size = potential_state[spike.source_neuron_id].size();
    potential_state[spike.source_neuron_id].resize(old_size +
                                                   potential_state_size);
    network.potential->InitState(
        potential_state[spike.source_neuron_id].data() + old_size);

    // Feed spike to other neurons.
    for (FwdConnection c : fwd_connections[spike.source_neuron_id]) {
      // The destination neuron is in its refractory period.
      if (!spikes[c.dest_neuron_id].empty() &&
          spikes[c.dest_neuron_id].back() + network.refractory_period >
              spike.timestamp) {
        continue;
      }

      // Invalidate all pending spikes on the destination neuron.
      queue.Remove(c.dest_neuron_id);

      // Compute candidate spiking time of next neuron.
      double dest_spike = network.potential->AddSpike(
          spike.timestamp, network.weights[c.weight_id],
          network.firing_threshold,
          potential_state[c.dest_neuron_id].data() +
              potential_state_size * spikes[c.dest_neuron_id].size());

      // Record event from last fired spike of spike.source to pending spike of
      // c.dest.
      FireEvent fire_event;
      fire_event.source_neuron_id = spike.source_neuron_id;
      fire_event.source_spike_id = spikes[spike.source_neuron_id].size() - 1;
      fire_event.weight_id = c.weight_id;
      fire_event.dest_neuron_id = c.dest_neuron_id;
      fire_event.dest_spike_id = spikes[c.dest_neuron_id].size();
      spike_propagations.push_back(fire_event);

      // If the destination spike violates causality (or does not exist),
      // discard it. Such situations might arise from connections with negative
      // weights.
      if (dest_spike < spike.timestamp ||
          dest_spike == PotentialFunction::kNoSpike) {
        continue;
      }

      queue.Add({dest_spike, c.dest_neuron_id});
    }
  }

  if (f != nullptr) fclose(f);

  // If not computing gradients, we are done.
  if (!backprop) {
    if (loss) {
      return loss->Loss(*outputs, expected);
    }
    return 0;
  }

  if (network.debug_info &&
      std::uniform_real_distribution<double>(0, 1)(rng) <
          network.debug_info->print_outputs_and_expected_probability) {
    size_t current_debug_file = network.debug_info->current_debug_file++;
    f = fopen((network.debug_info->debug_dir + "/output_and_expected_" +
               std::to_string(current_debug_file) + ".txt")
                  .c_str(),
              "w");
    IHM_CHECK(f != nullptr);
    fprintf(f, "Fraction of correct: %f\n",
            loss->FracCorrect(*outputs, expected));
    fprintf(f, "Expected: \n");
    for (size_t i = 0; i < expected.size(); i++) {
      for (size_t j = 0; j < expected[i].size(); j++) {
        fprintf(f, "%f ", expected[i][j]);
      }
      fprintf(f, "\n");
    }
    fprintf(f, "Outputs: \n");
    for (size_t i = 0; i < outputs->size(); i++) {
      for (size_t j = 0; j < (*outputs)[i].size(); j++) {
        fprintf(f, "%f ", (*outputs)[i][j]);
      }
      fprintf(f, "\n");
    }
    fclose(f);
  }

  // Derivatives of spikes.
  std::vector<std::vector<Kahan<double>>> d_spikes(spikes.size());
  for (size_t i = 0; i < spikes.size(); i++) {
    d_spikes[i].resize(spikes[i].size());
  }

  double train_loss;
  // Initialize derivative of outputs.
  {
    std::vector<std::vector<double>> d_outputs;
    train_loss = loss->LossDerivative(*outputs, expected, &d_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      for (size_t j = 0; j < d_outputs[i].size(); j++) {
        if (d_spikes[network.outputs[i]].size() <= j) continue;
        d_spikes[network.outputs[i]][j] = d_outputs[i][j];
      }
    }
  }

  // Backprop. Also updates gradients->d_weights.
  for (size_t i = spike_propagations.size(); i > 0; i--) {
    const FireEvent &event = spike_propagations[i - 1];
    // If the destination neuron did not end up spiking, discard the event.
    if (spikes[event.dest_neuron_id].size() <= event.dest_spike_id) {
      continue;
    }

    IHM_CHECK(event.dest_neuron_id >= first_noninput_neuron_id);
    double d_activation;
    double d_weight;
    network.potential->Derivatives(
        spikes[event.source_neuron_id][event.source_spike_id],
        network.weights[event.weight_id], network.firing_threshold,
        spikes[event.dest_neuron_id][event.dest_spike_id],
        &potential_state[event.dest_neuron_id]
                        [event.dest_spike_id * potential_state_size],
        &d_activation, &d_weight);

    if (training_params.clip_derivatives != 0) {
      d_activation = std::max(d_activation, -training_params.clip_derivatives);
      d_activation = std::min(d_activation, training_params.clip_derivatives);
      d_weight = std::max(d_weight, -training_params.clip_derivatives);
      d_weight = std::min(d_weight, training_params.clip_derivatives);
    }

    double d_output = static_cast<double>(
        d_spikes[event.dest_neuron_id][event.dest_spike_id]);
    d_spikes[event.source_neuron_id][event.source_spike_id] +=
        d_activation * d_output;
    gradient->d_weights[event.weight_id] += d_weight * d_output;
  }
  for (size_t i = 0; i < network.connections.size(); i++) {
    // If the neuron never spiked, apply penalty.
    if (spikes[i + first_noninput_neuron_id].empty()) {
      for (size_t j = 0; j < network.connections[i].size(); j++) {
        gradient->d_weights[network.connections[i][j].weight_id] -=
            training_params.no_spike_penalty;
      }
    }
  }

  // Copy derivatives of inputs and pulses gradients.
  for (size_t i = 0; i < network.n_inputs; i++) {
    for (size_t j = 0; j < d_spikes[i].size(); j++) {
      gradient->d_inputs[i][j] = d_spikes[i][j];
    }
  }
  for (size_t i = 0; i < network.pulses_t0.size(); i++) {
    gradient->d_pulses_t0[i] = 0;
    gradient->d_pulses_interval[i] = 0;
    for (size_t j = 0; j < d_spikes[i + network.n_inputs].size(); j++) {
      // time(j-th pulse from neuron i) = pulse_start[i] + j * pulse_interval[i]
      gradient->d_pulses_t0[i] +=
          static_cast<double>(d_spikes[i + network.n_inputs][j]);
      gradient->d_pulses_interval[i] +=
          static_cast<double>(d_spikes[i + network.n_inputs][j]) * j;
    }
  }

  // Assimilate gradients in global gradients.
  global_gradient->Assimilate(gradient);

  return train_loss;
}
}  // namespace

void NetworkGradient::Assimilate(NetworkGradient *other) {
  std::unique_lock<std::mutex> lck(mt);
  // Assumes that `other` has the same sizes as `this`, or that `this` is empty.
  d_weights.resize(other->d_weights.size());
  for (size_t i = 0; i < d_weights.size(); i++) {
    d_weights[i] += static_cast<double>(other->d_weights[i]);
    other->d_weights[i] = 0;
  }
  d_inputs.resize(other->d_inputs.size());
  for (size_t i = 0; i < d_inputs.size(); i++) {
    d_inputs[i].resize(other->d_inputs[i].size());
    for (size_t j = 0; j < d_inputs[i].size(); j++) {
      d_inputs[i][j] += static_cast<double>(other->d_inputs[i][j]);
      other->d_inputs[i][j] = 0;
    }
  }
  d_pulses_t0.resize(other->d_pulses_t0.size());
  for (size_t i = 0; i < d_pulses_t0.size(); i++) {
    d_pulses_t0[i] += static_cast<double>(other->d_pulses_t0[i]);
    other->d_pulses_t0[i] = 0;
  }
  d_pulses_interval.resize(other->d_pulses_interval.size());
  for (size_t i = 0; i < d_pulses_interval.size(); i++) {
    d_pulses_interval[i] += static_cast<double>(other->d_pulses_interval[i]);
    other->d_pulses_interval[i] = 0;
  }
}

void NetworkGradient::Clear() {
  std::unique_lock<std::mutex> lck(mt);
  d_weights.clear();
  for (size_t i = 0; i < d_inputs.size(); i++) {
    d_inputs[i].clear();
  }
  d_pulses_t0.clear();
  d_pulses_interval.clear();
}

double RunAndBackpropagate(const Network &network,
                           const TerminationParams &termination_params,
                           const std::vector<std::vector<double>> &inputs,
                           const TrainingParams &training_params,
                           const std::vector<std::vector<double>> &expected,
                           std::vector<std::vector<double>> *outputs,
                           NetworkGradient *gradient,
                           NetworkGradient *global_gradient) {
  return RunImpl</*backprop=*/true>(
      network, termination_params, inputs, training_params,
      training_params.loss.get(), expected, outputs, gradient, global_gradient);
}

double RunAndComputeLoss(const Network &network,
                         const TerminationParams &termination_params,
                         const std::vector<std::vector<double>> &inputs,
                         LossFunction *loss,
                         const std::vector<std::vector<double>> &expected,
                         std::vector<std::vector<double>> *outputs) {
  return RunImpl</*backprop=*/false>(network, termination_params, inputs,
                                     TrainingParams{}, loss, expected, outputs,
                                     nullptr, nullptr);
}

void Run(const Network &network, const TerminationParams &termination_params,
         const std::vector<std::vector<double>> &inputs,
         std::vector<std::vector<double>> *outputs) {
  RunImpl</*backprop=*/false>(network, termination_params, inputs,
                              TrainingParams{}, nullptr, {}, outputs, nullptr,
                              nullptr);
}

}  // namespace ihmehimmeli
