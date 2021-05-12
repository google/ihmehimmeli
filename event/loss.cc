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

#include "event/loss.h"

#include <algorithm>
#include <cstddef>

#include "absl/strings/str_cat.h"
#include "common/util.h"
#include "event/potential.h"

namespace ihmehimmeli {

namespace {
// Utility function to check if a classification problem is correct.
// `spike_idx` represents the number of the output that should be checked, i.e.
// if `spike_idx` is 0 then the first outputs of each neurons will be compared,
// and so on.
bool IsCorrect(const Outputs &outputs, const Outputs &expected, size_t idx) {
  IHM_CHECK(outputs.size() == expected.size());
  IHM_CHECK(idx < expected[0].size());
  bool found = false;
  size_t omin = 0;
  size_t emin = 0;
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i].size() > idx &&
        outputs[i][idx] != PotentialFunction::kNoSpike) {
      if (!found || outputs[i][idx] < outputs[omin][idx]) {
        omin = i;
        found = true;
      }
    }
    if (expected[emin][idx] > expected[i][idx]) {
      emin = i;
    }
  }
  return found && omin == emin;
}

// Produces outputs to feed to CrossEntropyLoss from outputs for CyclicLoss.
// Sorts all the spikes in `expected` by increasing time, then runs through
// outputs[0] (synchronization signals), computing the index of the neuron
// responsible for the immediately preceding spike in `expected` (after
// sorting), which corresponds to the correct output, and the spikes in
// `outputs[1:]` immediately after. The time difference between the spikes in
// `outputs[1:]` and the sync signals gets appended to `ce_outputs`, a spike at
// time `0` gets appended to `ce_expected` in the position corresponding to the
// correct output, and a `kNoSpike` gets appended to `ce_expected` in other
// positions. This allows using CrossEntropyLoss to compute the loss on the
// output after each synchronization signal.
void CyclicToCrossEntropy(const Outputs &outputs, const Outputs &expected,
                          std::vector<std::vector<size_t>> *ce_indexes,
                          Outputs *ce_outputs, Outputs *ce_expected) {
  IHM_CHECK(outputs.size() == expected.size() + 1);
  // Pairs of the form (spike_time, neuron_idx) indicating when there is a
  // change in the expected (target) output.
  std::vector<std::pair<double, size_t>> output_switch_times;
  for (size_t i = 0; i < expected.size(); i++) {
    for (size_t j = 0; j < expected[i].size(); j++) {
      output_switch_times.emplace_back(expected[i][j], i);
    }
  }
  std::sort(output_switch_times.begin(), output_switch_times.end());
  (*ce_outputs).resize(expected.size());
  (*ce_expected).resize(expected.size());
  (*ce_indexes).resize(expected.size());
  // Index of the target (correct output). Assumed to be 0 before any input is
  // received.
  size_t current_target = 0;
  // Index of the next neuron that fires after the current pulse.
  size_t switch_idx = 0;
  // Index of each output neuron's first spike occurring after current pulse.
  std::vector<size_t> spike_indexes(expected.size());
  // For each output sync pulse...
  for (size_t pulse_idx = 0; pulse_idx < outputs[0].size(); pulse_idx++) {
    /// Find the first neuron in `expected` that spiked after current pulse.
    while (switch_idx < output_switch_times.size() &&
           output_switch_times[switch_idx].first <= outputs[0][pulse_idx]) {
      current_target = output_switch_times[switch_idx].second;
      switch_idx++;
    }
    // Make the target vector for loss function.
    for (size_t j = 0; j < expected.size(); j++) {
      (*ce_expected)[j].push_back(
          j == current_target ? 0.0 : PotentialFunction::kNoSpike);
    }
    // For each output neuron...
    for (size_t j = 0; j < expected.size(); j++) {
      // ...find the index of the spike that comes right after current pulse.
      while (spike_indexes[j] < outputs[1 + j].size() &&
             outputs[1 + j][spike_indexes[j]] < outputs[0][pulse_idx]) {
        spike_indexes[j]++;
      }
      bool has_spike = true;
      if (spike_indexes[j] == outputs[1 + j].size()) {
        has_spike = false;
      } else if (pulse_idx + 1 < outputs[0].size() &&
                 outputs[0][pulse_idx + 1] <=
                     outputs[1 + j][spike_indexes[j]]) {
        // Next spike comes after the next sync signal.
        has_spike = false;
      }
      (*ce_outputs)[j].push_back(has_spike ? outputs[1 + j][spike_indexes[j]] -
                                                 outputs[0][pulse_idx]
                                           : PotentialFunction::kNoSpike);
      (*ce_indexes)[j].push_back(has_spike ? spike_indexes[j]
                                           : outputs[1 + j].size());
    }
  }
}

}  // namespace

double CrossEntropyLoss::Loss(const Outputs &outputs,
                              const Outputs &expected) const {
  Outputs d_outputs;
  return LossDerivative(outputs, expected, &d_outputs);
}

double CrossEntropyLoss::LossDerivative(const Outputs &outputs,
                                        const Outputs &expected,
                                        Outputs *d_outputs) const {
  IHM_CHECK(!outputs.empty());
  IHM_CHECK(outputs.size() == expected.size());
  d_outputs->resize(outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    IHM_CHECK(expected[i].size() == expected[0].size());
    d_outputs->data()[i].resize(expected[i].size());
  }
  size_t num_outs = expected[0].size();
  double loss = 0;
  std::vector<double> exp_outputs(outputs.size());
  std::vector<double> exp_expected(outputs.size());
  auto output = [&](size_t i, size_t out) {
    return out < outputs[i].size() ? outputs[i][out]
                                   : PotentialFunction::kNoSpike;
  };
  for (size_t out = 0; out < num_outs; out++) {
    double min_output = output(0, out);
    double min_expected = expected[0][out];
    for (size_t i = 1; i < outputs.size(); i++) {
      min_output = std::min(min_output, output(i, out));
      min_expected = std::min(min_expected, expected[i][out]);
    }
    double exp_sum_out = 0;
    double exp_sum_exp = 0;
    for (size_t i = 0; i < outputs.size(); i++) {
      exp_outputs[i] = exp(-output(i, out) + min_output);
      exp_sum_out += exp_outputs[i];
      exp_expected[i] = exp(-expected[i][out] + min_expected);
      exp_sum_exp += exp_expected[i];
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      exp_outputs[i] /= exp_sum_out;
      exp_expected[i] /= exp_sum_exp;
      constexpr double kMinExpOutput = 1e-15;
      loss -= exp_expected[i] * log(std::max(exp_outputs[i], kMinExpOutput));
      // When exp_outputs[i] is 1.0, all other spikes are kNoSpikes, and the
      // value of outputs[i][out] does not influence the result. The usual
      // derivative formula is invalid in this case and is likely the result of
      // an invalid cancellation.
      if (exp_outputs[i] < 1.0) {
        d_outputs->data()[i][out] = -(exp_outputs[i] - exp_expected[i]);
      }
    }
  }
  return loss;
}

double CrossEntropyLoss::FracCorrect(const Outputs &outputs,
                                     const Outputs &expected) const {
  double num_correct = 0;
  size_t num_outputs = expected[0].size();
  for (size_t i = 0; i < expected.size(); i++) {
    IHM_CHECK(num_outputs == expected[i].size());
  }
  for (size_t i = 0; i < num_outputs; i++) {
    num_correct += IsCorrect(outputs, expected, i);
  }
  return num_correct / num_outputs;
}

double CyclicLoss::Loss(const Outputs &outputs, const Outputs &expected) const {
  Outputs d_outputs;
  return LossDerivative(outputs, expected, &d_outputs);
}

std::vector<std::vector<size_t>> CountSpikesPerCycle(const Outputs &outputs) {
  // Assumes `outputs[i]` is sorted.
  std::vector<size_t> spike_index(outputs.size(), 0);
  std::vector<std::vector<size_t>> num_spikes_per_cycle(
      outputs.size(), std::vector<size_t>(outputs[0].size(), 0));
  // The spikes before the first sync pulse don't matter.
  for (size_t cycle_ind = 0; cycle_ind < outputs[0].size(); cycle_ind++) {
    ++num_spikes_per_cycle[0][cycle_ind];
    double low_bound = outputs[0][cycle_ind];
    double upper_bound = cycle_ind >= outputs[0].size() - 1
                             ? PotentialFunction::kNoSpike
                             : outputs[0][cycle_ind + 1];
    for (size_t output_ind = 1; output_ind < outputs.size(); output_ind++) {
      // Skip any spikes before the last sync.
      while (spike_index[output_ind] < outputs[output_ind].size() &&
             outputs[output_ind][spike_index[output_ind]] < low_bound) {
        ++spike_index[output_ind];
      }
      // Count spikes before last and current sync.
      while (spike_index[output_ind] < outputs[output_ind].size() &&
             outputs[output_ind][spike_index[output_ind]] < upper_bound) {
        ++spike_index[output_ind];
        ++num_spikes_per_cycle[output_ind][cycle_ind];
      }
    }
  }
  return num_spikes_per_cycle;
}

double CyclicLoss::LossDerivative(const Outputs &outputs,
                                  const Outputs &expected,
                                  Outputs *d_outputs) const {
  Outputs ce_outputs, ce_expected;
  std::vector<std::vector<size_t>> ce_indexes;
  CyclicToCrossEntropy(outputs, expected, &ce_indexes, &ce_outputs,
                       &ce_expected);
  Outputs ce_d_outputs;
  double loss =
      CrossEntropyLoss().LossDerivative(ce_outputs, ce_expected, &ce_d_outputs);

  (*d_outputs).resize(outputs.size());
  for (size_t i = 0; i < d_outputs->size(); i++) {
    (*d_outputs)[i].resize(outputs[i].size());
  }

  // Penalty for sparse sync pulses.
  double last_sync = 0;
  for (size_t i = 0; i < outputs[0].size(); i++) {
    double sync = outputs[0][i];
    if (sync > last_sync + cycle_length_) {
      double gap = sync - last_sync - cycle_length_;
      loss += sync_gap_penalty_multiplier_ * gap * gap;
      (*d_outputs)[0][i] += sync_gap_penalty_multiplier_ * 2 * gap;
      if (i != 0) {
        (*d_outputs)[0][i - 1] -= sync_gap_penalty_multiplier_ * 2 * gap;
      }
    }
    last_sync = sync;
  }
  double last_expected_output = 0;
  for (size_t i = 0; i < expected.size(); i++) {
    if (!expected[i].empty()) {
      last_expected_output =
          std::max(last_expected_output,
                   *std::max_element(expected[i].begin(), expected[i].end()));
    }
  }
  if (last_sync < last_expected_output) {
    double gap = last_expected_output - last_sync;
    loss += missing_sync_penalty_multiplier_ * gap * gap;
    (*d_outputs)[0].back() -= missing_sync_penalty_multiplier_ * gap;
  }

  // Penalty when outputs spike more than once per cycle.
  std::vector<std::vector<size_t>> num_spikes_per_cycle =
      CountSpikesPerCycle(outputs);
  for (size_t output_ind = 1; output_ind < outputs.size(); output_ind++) {
    for (size_t cycle_ind = 0;
         cycle_ind < num_spikes_per_cycle[output_ind].size(); ++cycle_ind) {
      if (num_spikes_per_cycle[output_ind][cycle_ind] > 1) {
        double diff = num_spikes_per_cycle[output_ind][cycle_ind] - 1;
        loss += diff * wrong_num_spikes_per_cycle_penalty_;
        for (size_t spike_ind = ce_indexes[output_ind - 1][cycle_ind] + 1;
             spike_ind < ce_indexes[output_ind - 1][cycle_ind] +
                             num_spikes_per_cycle[output_ind][cycle_ind];
             spike_ind++) {
          (*d_outputs)[output_ind][spike_ind] -=
              wrong_num_spikes_per_cycle_penalty_;
        }
      }
    }
  }

  // Cross-entropy loss derivative.
  for (size_t i = 0; i < ce_d_outputs.size(); i++) {
    for (size_t j = 0; j < ce_d_outputs[i].size(); j++) {
      // No spike after this sync signal: no derivative to propagate.
      if (ce_indexes[i][j] == outputs[1 + i].size()) continue;
      (*d_outputs)[0][j] -= ce_d_outputs[i][j];
      (*d_outputs)[1 + i][ce_indexes[i][j]] += ce_d_outputs[i][j];
    }
  }
  return loss;
}

double CyclicLoss::FracCorrect(const Outputs &outputs,
                               const Outputs &expected) const {
  Outputs ce_outputs, ce_expected;
  std::vector<std::vector<size_t>> ce_indexes;
  CyclicToCrossEntropy(outputs, expected, &ce_indexes, &ce_outputs,
                       &ce_expected);
  return CrossEntropyLoss().FracCorrect(ce_outputs, ce_expected);
}

}  // namespace ihmehimmeli
