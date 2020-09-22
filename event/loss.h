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

#ifndef IHMEHIMMELI_EVENT_LOSS_H
#define IHMEHIMMELI_EVENT_LOSS_H
#include <vector>

namespace ihmehimmeli {

using Outputs = std::vector<std::vector<double>>;

// Returns a vector of vectors of non-negative integers representing the
// number of spikes each output neuron produced in the intervals given by
// output sync pulses; the first dimension is the output neuron index and
// the second dimension is the cycle index. The first output neuron in both
// `outputs` and the returned vector corresponds to the sync pulses. Assumes
// `outputs[i]` is sorted.
std::vector<std::vector<size_t>> CountSpikesPerCycle(const Outputs &outputs);

class LossFunction {
 public:
  // A single output neuron may spike multiple times. We use a vector to
  // represent all the spikes produced by each neuron.
  virtual ~LossFunction() = default;

  virtual double Loss(const Outputs &outputs,
                      const Outputs &expected) const = 0;

  virtual double LossDerivative(const Outputs &outputs, const Outputs &expected,
                                Outputs *d_outputs) const = 0;

  // Returns the fraction of correct outputs.
  virtual double FracCorrect(const Outputs &outputs,
                             const Outputs &expected) const = 0;
};

// Also applies per-layer "spiking softmax" to both outputs and expected.
class CrossEntropyLoss : public LossFunction {
 public:
  double Loss(const Outputs &outputs, const Outputs &expected) const override;

  double LossDerivative(const Outputs &outputs, const Outputs &expected,
                        Outputs *d_outputs) const override;

  double FracCorrect(const Outputs &outputs,
                     const Outputs &expected) const override;
};

// Loss function for a network that produces periodic signals. The first output
// is interpreted as a start-of-event synchronization signal. Cross-entropy loss
// is then applied to the spikes that come just after this synchronization
// signal, with the "correct" output being defined by the last neuron in
// `expected` that spiked before the current output synchronization signal. A
// penalty is applied if synchronization signals are too sparse. One further
// penalty is applied if the last synchronization signal happens before the last
// spike in `expected`.
class CyclicLoss : public LossFunction {
 public:
  CyclicLoss(double cycle_length = 1.0) : cycle_length_(cycle_length) {}

  void SetSyncGapPenaltyMultiplier(double penalty) {
    sync_gap_penalty_multiplier_ = penalty;
  }
  void SetMissingSyncPenaltyMultiplier(double penalty) {
    missing_sync_penalty_multiplier_ = penalty;
  }
  void SetWrongNumSpikesPenaltyMultiplier(double penalty) {
    wrong_num_spikes_per_cycle_penalty_ = penalty;
  }

  double Loss(const Outputs &outputs, const Outputs &expected) const override;

  double LossDerivative(const Outputs &outputs, const Outputs &expected,
                        Outputs *d_outputs) const override;

  double FracCorrect(const Outputs &outputs,
                     const Outputs &expected) const override;

 private:
  double cycle_length_ = 1.0;
  double sync_gap_penalty_multiplier_ = 1.0;
  double missing_sync_penalty_multiplier_ = 1000.0;

  // This penalty is added if the number of spikes per cycle of an output neuron
  // is larger than 1. The penalty added is proportional to the absolute
  // difference from 1.
  double wrong_num_spikes_per_cycle_penalty_ = 0.0;
};

}  // namespace ihmehimmeli

#endif
