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

#include "event/spike_queue.h"

namespace ihmehimmeli {

void SpikeQueue::Init(uint32_t num_neurons) {
  spike_positions_.resize(num_neurons, kMissingSpike);
}

void SpikeQueue::Heapify(uint32_t pos) {
  // Bubble up if needed.
  while (pos != 0) {
    if (spikes_[pos] < spikes_[(pos - 1) / 2]) {
      SwapSpikesAtPosition(pos, (pos - 1) / 2);
      pos = (pos - 1) / 2;
    } else {
      break;
    }
  }
  // Bubble down if needed.
  while (true) {
    uint32_t child1 = pos * 2 + 1;
    if (child1 >= spikes_.size()) break;
    uint32_t smallest = child1;
    if (child1 + 1 < spikes_.size() &&
        spikes_[child1 + 1] < spikes_[smallest]) {
      smallest = child1 + 1;
    }
    if (spikes_[smallest] < spikes_[pos]) {
      SwapSpikesAtPosition(smallest, pos);
      pos = smallest;
    } else {
      break;
    }
  }
}

void SpikeQueue::Remove(uint32_t neuron_id) {
  uint32_t position = spike_positions_[neuron_id];
  if (position == kMissingSpike) return;
  SwapSpikesAtPosition(position, spikes_.size() - 1);
  Pop();
  if (position != spikes_.size()) Heapify(position);
}

void SpikeQueue::Add(Spike spike) {
  Push(spike);
  Heapify(spikes_.size() - 1);
}

bool SpikeQueue::Pop(Spike* spike) {
  if (spikes_.empty()) return false;
  *spike = spikes_[0];
  SwapSpikesAtPosition(0, spikes_.size() - 1);
  Pop();
  Heapify(0);
  return true;
}

}  // namespace ihmehimmeli
