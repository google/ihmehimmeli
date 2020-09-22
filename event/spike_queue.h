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

#ifndef IHMEHIMMELI_EVENT_SPIKE_QUEUE_H_
#define IHMEHIMMELI_EVENT_SPIKE_QUEUE_H_

#include <cstdint>
#include <limits>
#include <queue>
#include <tuple>

namespace ihmehimmeli {

// Holds pending spike information: timing of the spike and ID of the neuron
// that caused it.
struct Spike {
  double timestamp;
  // Neuron that is about to spike.
  uint32_t source_neuron_id;

  bool operator<(const Spike& other) const {
    return std::make_tuple(timestamp, source_neuron_id) <
           std::make_tuple(other.timestamp, other.source_neuron_id);
  }
};

// A priority queue of spike events that only holds up to one spike per neuron.
// Can hold at most pow(2, 32)-1 spikes at the same time.
class SpikeQueue {
 public:
  // Initializes the queue so that it will be able to hold spikes from
  // `num_neurons` distinct neurons.
  void Init(uint32_t num_neurons);

  // Removes spikes from the given neuron, if any.
  void Remove(uint32_t neuron_id);

  // Adds a spike from the given neuron at the given time. There should be no
  // other spike from that neuron already in the queue.
  void Add(Spike spike);

  // Gets the next enqueued spike, if any. If no enqueued spike is present,
  // returns false, other wise returns true.
  bool Pop(Spike* spike);

 private:
  // Queue of spike events.
  std::vector<Spike> spikes_;

  // Position of spikes from each neuron.
  std::vector<uint32_t> spike_positions_;

  void SwapSpikesAtPosition(uint32_t a, uint32_t b) {
    std::swap(spikes_[a], spikes_[b]);
    spike_positions_[spikes_[a].source_neuron_id] = a;
    spike_positions_[spikes_[b].source_neuron_id] = b;
  }

  // Heap operations.
  void Pop() {
    spike_positions_[spikes_.back().source_neuron_id] = kMissingSpike;
    spikes_.pop_back();
  }

  void Push(Spike spike) {
    spike_positions_[spike.source_neuron_id] = spikes_.size();
    spikes_.push_back(std::move(spike));
  }

  void Heapify(uint32_t pos);

  static constexpr uint32_t kMissingSpike =
      std::numeric_limits<uint32_t>::max();
};

};  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_EVENT_SPIKE_QUEUE_H_
