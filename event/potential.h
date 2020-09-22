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

#ifndef IHMEHIMMELI_EVENT_POTENTIAL_H
#define IHMEHIMMELI_EVENT_POTENTIAL_H
#include <stdint.h>
#include <string.h>

#include <limits>
#include <vector>

namespace ihmehimmeli {

// TODO: does it make sense to consider networks with mixed activation
// functions?
class PotentialFunction {
 public:
  static constexpr double kNoSpike = std::numeric_limits<double>::max();
  virtual ~PotentialFunction() = default;

  // Size of the state that should be allocated per-neuron.
  virtual size_t StateSize() const = 0;

  // Update the `state` with a new input spike at time `t` and of weight `w` and
  // return the time of the output spike; if none, return kNoSpike.
  virtual double AddSpike(double t, double w, double firing_threshold,
                          double *state) const = 0;

  // Compute the derivative of the input spike (`input_d`) and the weight
  // (`w_d`) as a function of the input (`t`), weight (`w`), firing threshold,
  // spike time (`output`) and `state` at the moment of the spike. The `state`
  // should point to the same memory that was updated by `AddSpike` when it
  // returned the spike time that is passed as `output`.
  virtual void Derivatives(double t, double w, double firing_threshold,
                           double output, const double *state, double *input_d,
                           double *w_d) const = 0;

  // Initialize state. Defaults to setting it to 0.
  virtual void InitState(double *state) const {
    memset(state, 0, StateSize() * sizeof(*state));
  }
};

struct AlphaPotential : public PotentialFunction {
  AlphaPotential(double decay_rate = 0.1) { this->decay_rate = decay_rate; }
  size_t StateSize() const override final;
  double AddSpike(double t, double w, double firing_threshold,
                  double *state) const override final;
  void Derivatives(double t, double w, double firing_threshold, double output,
                   const double *state, double *input_d,
                   double *w_d) const override final;
  double decay_rate = 1.0;
};
}  // namespace ihmehimmeli

#endif
