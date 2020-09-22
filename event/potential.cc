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

#include "event/potential.h"

#include <cmath>

#include "absl/strings/str_format.h"
#include "common/lambertw.h"
#include "common/util.h"

namespace ihmehimmeli {

size_t AlphaPotential::StateSize() const { return 5; }

double AlphaPotential::AddSpike(double t, double w, double firing_threshold,
                                double *state) const {
  double &A = state[0];
  double &B = state[1];
  double &W = state[2];

  // In long-running networks, `t * decay_rate` might become big. If it becomes
  // larger than approximately 700, in particular, its exponential is outside of
  // the range representable by `double`, which causes `A` and `B` to become
  // Inf and the whole function to produce NaNs.
  // Thus, we subtract the time of the first spike from all the subsequent
  // spikes and add it back in the end.
  double &base_time = state[3];

  // Weird usage of doubles to keep bool state.
  // TODO: modify the PotentialFunction interface to allow to use a
  // struct for state. This would allow using a proper `bool` here.
  double &has_one_spike = state[4];

  if (has_one_spike < 0.5) {
    base_time = t;
    has_one_spike = 1.0;
  }

  // The gap between the first and the current spike received by this neuron is
  // too large. Reset the state.
  // TODO: this is not entirely correct, as a sequence of frequent input
  // spikes that don't cause an output spike will be incorrectly discarded by
  // this if.
  constexpr double kMaxTimeGap = 500;  // exp(500) ~ 10e200, which is safe.
  if (decay_rate * (t - base_time) > kMaxTimeGap) {
    InitState(state);
    base_time = t;
    has_one_spike = 1.0;
  }

  t -= base_time;

  // All checks in this function are negated to cause early returns in case of
  // NaNs.

  const double w_exp_z = w * std::exp(decay_rate * t);
  A += w_exp_z;
  B += w_exp_z * t;

  // The value of the first derivative of the activation function in the
  // intersection point with the fire threshold is given by *A multiplied by a
  // never-negative value. Thus, if *A is negative the intersection will be in
  // a decreasing-potential area, and thus not a spike.
  if (!(A >= 0)) {
    return kNoSpike;
  }

  // Compute Lambert W argument for solving the threshold crossing.
  const double lambert_arg =
      -decay_rate * firing_threshold / A * std::exp(decay_rate * B / A);

  if (!(lambert_arg >= kMinLambertArg) || !(lambert_arg <= kMaxLambertArg)) {
    return kNoSpike;
  }

  IHM_CHECK(LambertW0(lambert_arg, &W),
            absl::StrFormat("Error computing Lambert W on: %f", lambert_arg));

  if (!(W <= kNoSpike)) return kNoSpike;

  return base_time + B / A - W / decay_rate;
}

void AlphaPotential::Derivatives(double t, double w, double firing_threshold,
                                 double output, const double *state,
                                 double *input_d, double *w_d) const {
  const double A = state[0];
  // const double B = state[1];
  const double W = state[2];
  const double K = decay_rate;
  const double base_time = state[3];
  t -= base_time;
  const double e_K_tp = std::exp(K * t);
  const double mul = e_K_tp / (A * (1.0f + W));
  const double tp_BA_WK = t - (output - base_time);  // t - B / A + W / K
  *input_d = w * (K * tp_BA_WK + 1) * mul;
  *w_d = tp_BA_WK * mul;
}

}  // namespace ihmehimmeli
