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

#include "tempcoding/lambertw.h"

#include <cmath>
#include <cstdlib>

// Implements a LambertW function, 0-branch approximation. Based partly on:
// https://arxiv.org/pdf/1003.1628.pdf

namespace {

double LambertW0InitialGuess(double x) {
  constexpr double kNearBranchCutoff = -0.3235;
  constexpr double kE = 2.718281828459045;

  // Sqrt approximation near branch cutoff.
  if (x < kNearBranchCutoff) return -1.0 + std::sqrt(2.0 * (1 + kE * x));

  // Taylor series between [-1/e and 1/e].
  if (x > kNearBranchCutoff && x < -kNearBranchCutoff) {
    return x * (1 + x * (-1 + x * (3.0 / 2.0 - 8.0 / 3.0 * x)));
  }

  // Series of piecewise linear approximations.
  if (x < 0.6) return 0.23675531078855933 + (x - 0.3) * 0.5493610866617109;
  if (x < 0.8999999999999999)
    return 0.4015636367870726 + (x - 0.6) * 0.4275644294878729;
  if (x < 1.2)
    return 0.5298329656334344 + (x - 0.8999999999999999) * 0.3524368357714513;
  if (x < 1.5) return 0.6355640163648698 + (x - 1.2) * 0.30099113800452154;
  if (x < 1.8) return 0.7258613577662263 + (x - 1.5) * 0.2633490154764343;
  if (x < 2.0999999999999996)
    return 0.8048660624091566 + (x - 1.8) * 0.2345089875713013;
  if (x < 2.4)
    return 0.8752187586805469 + (x - 2.0999999999999996) * 0.2116494532726034;
  if (x < 2.6999999999999997)
    return 0.938713594662328 + (x - 2.4) * 0.19305046534383152;
  if (x < 2.9999999999999996)
    return 0.9966287342654774 + (x - 2.6999999999999997) * 0.17760053566187495;

  // Asymptotic approximation.
  const double l = std::log(x);
  const double ll = std::log(l);
  return l - ll + ll / l;
}
}  // namespace

namespace ihmehimmeli {

bool LambertW0(double x, double *output) {
  constexpr double kReciprocalE = 0.36787944117;
  constexpr double kDesiredAbsoluteDifference = 1e-3;
  constexpr int kNumMaxIters = 10;

  if (x <= -kReciprocalE) return false;
  if (x == 0.0) {
    *output = 0;
    return true;
  }
  if (x == -kReciprocalE) {
    *output = -1.0;
    return true;
  }

  // Current guess.
  double w_n = LambertW0InitialGuess(x);
  bool have_convergence = false;

  // Fritsch iteration.
  for (int i = 0; i < kNumMaxIters; i++) {
    const double z_n = std::log(x / w_n) - w_n;
    const double q_n = 2.0 * (1.0 + w_n) * (1.0 + w_n + 2.0 / 3.0 * z_n);
    const double e_n = (z_n / (1.0 + w_n)) * ((q_n - z_n) / (q_n - 2.0 * z_n));
    w_n *= (1.0 + e_n);

    // Done this way as the log is the expensive part above.
    if (std::abs(z_n) < kDesiredAbsoluteDifference) {
      have_convergence = true;
      break;
    }
  }

  *output = w_n;
  return have_convergence;
}

}  // namespace ihmehimmeli
