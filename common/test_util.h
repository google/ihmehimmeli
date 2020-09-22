/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef IHMEHIMMELI_COMMON_TEST_UTIL_H_
#define IHMEHIMMELI_COMMON_TEST_UTIL_H_

#include <algorithm>

#include "testing/base/public/gunit.h"
#include "common/kahan.h"

static constexpr double kEps = 1e-5;

static constexpr double kDoubleNearEps = 1e-6;

testing::AssertionResult IsDerivativeClose(double derivative, double fxh,
                                           double fx, double h,
                                           double error = 1e-5) {
  double approx_derivative = (fxh - fx) / h;
  double max = std::max(std::abs(approx_derivative), std::abs(derivative));
  if (max < error) {
    if (std::abs(approx_derivative - derivative) > error) {
      return testing::AssertionFailure()
             << "Absolute error too high: " << derivative << " vs "
             << approx_derivative;
    }
    return testing::AssertionSuccess();
  }
  if (std::abs(approx_derivative - derivative) > error * max) {
    return testing::AssertionFailure()
           << "Relative error "
           << std::abs(approx_derivative - derivative) / max
           << " too high: " << derivative << " vs " << approx_derivative;
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult IsDerivativeClose(
    ihmehimmeli::Kahan<double> derivative, double fxh, double fx, double h,
    double error) {
  return IsDerivativeClose(static_cast<double>(derivative), fxh, fx, h, error);
}

#endif  // IHMEHIMMELI_COMMON_TEST_UTIL_H_
