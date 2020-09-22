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

#ifndef IHMEHIMMELI_COMMON_LAMBERTW_H_
#define IHMEHIMMELI_COMMON_LAMBERTW_H_
#include <cmath>

namespace ihmehimmeli {
// Computes values on the principal branch of the Lambert W function.
// This is meant to be used for x > -1/e.

// Minimum argument for the main branch of the Lambert W function.
constexpr double kMinLambertArg = -1.0 / M_E;

// Maximum argument for which LambertW0 produces a valid result.
constexpr double kMaxLambertArg = 1.7976131e+308;

bool LambertW0(double x, double *output);
}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_COMMON_LAMBERTW_H_
