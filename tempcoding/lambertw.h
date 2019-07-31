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

#ifndef IHMEHIMMELI_LAMBERTW_H_
#define IHMEHIMMELI_LAMBERTW_H_

namespace ihmehimmeli {
// Computes values on the principal branch of the Lambert W function.
// This is meant to be used for x > -1/e.

bool LambertW0(double x, double *output);
}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_LAMBERTW_H_
