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

#ifndef IHMEHIMMELI_COMMON_MNIST_LOADER_H_
#define IHMEHIMMELI_COMMON_MNIST_LOADER_H_

#include <string>
#include <vector>

namespace ihmehimmeli {
void LoadMNISTDataAndLabels(const std::string &data_path,
                            const std::string &labels_path, int *n_inputs,
                            int *n_examples,
                            std::vector<float> *mnist_all_data_float,
                            std::vector<int> *mnist_labels);
}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_COMMON_MNIST_LOADER_H_
