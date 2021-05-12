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

#ifndef IHMEHIMMELI_EVENT_ADAM_H
#define IHMEHIMMELI_EVENT_ADAM_H

#include <vector>

#include "common/kahan.h"
#include "common/util.h"

namespace ihmehimmeli {

// Keeps momentum and variance information to run ADAM update steps on a given
// set of parameters.
struct AdamUpdater {
  // Update momentum and variance information using the given gradient `d`,
  // which is the result of accumulating `batch_size` gradients from network
  // runs, produce parameter updates using the learning rate `lr` and update the
  // parameted passed in `values`.
  void Update(const std::vector<Kahan<double>> &d, size_t batch_size, double lr,
              std::vector<double> *values) {
    if (momentum_.empty()) {
      momentum_.resize(d.size());
      var_.resize(d.size());
    }
    epoch_++;
    double inv_batch_size = 1.0f / batch_size;
    double m_scale = 1.0f / (1.0f - IPow(kB1, epoch_));
    double v_scale = 1.0f / (1.0f - IPow(kB2, epoch_));
    for (size_t j = 0; j < d.size(); j++) {
      double scaled_d = static_cast<double>(d[j]) * inv_batch_size;
      momentum_[j] = kB1 * momentum_[j] + ckB1 * scaled_d;
      var_[j] = kB2 * var_[j] + ckB2 * scaled_d * scaled_d;
      values->data()[j] -=
          lr * momentum_[j] * m_scale / std::sqrt(var_[j] * v_scale + kEps);
    }
  }

 private:
  size_t epoch_ = 0;
  std::vector<double> momentum_;
  std::vector<double> var_;
  static constexpr double kB1 = 0.9;
  static constexpr double kB2 = 0.999;
  static constexpr double kEps = 1e-8;
  static constexpr double ckB1 = 1.0f - kB1;
  static constexpr double ckB2 = 1.0f - kB2;
};

}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_EVENT_ADAM_H
