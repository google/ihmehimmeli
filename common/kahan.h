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
#ifndef IHMEHIMMELI_COMMON_KAHAN_H_
#define IHMEHIMMELI_COMMON_KAHAN_H_

#include <cstdlib>

namespace ihmehimmeli {

template <typename Float>
class Kahan {
 public:
  Kahan() : Kahan(0) {}
  explicit Kahan(Float value) : value_(value), compensation_(0) {}
  explicit operator Float() const { return value_ + compensation_; }

  // Kahan summation algorithm (Neumaier variant).
  Kahan& operator+=(Float v) {
    Float t = value_ + v;
    if (std::abs(value_) >= std::abs(v)) {
      compensation_ += (value_ - t) + v;
    } else {
      compensation_ += (v - t) + value_;
    }
    value_ = t;
    return *this;
  }

  Kahan& operator-=(Float v) {
    *this += -v;
    return *this;
  }

  Kahan& operator=(Float v) {
    value_ = v;
    compensation_ = 0;
    return *this;
  }

 private:
  Float value_;
  Float compensation_;
};

}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_COMMON_KAHAN_H_
