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

#ifndef IHMEHIMMELI_TEMPCODING_UTIL_H_
#define IHMEHIMMELI_TEMPCODING_UTIL_H_

#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace ihmehimmeli {

enum class LogSeverity { INFO = 0, WARNING = 1, ERROR = 2, FATAL = 3 };

namespace internal {

static constexpr char kLogInitials[] = {"IWEF"};

static std::string log_time_string() {
  absl::Time t1 = absl::Now();
  absl::TimeZone utc = absl::LocalTimeZone();
  return absl::FormatTime(t1, utc);
}

[[noreturn]] static void fail_check_and_die(const std::string &expression,
                                            const std::string &location,
                                            int line,
                                            const std::string &message = "") {
  std::cerr << "IHM_CHECK(" << expression << ") failed at " << location << ":"
            << line;
  if (message != "") std::cerr << ": " << message;
  std::cerr << std::endl;
  abort();
}

}  // namespace internal

#define IHM_LOG(severity, message)                                             \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<decltype((severity)), ::ihmehimmeli::LogSeverity>::value, \
        "Wrong severity type");                                                \
                                                                               \
    std::cerr                                                                  \
        << ::ihmehimmeli::internal::kLogInitials[static_cast<int>((severity))] \
        << ::ihmehimmeli::internal::log_time_string() << " " << __FILE__       \
        << ":" << __LINE__ << "] " << (message) << std::endl;                  \
    if ((severity) == ::ihmehimmeli::LogSeverity::FATAL) {                     \
      abort();                                                                 \
    }                                                                          \
  }

#define IHM_CHECK(condition, ...)                                     \
  {                                                                   \
    while (!(condition)) {                                            \
      ::ihmehimmeli::internal::fail_check_and_die(                    \
          #condition, __FILE__, __LINE__, absl::StrCat(__VA_ARGS__)); \
    }                                                                 \
  }

template <typename T>
static std::string VecToString(const std::vector<T> &v) {
  return absl::StrCat("[", absl::StrJoin(v, ", "), "]");
}

template <typename T>
static T IPow(T base, uint32_t exp) {
  T result = 1;
  T mul = base;

  while (exp) {
    if (exp & 1) result *= mul;
    mul *= mul;
    exp >>= 1;
  }

  return result;
}

template <typename T>
class BasicStats {
 public:
  BasicStats(const std::vector<T> &v) {
    if (v.empty()) {
      mean_ = median_ = stddev_ = 0;
      return;
    }

    mean_ = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0;
    for (const T &x : v) {
      var += (x - mean_) * (x - mean_);
    }
    var /= (v.size() - 1.0);
    stddev_ = std::sqrt(var);

    std::vector<T> v1(v);
    std::sort(v1.begin(), v1.end());
    const int n_mid = v.size() / 2;
    if (v.size() & 1) {
      median_ = v1[n_mid];
    } else {
      median_ = 0.5 * (v1[n_mid - 1] + v1[n_mid]);
    }
  }

  double mean() const { return mean_; }
  double median() const { return median_; }
  double stddev() const { return stddev_; }

 private:
  double mean_;
  double median_;
  double stddev_;
};

}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_TEMPCODING_UTIL_H_
