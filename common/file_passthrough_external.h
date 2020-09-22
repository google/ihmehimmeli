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

#ifndef IHMEHIMMELI_COMMON_FILE_PASSTHROUGH_EXTERNAL_H_
#define IHMEHIMMELI_COMMON_FILE_PASSTHROUGH_EXTERNAL_H_

#include <stdint.h>

#include <cstdio>
#include <string>

#include "absl/strings/string_view.h"

namespace ihmehimmeli {
namespace file {

// A small abstraction layer which intermediates the file API calls we use for
// different possible backends. It's only meant to be feature complete to the
// extent to which we use it.

class CIhmFile {
 public:
  CIhmFile() : file_(nullptr){};
  ~CIhmFile();

  CIhmFile(const CIhmFile &) = delete;
  CIhmFile &operator=(const CIhmFile &) = delete;

  CIhmFile(CIhmFile &&other) : file_(nullptr) {
    file_ = other.file_;
    other.file_ = nullptr;
  }

  CIhmFile &operator=(CIhmFile &&other) {
    if (this != &other) {
      if (file_) {
        fclose(file_);
      }
      file_ = other.file_;
      other.file_ = nullptr;
    }
    return *this;
  }

  bool Write(absl::string_view string, int64_t *bytes_written);
  bool WriteString(absl::string_view string);
  bool Open(absl::string_view path, absl::string_view mode);
  bool Close();
  bool Read(int64_t max_to_read, std::string *str);
  bool ReadWholeFileToString(std::string *str);

 private:
  FILE *file_;
};

using IhmFile = CIhmFile;

std::string JoinPath(absl::string_view p1, absl::string_view p2);
bool Open(absl::string_view path, absl::string_view mode, IhmFile *output);
IhmFile OpenOrDie(absl::string_view path, absl::string_view mode);

}  // namespace file
}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_COMMON_FILE_PASSTHROUGH_EXTERNAL_H_
