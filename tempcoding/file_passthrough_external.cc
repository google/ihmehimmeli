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

#include "tempcoding/file_passthrough_external.h"

#include <stddef.h>

#include <cstdio>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tempcoding/util.h"

namespace ihmehimmeli {
namespace file {

CIhmFile::~CIhmFile() {
  if (file_ != nullptr) {
    fclose(file_);
  }
}

bool CIhmFile::Write(absl::string_view string, int64_t *bytes_written) {
  *bytes_written = fwrite(string.data(), 1, string.length(), file_);
  return true;
}

bool CIhmFile::WriteString(absl::string_view string) {
  return fwrite(string.data(), 1, string.length(), file_) == string.length();
}

bool CIhmFile::Open(absl::string_view path, absl::string_view mode) {
  file_ = fopen(std::string(path).c_str(), std::string(mode).c_str());
  return file_ != nullptr;
}

bool CIhmFile::Close() {
  if (fclose(file_) == 0) {
    file_ = nullptr;
    return true;
  }
  return false;
}

bool CIhmFile::Read(int64_t max_to_read, std::string *str) {
  std::vector<char> buf(max_to_read);
  size_t num_read = fread(buf.data(), 1, max_to_read, file_);
  buf.resize(num_read);
  str->assign(buf.begin(), buf.end());
  if (num_read < max_to_read) {
    if (feof(file_)) return true;
    return false;
  }

  return true;
}

bool CIhmFile::ReadWholeFileToString(std::string *str) {
  fseek(file_, 0, SEEK_END);
  long int size = ftell(file_);
  rewind(file_);

  return Read(size, str);
}

std::string JoinPath(absl::string_view p1, absl::string_view p2) {
  if (!p1.empty() && p1.back() == '/') p1 = p1.substr(0, p1.size() - 1);
  if (!p2.empty() && p2.front() == '/') p2 = p2.substr(1, p2.size() - 1);
  return absl::StrCat(p1, "/", p2);
};

bool Open(absl::string_view path, absl::string_view mode, CIhmFile *output) {
  CIhmFile file;
  if (file.Open(path, mode)) {
    *output = std::move(file);
    return true;
  }

  return false;
}

CIhmFile OpenOrDie(absl::string_view path, absl::string_view mode) {
  CIhmFile file;
  IHM_CHECK(file.Open(path, mode));
  return file;
}

}  // namespace file
}  // namespace ihmehimmeli
