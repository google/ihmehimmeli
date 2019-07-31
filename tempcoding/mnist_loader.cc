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

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "tempcoding/file_passthrough.h"
#include "tempcoding/util.h"

uint32_t ReadBigEndian(const uint8_t *ptr) {
  return ((static_cast<uint32_t>(ptr[3])) |
          (static_cast<uint32_t>(ptr[2]) << 8) |
          (static_cast<uint32_t>(ptr[1]) << 16) |
          (static_cast<uint32_t>(ptr[0]) << 24));
}

void LoadMNISTDataFromBytes(const std::string &data_bytes,
                            const std::string &labels_bytes, int *num_inputs,
                            int *num_examples,
                            std::vector<float> *mnist_all_data_float,
                            std::vector<int> *mnist_labels) {
  IHM_CHECK(data_bytes.size() > 4 * sizeof(uint32_t));
  IHM_CHECK(labels_bytes.size() > 2 * sizeof(uint32_t));

  const uint8_t *p_data = reinterpret_cast<const uint8_t *>(data_bytes.data());
  const uint8_t *p_labels =
      reinterpret_cast<const uint8_t *>(labels_bytes.data());

  uint32_t magic = ReadBigEndian(p_data);
  p_data += sizeof(uint32_t);
  uint32_t num_images = ReadBigEndian(p_data);
  p_data += sizeof(uint32_t);
  uint32_t num_rows = ReadBigEndian(p_data);
  p_data += sizeof(uint32_t);
  uint32_t num_cols = ReadBigEndian(p_data);
  p_data += sizeof(uint32_t);

  IHM_CHECK(magic == 2051);
  IHM_CHECK(num_rows == 28);
  IHM_CHECK(num_cols == 28);

  uint32_t labels_magic = ReadBigEndian(p_labels);
  p_labels += sizeof(uint32_t);
  uint32_t num_labels = ReadBigEndian(p_labels);
  p_labels += sizeof(uint32_t);

  IHM_CHECK(labels_magic == 2049);
  IHM_CHECK(num_labels == num_images);

  const size_t image_size = num_rows * num_cols;
  *num_inputs = image_size;
  *num_examples = num_images;

  mnist_all_data_float->reserve(num_images * image_size);
  mnist_labels->reserve(num_images);

  for (int i = 0; i < num_images; ++i) {
    mnist_labels->push_back(static_cast<int>(*p_labels));

    for (int j = 0; j < image_size; ++j) {
      mnist_all_data_float->push_back(static_cast<float>(*p_data) / 256.0);
      p_data++;
    }

    p_labels++;
  }
}

namespace ihmehimmeli {

void LoadMNISTDataAndLabels(const std::string &data_path,
                            const std::string &labels_path, int *n_inputs,
                            int *n_examples,
                            std::vector<float> *mnist_all_data_float,
                            std::vector<int> *mnist_labels) {
  std::string data_bytes;
  {
    file::IhmFile file = file::OpenOrDie(data_path, "r");
    IHM_CHECK(file.ReadWholeFileToString(&data_bytes));
    IHM_CHECK(file.Close());
  }

  std::string labels_bytes;
  {
    file::IhmFile file = file::OpenOrDie(labels_path, "r");
    IHM_CHECK(file.ReadWholeFileToString(&labels_bytes));
    IHM_CHECK(file.Close());
  }

  return LoadMNISTDataFromBytes(data_bytes, labels_bytes, n_inputs, n_examples,
                                mnist_all_data_float, mnist_labels);
}

}  // namespace ihmehimmeli
