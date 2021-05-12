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

#ifndef IHMEHIMMELI_TEMPCODING_RUNNER_H_
#define IHMEHIMMELI_TEMPCODING_RUNNER_H_

#include <cstddef>
#include <tuple>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "common/data_parallel.h"
#include "common/util.h"
#include "tempcoding/spiking_problem.h"
#include "tempcoding/tempcoder.h"

namespace ihmehimmeli {

struct RunnerOptions {
  bool compute_train_accuracy = false;
  bool print_train_spike_stats = false;
  bool print_test_spike_stats = false;
  bool print_network = false;
  bool print_network_stats = false;
  std::string train_embeddings_filename = "";
  std::string test_embeddings_filename = "";
};

// Runs or trains a Tempcoder on a SpikingProblem.
class Runner {
 public:
  Runner(Tempcoder* tempcoder, SpikingProblem* problem)
      : tempcoder_(tempcoder), problem_(problem) {}

  Runner(Tempcoder* tempcoder, SpikingProblem* problem, int num_threads)
      : tempcoder_(tempcoder),
        problem_(problem),
        // If num_threads is 1, set it to 0 to keep all the work in the main
        // thread for less overhead.
        thread_pool_train_(num_threads == 1 ? 0 : num_threads) {
    IHM_LOG(LogSeverity::INFO,
            absl::StrFormat("Using ThreadPool with %d threads.",
                            thread_pool_train_.NumThreads()));
  }

  // Returns the number of correctly classified training samples in the batch.
  int ProcessBatch(size_t train_ind, VectorXd* train_errors);

  // Runs all `examples` through the `tempcoder`, without updating the
  // gradients. Updates `examples` with the predictions. Returns the number of
  // correct predictions and the total loss.
  std::tuple<int, double> FeedforwardNoUpdatesParallel(
      std::vector<Example>* examples,
      const RunnerOptions& runner_options = RunnerOptions());

  // Returns a tuple containing the accuracy in the
  // (combined training and validation sets, test set).
  std::tuple<float, float, float, float> TestModel(
      const RunnerOptions& runner_options = RunnerOptions());

  int batch_size_ = 1;
  TempcoderOptions tempcoder_options_;

 private:
  Tempcoder* tempcoder_;
  SpikingProblem* problem_;
  absl::Mutex mutex_;
  ihmehimmeli::ThreadPool thread_pool_train_;
  ihmehimmeli::ThreadPool thread_pool_nontrain_;
};

}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_TEMPCODING_RUNNER_H_
