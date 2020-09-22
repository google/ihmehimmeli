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

#ifndef IHMEHIMMELI_EVENT_RUNNER_H
#define IHMEHIMMELI_EVENT_RUNNER_H

#include <functional>

#include "event/event.h"
#include "event/problem.h"

namespace ihmehimmeli {

struct LearningParams {
  double learning_rate = 0.0001;
  double learning_rate_pulses = 0.00005;
  size_t num_epochs = 10;
  size_t batch_size = 40;
};

struct TrainingOutcome {
  double last_train_accuracy;
  double last_validation_accuracy;
  double best_validation_accuracy;
  double elapsed_time;
  Network best_network = Network(nullptr);
  Network last_network = Network(nullptr);
};

// An object implementing this interface can be used to execute code during
// training and at the end of each epoch, typically to report progress.
class TrainCallback {
 public:
  virtual void TrainProgress(size_t cur, size_t total) = 0;
  virtual void ValidProgress(size_t cur, size_t total) = 0;
  virtual void EpochDone(size_t epoch, float train_accuracy, float train_loss,
                         float valid_accuracy, float valid_loss,
                         float seconds) = 0;
  virtual ~TrainCallback() {}
};

// TrainCallback that prints training information on standard error.
class StderrTrainCallback : public TrainCallback {
 public:
  virtual void TrainProgress(size_t cur, size_t total) override {
    fprintf(stderr, "T %9lu/%9lu\r", cur, total);
    fflush(stderr);
  };
  virtual void ValidProgress(size_t cur, size_t total) override {
    fprintf(stderr, "V %9lu/%9lu\r", cur, total);
    fflush(stderr);
  };
  void EpochDone(size_t epoch, float train_accuracy, float train_loss,
                 float valid_accuracy, float valid_loss,
                 float seconds) override {
    fprintf(stderr,
            "E %10zu: train %6.2f%% / %10.5f, valid \033[;1m%6.2f%% / "
            "%10.5f\033[;m %8.3fs\n",
            epoch, train_accuracy, train_loss, valid_accuracy, valid_loss,
            seconds);
  }
};

// Similar to TrainCallback, but for test progress/results.
class TestCallback {
 public:
  virtual void TestProgress(size_t cur, size_t total) = 0;
  virtual void Done(float accuracy, float loss, float seconds) = 0;
  virtual ~TestCallback() {}
};

// TestCallback that prints training information on standard error.
class StderrTestCallback : public TestCallback {
 public:
  virtual void TestProgress(size_t cur, size_t total) override {
    fprintf(stderr, "T %9lu/%9lu\r", cur, total);
    fflush(stderr);
  };
  void Done(float accuracy, float loss, float seconds) override {
    fprintf(stderr, "Test accuracy: %.2f%%\nTest loss: %.5f\nElapsed: %8.3fs\n",
            accuracy, loss, seconds);
  }
};

// Train a network with the given parameters, applying ADAM updates after each
// batch. Each batch will be run in parallel, using a total of up to
// batch_size/2 threads.
TrainingOutcome TrainNetwork(const LearningParams& learning_params,
                             const TrainingParams& training_params,
                             const TerminationParams& termination_params,
                             const Problem& problem,
                             const std::vector<uint32_t>& training,
                             const std::vector<uint32_t>& validation,
                             Network* network,
                             TrainCallback* callback = nullptr);

// Run the network on the given test set, in parallel on as many threads as are
// available.
void TestNetwork(const TerminationParams& termination_params,
                 const Problem& problem, const std::vector<uint32_t>& test,
                 const Network& network, LossFunction* loss_function,
                 TestCallback* callback = nullptr);

}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_EVENT_RUNNER_H
