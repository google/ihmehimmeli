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

#include "event/runner.h"

#include "common/data_parallel.h"
#include "event/adam.h"
#include "event/event.h"
#include "event/loss.h"

namespace ihmehimmeli {

TrainingOutcome TrainNetwork(const LearningParams& learning_params,
                             const TrainingParams& training_params,
                             const TerminationParams& termination_params,
                             const Problem& problem,
                             const std::vector<uint32_t>& training,
                             const std::vector<uint32_t>& validation,
                             Network* network, TrainCallback* callback) {
  // Thread pool and per-thread storage.
  // Limit number of threads to half batch size (rounding up).
  ThreadPool pool(std::min<size_t>((learning_params.batch_size + 1) / 2,
                                   std::thread::hardware_concurrency()));
  ThreadPool validation_pool;
  size_t max_num_threads =
      std::max(pool.NumThreads(), validation_pool.NumThreads());
  NetworkGradient aggregate_gradients;
  std::vector<NetworkGradient> gradients(pool.NumThreads());
  std::vector<std::vector<std::vector<double>>> inputs(max_num_threads);
  std::vector<std::vector<std::vector<double>>> outputs(max_num_threads);
  std::vector<std::vector<std::vector<double>>> expected(max_num_threads);
  std::vector<double> num_correct(max_num_threads);
  std::vector<double> total_loss(max_num_threads);

  TrainingOutcome training_outcome;
  training_outcome.last_validation_accuracy = 0.0;
  training_outcome.last_train_accuracy = 0.0;
  Network best_network = *network;

  AdamUpdater weight_updater;
  AdamUpdater pulse_t0_updater;
  AdamUpdater pulse_interval_updater;

  // Training loop.
  auto run_and_train = [&](size_t idx, size_t thread) {
    problem.Example(training[idx], &inputs[thread], &expected[thread]);
    double loss =
        RunAndBackpropagate(*network, termination_params, inputs[thread],
                            training_params, expected[thread], &outputs[thread],
                            &gradients[thread], &aggregate_gradients);
    total_loss[thread] += loss;
    num_correct[thread] +=
        training_params.loss->FracCorrect(outputs[thread], expected[thread]);
  };

  std::atomic<uint32_t> num_valid_done{0};
  auto run_validation = [&](size_t idx, size_t thread) {
    problem.Example(validation[idx], &inputs[thread], &expected[thread]);
    double loss = RunAndComputeLoss(*network, termination_params,
                                    inputs[thread], training_params.loss.get(),
                                    expected[thread], &outputs[thread]);
    total_loss[thread] += loss;
    num_correct[thread] +=
        training_params.loss->FracCorrect(outputs[thread], expected[thread]);
    num_valid_done++;
    if (callback) callback->ValidProgress(num_valid_done, validation.size());
  };

  for (size_t epoch = 0; epoch < learning_params.num_epochs; epoch++) {
    auto epoch_start = std::chrono::high_resolution_clock::now();
    // Clear out per-epoch data.
    ZeroOutVector(&num_correct);
    ZeroOutVector(&total_loss);
    for (size_t batch = 0; batch < training.size();
         batch += learning_params.batch_size) {
      aggregate_gradients.Clear();
      // Run epochs.
      size_t batch_end =
          std::min(batch + learning_params.batch_size, training.size());
      pool.Run(batch, batch_end, run_and_train);

      // ADAM step.
      weight_updater.Update(aggregate_gradients.d_weights, batch_end - batch,
                            learning_params.learning_rate, &network->weights);
      pulse_t0_updater.Update(
          aggregate_gradients.d_pulses_t0, batch_end - batch,
          learning_params.learning_rate_pulses, &network->pulses_t0);
      pulse_interval_updater.Update(
          aggregate_gradients.d_pulses_interval, batch_end - batch,
          learning_params.learning_rate_pulses, &network->pulses_interval);
      // Ensure no negative pulses.
      for (size_t i = 0; i < network->pulses_t0.size(); i++) {
        network->pulses_t0[i] = std::max(0.0, network->pulses_t0[i]);
      }
      // Ensure repeated pulses are not too frequent.
      for (size_t i = 0; i < network->pulses_t0.size(); i++) {
        network->pulses_interval[i] =
            std::max(network->refractory_period, network->pulses_interval[i]);
      }
      if (callback) callback->TrainProgress(batch, training.size());
    }

    // Compute stats.
    double train_loss =
        std::accumulate(total_loss.begin(), total_loss.end(), 0.0);
    double train_correct =
        std::accumulate(num_correct.begin(), num_correct.end(), 0.0);

    // Validation run.
    ZeroOutVector(&num_correct);
    ZeroOutVector(&total_loss);
    num_valid_done = 0;
    validation_pool.Run(0, validation.size(), run_validation);
    double validation_loss =
        std::accumulate(total_loss.begin(), total_loss.end(), 0.0);
    double validation_correct =
        std::accumulate(num_correct.begin(), num_correct.end(), 0.0);

    double training_accuracy = 100.0 * train_correct / training.size();
    double validation_accuracy = 100.0 * validation_correct / validation.size();
    if (validation_accuracy > training_outcome.best_validation_accuracy) {
      training_outcome.best_validation_accuracy = validation_accuracy;
      best_network = *network;
    }

    auto epoch_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         epoch_end - epoch_start)
                         .count() *
                     0.001f;
    if (callback)
      callback->EpochDone(epoch, training_accuracy,
                          train_loss / training.size(), validation_accuracy,
                          validation_loss / validation.size(), elapsed);

    training_outcome.last_train_accuracy = training_accuracy;
    training_outcome.last_validation_accuracy = validation_accuracy;
    training_outcome.elapsed_time += elapsed;
  }
  training_outcome.best_network = best_network;
  training_outcome.last_network = *network;
  return training_outcome;
}

void TestNetwork(const TerminationParams& termination_params,
                 const Problem& problem, const std::vector<uint32_t>& test,
                 const Network& network, LossFunction* loss_function,
                 TestCallback* callback) {
  auto start = std::chrono::high_resolution_clock::now();
  ThreadPool pool;
  std::vector<std::vector<std::vector<double>>> inputs(pool.NumThreads());
  std::vector<std::vector<std::vector<double>>> outputs(pool.NumThreads());
  std::vector<std::vector<std::vector<double>>> expected(pool.NumThreads());
  std::vector<double> num_correct(pool.NumThreads());
  std::vector<double> total_loss(pool.NumThreads());
  std::atomic<uint32_t> num_done{0};
  auto run_test = [&](size_t idx, size_t thread) {
    problem.Example(test[idx], &inputs[thread], &expected[thread]);
    double loss =
        RunAndComputeLoss(network, termination_params, inputs[thread],
                          loss_function, expected[thread], &outputs[thread]);
    total_loss[thread] += loss;
    num_correct[thread] +=
        loss_function->FracCorrect(outputs[thread], expected[thread]);
    num_done++;
    if (callback) callback->TestProgress(num_done, test.size());
  };
  pool.Run(0, test.size(), run_test);
  double test_loss = std::accumulate(total_loss.begin(), total_loss.end(), 0.0);
  double test_correct =
      std::accumulate(num_correct.begin(), num_correct.end(), 0.0);
  auto end = std::chrono::high_resolution_clock::now();
  double elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count() *
      0.001f;
  if (callback)
    callback->Done(100.0f * test_correct / test.size(), test_loss / test.size(),
                   elapsed);
}

}  // namespace ihmehimmeli
