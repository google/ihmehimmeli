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

#include "tempcoding/runner.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>

#include "common/util.h"

namespace ihmehimmeli {

std::tuple<int, double> Runner::FeedforwardNoUpdatesParallel(
    std::vector<Example>* examples, bool print_spike_stats) {
  int num_threads = std::max<size_t>(thread_pool_nontrain_.NumThreads(), 1);
  std::vector<int> correct_per_thread(num_threads, 0);
  std::vector<double> error_per_thread(num_threads, 0);

  // Currently using the mean, but we could use the median in the future.
  std::vector<std::vector<double>> mean_spike_time_per_layer(
      tempcoder_->layer_sizes().size(), std::vector<double>(examples->size()));
  std::vector<double> first_output_spike_times(examples->size());

  thread_pool_nontrain_.Run(
      0, examples->size(),
      [this, &examples, &correct_per_thread, &error_per_thread,
       &mean_spike_time_per_layer, &first_output_spike_times,
       print_spike_stats](const int i, const int num_thread) {
        auto& example = (*examples)[i];
        TrainOptions train_opts;
        train_opts.compute_spike_stats = print_spike_stats;
        example.prediction =
            tempcoder_->FeedforwardAndOptionallyAccumulateUpdates(
                example.inputs, example.targets, nullptr, train_opts);
        for (int layer = 0;
             layer < example.prediction.spike_stats_per_layer.size(); ++layer) {
          mean_spike_time_per_layer[layer][i] =
              example.prediction.spike_stats_per_layer[layer].mean();
        }
        first_output_spike_times[i] =
            example.prediction.first_output_spike_time;
        correct_per_thread[num_thread] += example.prediction.is_correct;
        error_per_thread[num_thread] += example.prediction.error;
      });

  int total_correct = 0;
  for (int num_correct : correct_per_thread) {
    total_correct += num_correct;
  }
  double total_error =
      std::accumulate(error_per_thread.begin(), error_per_thread.end(), 0.0);

  if (print_spike_stats) {
    for (int layer = 0; layer < mean_spike_time_per_layer.size(); ++layer) {
      BasicStats<double> layer_spike_stats(mean_spike_time_per_layer[layer]);
      PrintLayerStats(layer, "spikes", layer_spike_stats);
    }
    first_output_spike_times.erase(
        std::remove(first_output_spike_times.begin(),
                    first_output_spike_times.end(), Tempcoder::kNoSpike),
        first_output_spike_times.end());
    BasicStats<double> first_output_spike_stats(first_output_spike_times);
    std::cout << "First output spike time (mean median sd): "
              << first_output_spike_stats.mean() << " "
              << first_output_spike_stats.median() << " "
              << first_output_spike_stats.stddev() << std::endl;
  }

  return std::tie(total_correct, total_error);
}

int Runner::ProcessBatch(size_t train_ind, VectorXd* train_errors) {
  // Feedforward pass for batch.
  const int train_ind_high_bound =
      std::min(train_ind + batch_size_, problem_->train_examples().size());
  const int num_training_threads =
      std::max<size_t>(thread_pool_train_.NumThreads(), 1);
  std::vector<int> train_correct_per_thread(num_training_threads, 0);

  thread_pool_train_.Run(
      train_ind, train_ind_high_bound,
      [this, &train_errors, &train_correct_per_thread](const int i,
                                                       const int num_thread) {
        VectorXd inputs = problem_->train_examples()[i].inputs;

        const Prediction prediction =
            tempcoder_->FeedforwardAndOptionallyAccumulateUpdates(
                inputs, problem_->train_examples()[i].targets, &mutex_,
                train_options_);
        problem_->train_examples()[i].prediction = prediction;

        (*train_errors)[i] = prediction.error;
        train_correct_per_thread[num_thread] += prediction.is_correct;
      });

  double total_train_error = 0;
  for (double error : (*train_errors)) {
    total_train_error += error;
  }

  int batch_correct = 0;
  for (int num_correct : train_correct_per_thread) {
    batch_correct += num_correct;
  }

  VectorXXXd weight_updates;
  VectorXXd pulse_updates;
  std::tie(weight_updates, pulse_updates) = tempcoder_->ComputeUpdates();

  if (gradient_landscape_options_.ShouldPrint()) {
    PrintGradientLandscape(tempcoder_, problem_, &thread_pool_nontrain_,
                           total_train_error, train_ind, train_ind_high_bound,
                           weight_updates, pulse_updates,
                           gradient_landscape_options_);
    ++gradient_landscape_options_.batch_num;
  }

  if (gradient_landscape_options_.check_gradient) {
    IHM_CHECK((train_ind_high_bound - train_ind) == 1,
              "Gradient check needs batch size 1.");
    IHM_CHECK(!(tempcoder_->use_adam()),
              "Gradient check does not work with Adam.");
    if (batch_correct) {
      std::cout << "No updates." << std::endl;
    } else {
      const VectorXd estimated_grads = tempcoder_->CheckGradient(
          problem_->train_examples()[train_ind].inputs,
          problem_->train_examples()[train_ind].targets);
      std::cout << "Backprop gradients:\n"
                << PPrintVectorXXXd(tempcoder_->d_weights()) << " "
                << PPrintVectorXXd(tempcoder_->d_pulses()) << "\n"
                << "Estimated gradients:\n"
                << VecToString(estimated_grads) << std::endl;
    }
    std::cin.get();
  }

  // Backpropagate for this batch.
  tempcoder_->ApplyTrainingUpdates(weight_updates, pulse_updates);
  return batch_correct;
}

std::tuple<float, float> Runner::TestModel(const TestOptions& test_options) {
  if (test_options.print_network) tempcoder_->PrintNetwork();
  if (test_options.print_network_stats) tempcoder_->PrintNetworkStats();

  float train_valid_accuracy = std::numeric_limits<float>::infinity();
  if (test_options.compute_train_accuracy) {
    int train_correct;
    double train_error;
    std::tie(train_correct, train_error) = FeedforwardNoUpdatesParallel(
        &problem_->train_examples(), test_options.print_train_spike_stats);
    int validation_correct;
    double validation_error;
    std::tie(validation_correct, validation_error) =
        FeedforwardNoUpdatesParallel(&problem_->validation_examples(),
                                     test_options.print_train_spike_stats);
    int train_valid_correct = train_correct + validation_correct;
    train_valid_accuracy = 100.0 * train_valid_correct /
                           (problem_->train_examples().size() +
                            problem_->validation_examples().size());
    const float train_valid_error_scaled =
        (train_error + validation_error) /
        (problem_->validation_examples().size() +
         problem_->train_examples().size());
    std::cout << "Train correct / error: " << train_valid_accuracy << "% "
              << train_valid_error_scaled << std::endl;
  }
  int test_correct;
  double test_error;
  std::tie(test_correct, test_error) = FeedforwardNoUpdatesParallel(
      &problem_->test_examples(), test_options.print_test_spike_stats);
  const float test_accuracy =
      100.0 * test_correct / problem_->test_examples().size();
  std::cout << "Test correct / error: " << test_accuracy << "% "
            << test_error / problem_->test_examples().size() << std::endl;

  return std::tie(train_valid_accuracy, test_accuracy);
}

}  // namespace ihmehimmeli
