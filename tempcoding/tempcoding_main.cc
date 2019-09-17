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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>  //NOLINT
#include <tuple>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tempcoding/data_parallel.h"
#include "tempcoding/file_passthrough.h"
#include "tempcoding/gradient_landscape_helper.h"
#include "tempcoding/runner.h"
#include "tempcoding/spiking_problem.h"
#include "tempcoding/tempcoder.h"
#include "tempcoding/util.h"

// Flags for problem/network definition.
ABSL_FLAG(
    std::string, problem, "xor",
    "problem (one of: and, or, xor, circle, iris, mnist, mnist2, cifar10)");
ABSL_FLAG(int64_t, conv_edge, 1,
          "length of convolution window edge for 2D problems (mnist)");
ABSL_FLAG(int64_t, n_inputs, 2, "number of inputs for and, or, xor");
ABSL_FLAG(int64_t, n_digits, 10, "number of digits to use for MNIST");
ABSL_FLAG(double, input_range_min, 0, "low end of range to scale inputs to");
ABSL_FLAG(double, input_range_max, 1, "high end of range to scale inputs to");
ABSL_FLAG(std::string, n_hidden, "4", "hidden layer size (vector)");
ABSL_FLAG(int64_t, n_pulses, 2, "number of sync pulses per layer (int)");
ABSL_FLAG(double, learning_rate, 0.0001, "learning rate");
ABSL_FLAG(double, learning_rate_pulses, 0.001, "learning rate of pulses");
ABSL_FLAG(double, learning_rate_multiplier, 1.0,
          "value to multiply learning rate with at every epoch");
ABSL_FLAG(double, fire_threshold, 0.4, "neuron firing threshold");
ABSL_FLAG(double, decay_rate, 1.0, "neuron membrane potential decay rate");
ABSL_FLAG(double, penalty_no_spike, 1.0, "penalty for no spike");
ABSL_FLAG(double, penalty_output_spike_time, 0.0,
          "output spike time penalty (to encourage faster spike times)");
ABSL_FLAG(double, initial_weights_lower_bound, 0.5, "initial weights min");
ABSL_FLAG(double, initial_weights_upper_bound, 0.9, "initial weights max");
ABSL_FLAG(bool, use_glorot_initialization, true,
          "use glorot weight initialization?");
ABSL_FLAG(double, pulse_weight_mean_multiplier, 0.0,
          "If glorot weight initialization is used, the pulse weights are "
          "drawn from a Gaussian distribution with stddev sqrt(2.0/(layer_size "
          "+ previous_layer_size)) and mean equal to stddev times this value");
ABSL_FLAG(double, nonpulse_weight_mean_multiplier, 0.0,
          "If glorot weight initialization is used, the nonpulse weights are "
          "drawn from a Gaussian distribution with stddev sqrt(2.0/(layer_size "
          "+ previous_layer_size)) and mean equal to stddev times this value");
ABSL_FLAG(bool, use_dual_exponential, false,
          "use dual exponential instead of alpha activation function");

// Flags for training/run parameters.
ABSL_FLAG(bool, use_adam, true, "use adam?");
ABSL_FLAG(bool, update_all_datapoints, false, "update on all data points?");
ABSL_FLAG(int64_t, batch_size, 1, "batch size");
ABSL_FLAG(double, batch_size_multiplier, 1,
          "value to multiply batch size with at every epoch (the floor of the "
          "result is taken)");
ABSL_FLAG(double, clip_gradient, 0.0,
          "clip the gradient at +- this value (0 disables)");
ABSL_FLAG(double, clip_derivative, 0.0,
          "clip the derivative of the activation function at +- this value (0 "
          "disables)");
ABSL_FLAG(double, input_jittering_sigma, 0.0,
          "add jitter to input spikes (both positive and negative) with mean "
          "input_jittering_mean and standard deviation input_jittering_sigma");
ABSL_FLAG(double, input_jittering_mean, 0.0,
          "add jitter to input spikes (both positive and negative) with mean "
          "input_jittering_mean and standard deviation input_jittering_sigma");
ABSL_FLAG(double, input_jittering_new_spike_probability, 0.0,
          "probability that a input neuron that would not normally fire "
          "instead fires");
ABSL_FLAG(
    double, noninput_jittering_sigma, 0.0,
    "add jitter to noninput spikes (both positive and negative) with mean "
    "noninput_jittering_mean and standard deviation noninput_jittering_sigma");
ABSL_FLAG(
    double, noninput_jittering_mean, 0.0,
    "add jitter to noninput spikes (both positive and negative) with mean "
    "noninput_jittering_mean and standard deviation noninput_jittering_sigma");
ABSL_FLAG(double, noninput_jittering_new_spike_probability, 0.0,
          "probability that a noninput neuron that would not normally fire "
          "instead fires");
ABSL_FLAG(int64_t, batch_size_clip, 0, "maximum acceptable batch size");
ABSL_FLAG(int64_t, n_runs, 10, "number of runs");
ABSL_FLAG(int64_t, n_epochs, 100, "number of training epochs per run");
ABSL_FLAG(int64_t, n_train, 1000, "number of training examples");
ABSL_FLAG(int64_t, n_validation, 100, "number of validation examples");
ABSL_FLAG(int64_t, n_test, 100, "number of test examples");
ABSL_FLAG(bool, shuffle_between_epochs, true,
          "shuffle training data between epochs?");
ABSL_FLAG(bool, shuffle_loaded_examples, true,
          "shuffle loaded training/validation data?");

// Flags for gradient landscape printing.
ABSL_FLAG(bool, check_gradient, false, "check gradient?");
ABSL_FLAG(bool, print_gradient_landscape, false,
          "print objective values along the gradient line? enables [gradient]");
ABSL_FLAG(double, line_points_multiplier, 1.0,
          "[gradient] value for multiplying default line search point values");
ABSL_FLAG(int64_t, show_objectives_every_n_batches, 100,
          "[gradient] show objectives every [n] batches.");
ABSL_FLAG(int64_t, custom_num_objective_samples, 0,
          "[gradient] when > 0, the number of points to sample objective "
          "function at");
ABSL_FLAG(bool, print_spikes_along_gradient, false,
          "[gradient] print the output spike times of the first batch example "
          "along the gradient line?");

// Flags for printing to console output.
ABSL_FLAG(bool, print_best_model, false, "print best model?");
ABSL_FLAG(bool, print_test_set, false, "print test set?");
ABSL_FLAG(bool, print_train_set, false, "print training set?");
ABSL_FLAG(bool, wait_for_input_between_runs, true,
          "wait for input between runs?");

// Flags for file I/O.
ABSL_FLAG(bool, write_best_model, false, "write best model to file?");
ABSL_FLAG(bool, write_all_models, false, "write all models to file?");
ABSL_FLAG(std::string, output_dir, "/tmp/", "folder to save files in");
ABSL_FLAG(std::string, model_to_load, "",
          "MNIST/CIFAR10 model filename to start from");

// Flags for data loading.
ABSL_FLAG(std::string, mnist_data_path, "",
          "path where MNIST data should be loaded from");
ABSL_FLAG(std::string, cifar10_data_path, "",
          "path where CIFAR10 data should be loaded from");

// Flags for test mode.
ABSL_FLAG(std::string, model_to_test, "",
          "MNIST/CIFAR10 model filename to test (enables [testing] mode)");
ABSL_FLAG(bool, print_test_stats, false, "[testing] print stats for test set?");
ABSL_FLAG(bool, print_train_stats, false,
          "[testing] print stats for train set?");
ABSL_FLAG(bool, print_network, false, "[testing] print network?");
ABSL_FLAG(bool, print_network_stats, false, "[testing] print network stats?");
ABSL_FLAG(bool, compute_train_accuracy, true,
          "[testing] print accuracy on train set?");

namespace ihmehimmeli {

void PrintExamples(const std::vector<Example>& examples) {
  std::cout << std::endl << "data = [";
  for (const auto& example : examples) {
    std::cout << "[" << VecToString(example.inputs) << ", "
              << VecToString(example.targets) << ", " << std::setprecision(10)
              << VecToString(example.prediction.outputs) << "], ";
  }
  std::cout << "]" << std::endl;
}

void DumpArgs(absl::string_view output_dir, absl::string_view initial_timestamp,
              absl::string_view problem_flag, int argc, char** argv) {
  const std::string args_file_path = file::JoinPath(
      output_dir,
      absl::StrFormat("%s_%s_args.log", initial_timestamp, problem_flag));
  file::IhmFile args_file;
  IHM_CHECK(file::Open(args_file_path, "w", &args_file));
  for (int i = 0; i < argc; i++)
    IHM_CHECK(args_file.Write(absl::StrCat(argv[i], "\n"), nullptr));
  IHM_CHECK(args_file.Close());
}

SpikingProblem CreateSpikingProblem() {
  const std::string problem_flag(absl::GetFlag(FLAGS_problem));
  const ProblemType problem_type = ParseProblemType(problem_flag);
  const int conv_edge = absl::GetFlag(FLAGS_conv_edge);
  IHM_CHECK(conv_edge > 0);
  const int n_inputs = absl::GetFlag(FLAGS_n_inputs);
  IHM_CHECK(n_inputs > 0);
  int n_outputs = 2;
  if (problem_type == ProblemType::MNIST ||
      problem_type == ProblemType::MNIST_BIPOLAR) {
    n_outputs = absl::GetFlag(FLAGS_n_digits);
  }
  if (problem_type == ProblemType::CIFAR10) {
    n_outputs = 10;
  }
  double input_range_min = absl::GetFlag(FLAGS_input_range_min);
  double input_range_max = absl::GetFlag(FLAGS_input_range_max);
  IHM_CHECK(input_range_max > input_range_min);
  const std::pair<double, double> input_range =
      std::make_pair(input_range_min, input_range_max);
  SpikingProblem problem(problem_type, n_inputs, n_outputs, input_range,
                         conv_edge);
  std::string mnist_data_path = absl::GetFlag(FLAGS_mnist_data_path);
  if (!mnist_data_path.empty()) {
    problem.mnist_data_path() = mnist_data_path;
  }
  std::string cifar10_data_path = absl::GetFlag(FLAGS_cifar10_data_path);
  if (!cifar10_data_path.empty()) {
    problem.cifar10_data_path() = cifar10_data_path;
  }
  return problem;
}

void Main(int argc, char** argv) {
  SpikingProblem problem = CreateSpikingProblem();

  const int n_train = absl::GetFlag(FLAGS_n_train);
  const int n_validation = absl::GetFlag(FLAGS_n_validation);
  const int n_test = absl::GetFlag(FLAGS_n_test);
  const bool shuffle_loaded_examples =
      absl::GetFlag(FLAGS_shuffle_loaded_examples);

  // Transfer function configuration.
  const double fire_threshold = absl::GetFlag(FLAGS_fire_threshold);
  const double decay_rate = absl::GetFlag(FLAGS_decay_rate);
  const bool use_dual_exponential = absl::GetFlag(FLAGS_use_dual_exponential);

  // If test mode is enabled, load and test a model.
  std::string model_to_test_filename = absl::GetFlag(FLAGS_model_to_test);
  if (!model_to_test_filename.empty()) {
    Tempcoder tempcoder =
        Tempcoder::LoadTempcoderFromFile(model_to_test_filename);
    tempcoder.decay_params().set_decay_rate(decay_rate);
    tempcoder.penalty_output_spike_time() =
        absl::GetFlag(FLAGS_penalty_output_spike_time);
    problem.GenerateExamples(n_train, n_validation, n_test,
                             shuffle_loaded_examples);
    Runner test_runner(&tempcoder, &problem);
    const TestOptions test_options = [&]() {
      TestOptions options;
      options.compute_train_accuracy =
          absl::GetFlag(FLAGS_compute_train_accuracy);
      options.print_train_spike_stats = absl::GetFlag(FLAGS_print_train_stats);
      options.print_test_spike_stats = absl::GetFlag(FLAGS_print_test_stats);
      options.print_network = absl::GetFlag(FLAGS_print_network);
      options.print_network_stats = absl::GetFlag(FLAGS_print_network_stats);
      return options;
    }();
    test_runner.TestModel(test_options);
    return;
  }

  // Load a model from file, if specified.
  std::unique_ptr<Tempcoder> tempcoder_loaded;
  std::string model_to_load_filename = absl::GetFlag(FLAGS_model_to_load);
  if (!model_to_load_filename.empty()) {
    tempcoder_loaded = absl::make_unique<Tempcoder>(
        Tempcoder::LoadTempcoderFromFile(model_to_load_filename));
  }

  // Network configuration.
  const double initial_weights_lower_bound =
      absl::GetFlag(FLAGS_initial_weights_lower_bound);
  const double initial_weights_upper_bound =
      absl::GetFlag(FLAGS_initial_weights_upper_bound);
  const bool use_glorot_initialization =
      absl::GetFlag(FLAGS_use_glorot_initialization);
  const double pulse_weight_mean_multiplier =
      absl::GetFlag(FLAGS_pulse_weight_mean_multiplier);
  const double nonpulse_weight_mean_multiplier =
      absl::GetFlag(FLAGS_nonpulse_weight_mean_multiplier);
  const WeightInitializationOptions weight_options = [&]() {
    WeightInitializationOptions options;
    options.use_glorot_initialization = use_glorot_initialization;
    options.weights_lower_bound = initial_weights_lower_bound;
    options.weights_upper_bound = initial_weights_upper_bound;
    options.nonpulse_weight_mean_multiplier = nonpulse_weight_mean_multiplier;
    options.pulse_weight_mean_multiplier = pulse_weight_mean_multiplier;
    return options;
  }();

  std::vector<int> layer_sizes;
  std::string layer_str = absl::GetFlag(FLAGS_n_hidden);
  for (absl::string_view l : absl::StrSplit(layer_str, ',')) {
    int layer_size;
    if (!(absl::SimpleAtoi(l, &layer_size))) {
      IHM_LOG(LogSeverity::FATAL, "Invalid n_hidden argument.");
    }
    layer_sizes.push_back(layer_size);
  }

  layer_sizes.insert(layer_sizes.begin(), problem.n_inputs());
  layer_sizes.insert(layer_sizes.end(), problem.n_outputs());
  IHM_LOG(LogSeverity::INFO, absl::StrFormat("Network architecture: %s",
                                             VecToString(layer_sizes)));

  // Initialise pulses spike times evenly in the input range.
  // Start with the same pulse values in each layer.
  const VectorXXd pulses(
      layer_sizes.size(),
      GeneratePulses(absl::GetFlag(FLAGS_n_pulses), problem.input_range()));
  IHM_LOG(LogSeverity::INFO,
          absl::StrFormat("Sync pulses: %s", VecToString(pulses.front())));

  // Parameters of the run.
  const int n_runs = absl::GetFlag(FLAGS_n_runs);
  const int n_epochs = absl::GetFlag(FLAGS_n_epochs);
  float batch_size = absl::GetFlag(FLAGS_batch_size);
  IHM_CHECK(batch_size > 0, "batch_size must be strictly positive.");
  const float batch_size_multiplier =
      absl::GetFlag(FLAGS_batch_size_multiplier);
  IHM_CHECK(batch_size_multiplier > 0,
            "batch_size_multiplier must be strictly positive.");
  // Maximum batch size expected at the end of training.
  const unsigned int batch_size_max = std::min(
      1e9f, std::floor(batch_size *
                       ihmehimmeli::IPow(batch_size_multiplier, n_epochs - 1)));
  const unsigned int batch_size_clip =
      absl::GetFlag(FLAGS_batch_size_clip) > 0
          ? absl::GetFlag(FLAGS_batch_size_clip)
          : n_train;
  const bool print_best_model = absl::GetFlag(FLAGS_print_best_model);
  const bool print_train_set = absl::GetFlag(FLAGS_print_train_set);
  const bool print_test_set = absl::GetFlag(FLAGS_print_test_set);
  const bool write_best_model = absl::GetFlag(FLAGS_write_best_model);
  const bool write_all_models = absl::GetFlag(FLAGS_write_all_models);
  const std::string output_dir = absl::GetFlag(FLAGS_output_dir);

  // Network training parameters.
  const double learning_rate = absl::GetFlag(FLAGS_learning_rate);
  const double learning_rate_multiplier =
      absl::GetFlag(FLAGS_learning_rate_multiplier);
  const double learning_rate_pulses = absl::GetFlag(FLAGS_learning_rate_pulses);
  const double penalty_no_spike = absl::GetFlag(FLAGS_penalty_no_spike);
  const double penalty_output_spike_time =
      absl::GetFlag(FLAGS_penalty_output_spike_time);
  const double clip_gradient = absl::GetFlag(FLAGS_clip_gradient);
  const double clip_derivative = absl::GetFlag(FLAGS_clip_derivative);
  const double input_jittering_sigma =
      absl::GetFlag(FLAGS_input_jittering_sigma);
  const double input_jittering_mean = absl::GetFlag(FLAGS_input_jittering_mean);
  const double input_jittering_new_spike_probability =
      absl::GetFlag(FLAGS_input_jittering_new_spike_probability);
  const double noninput_jittering_sigma =
      absl::GetFlag(FLAGS_noninput_jittering_sigma);
  const double noninput_jittering_mean =
      absl::GetFlag(FLAGS_noninput_jittering_mean);
  const double noninput_jittering_new_spike_probability =
      absl::GetFlag(FLAGS_noninput_jittering_new_spike_probability);
  const double use_adam = absl::GetFlag(FLAGS_use_adam);
  const bool shuffle_between_epochs =
      absl::GetFlag(FLAGS_shuffle_between_epochs);

  IHM_CHECK(n_runs > 0);
  IHM_CHECK(n_epochs > 0);
  IHM_CHECK(n_train > 0);
  IHM_CHECK(n_validation >= 0);
  IHM_CHECK(n_test >= 0);
  IHM_CHECK(clip_gradient >= 0.0);
  IHM_CHECK(clip_derivative >= 0.0);
  IHM_CHECK(input_jittering_sigma >= 0.0);
  IHM_CHECK(input_jittering_new_spike_probability >= 0.0);
  IHM_CHECK(input_jittering_new_spike_probability <= 1.0);
  IHM_CHECK(noninput_jittering_sigma >= 0.0);
  IHM_CHECK(noninput_jittering_new_spike_probability >= 0.0);
  IHM_CHECK(noninput_jittering_new_spike_probability <= 1.0);

  if (n_validation == 0) {
    IHM_LOG(LogSeverity::WARNING,
            "No validation set. Saving the model that performs best on "
            "training set.");
  } else {
    IHM_LOG(LogSeverity::INFO,
            "Saving the model that performs best on validation set.");
  }

  // Gradient landscape analysis parameters.
  const GradientLandscapeOptions gradient_landscape_options = [&]() {
    GradientLandscapeOptions options;
    options.check_gradient = absl::GetFlag(FLAGS_check_gradient);
    options.print_gradient_landscape =
        absl::GetFlag(FLAGS_print_gradient_landscape);
    options.batch_num = 0;
    options.line_points_multiplier =
        absl::GetFlag(FLAGS_line_points_multiplier);
    options.custom_num_objective_samples =
        absl::GetFlag(FLAGS_custom_num_objective_samples);
    options.show_objectives_every_n_batches =
        absl::GetFlag(FLAGS_show_objectives_every_n_batches);
    options.print_spikes_along_gradient =
        absl::GetFlag(FLAGS_print_spikes_along_gradient);
    return options;
  }();

  // Set up runner options.
  const TrainOptions train_options = [&]() {
    TrainOptions options;
    options.accumulate_weight_updates = true;
    options.update_all_datapoints = absl::GetFlag(FLAGS_update_all_datapoints);
    return options;
  }();
  // This is to prevent a performance loss from too many threads fighting
  // for insufficiently big workload.
  const int num_threads = std::min(std::min(batch_size_clip, batch_size_max),
                                   std::thread::hardware_concurrency());

  VectorXd test_errors;
  for (int run = 0; run < n_runs; ++run) {
    // New dataset at each run.
    problem.GenerateExamples(n_train, n_validation, n_test,
                             shuffle_loaded_examples);
    Tempcoder tempcoder = tempcoder_loaded
                              ? *tempcoder_loaded
                              : Tempcoder(layer_sizes, pulses, fire_threshold,
                                          weight_options, use_dual_exponential);
    tempcoder.learning_rate() = learning_rate;
    tempcoder.learning_rate_pulses() = learning_rate_pulses;
    tempcoder.penalty_no_spike() = penalty_no_spike;
    tempcoder.penalty_output_spike_time() = penalty_output_spike_time;
    tempcoder.clip_gradient() = clip_gradient;
    tempcoder.clip_derivative() = clip_derivative;
    tempcoder.input_jittering_params().sigma = input_jittering_sigma;
    tempcoder.input_jittering_params().mean = input_jittering_mean;
    tempcoder.input_jittering_params().new_spike_probability =
        input_jittering_new_spike_probability;
    tempcoder.noninput_jittering_params().sigma = noninput_jittering_sigma;
    tempcoder.noninput_jittering_params().mean = noninput_jittering_mean;
    tempcoder.noninput_jittering_params().new_spike_probability =
        noninput_jittering_new_spike_probability;
    tempcoder.decay_params().set_decay_rate(decay_rate);
    tempcoder.use_adam() = use_adam;

    Runner runner(&tempcoder, &problem, num_threads);
    runner.batch_size_ = batch_size;
    runner.train_options_ = train_options;
    runner.gradient_landscape_options_ = gradient_landscape_options;

    int best_validation_accuracy = -1;
    int best_train_accuracy = -1;
    Tempcoder* best_model = nullptr;
    std::vector<int64_t> timings;
    int64_t t_nanos = 0;

    const std::string initial_timestamp =
        absl::FormatTime("%Y%m%d%H%M%S", absl::Now(), absl::UTCTimeZone());
    if (write_all_models) {
      DumpArgs(output_dir, initial_timestamp,
               ProblemTypeToString(problem.problem_type()), argc, argv);
      tempcoder.WriteNetworkToFile(file::JoinPath(
          output_dir,
          absl::StrFormat("%s_%s_initial", initial_timestamp,
                          ProblemTypeToString(problem.problem_type()))));
    }
    std::default_random_engine generator(std::random_device{}());

    // A run does training + validation, then tests the best model.
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
      if (shuffle_between_epochs) {
        shuffle(problem.train_examples().begin(),
                problem.train_examples().end(), generator);
      }

      // Do training pass.
      int train_correct = 0;
      VectorXd train_errors(problem.train_examples().size());
      t_nanos = absl::GetCurrentTimeNanos();
      for (size_t train_ind = 0; train_ind < problem.train_examples().size();
           train_ind += batch_size) {
        train_correct += runner.ProcessBatch(train_ind, &train_errors);
      }

      int64_t t_nanos_2 = absl::GetCurrentTimeNanos();
      timings.push_back((t_nanos_2 - t_nanos) / 1000000);

      // Compute accuracy on validation set.
      int valid_correct;
      double validation_error;
      std::tie(valid_correct, validation_error) =
          runner.FeedforwardNoUpdatesParallel(&problem.validation_examples());
      if (valid_correct > best_validation_accuracy) {
        best_validation_accuracy = valid_correct;
        best_model =
            new Tempcoder(tempcoder.layer_sizes(), tempcoder.pulses(),
                          tempcoder.fire_threshold(), tempcoder.weights());
        best_model->decay_params().set_decay_rate(
            tempcoder.decay_params().rate());
      }

      // If no validation set, save model that trained best.
      if (train_correct > best_train_accuracy) {
        best_train_accuracy = train_correct;
        if (n_validation == 0) {
          best_model =
              new Tempcoder(tempcoder.layer_sizes(), tempcoder.pulses(),
                            tempcoder.fire_threshold(), tempcoder.weights());
          best_model->decay_params().set_decay_rate(
              tempcoder.decay_params().rate());
        }
      }

      // Log results of this epoch.
      BasicStats<double> error_stats(train_errors);
      float validation_accuracy =
          valid_correct * 100.0 / problem.validation_examples().size();

      IHM_LOG(LogSeverity::INFO,
              absl::StrFormat(
                  "run %d\tepoch %d\ttrain_error "
                  "%.2f\ttrain_acc.%% %f\tvalid acc.%% %f\telapsed "
                  "%dms\tSync pulses: %s\tLR: %.8f\tbatch sz: %d",
                  run, epoch, error_stats.mean(),
                  train_correct * 100.0 / problem.train_examples().size(),
                  validation_accuracy, timings.back(),
                  PPrintVectorXXd(tempcoder.pulses()),
                  tempcoder.learning_rate(), static_cast<int>(batch_size)));

      if (print_train_set) PrintExamples(problem.train_examples());
      if (write_all_models) {
        const std::string timestamp =
            absl::FormatTime("%Y%m%d%H%M%S", absl::Now(), absl::UTCTimeZone());
        tempcoder.WriteNetworkToFile(file::JoinPath(
            output_dir,
            absl::StrFormat("%s_%s_%s_validation_%.2f", initial_timestamp,
                            timestamp,
                            ProblemTypeToString(problem.problem_type()),
                            validation_accuracy)));
      }

      // Update model for next epoch.
      tempcoder.learning_rate() *= learning_rate_multiplier;
      batch_size = std::min(batch_size * batch_size_multiplier,
                            static_cast<float>(batch_size_clip));
    }

    float final_train_valid_accuracy, final_test_accuracy;
    // Compute accuracy on test set.
    Runner test_runner(best_model, &problem);
    std::tie(final_train_valid_accuracy, final_test_accuracy) =
        test_runner.TestModel(TestOptions{.compute_train_accuracy = true});
    test_errors.push_back(final_test_accuracy);

    // Print results.
    if (print_test_set) PrintExamples(problem.test_examples());
    if (print_best_model) {
      best_model->PrintNetwork();
      std::cout << "layer_nodes = " << VecToString(layer_sizes) << std::endl;
    }

    if (write_best_model) {
      const std::string timestamp =
          absl::FormatTime("%Y%m%d%H%M%S", absl::Now(), absl::UTCTimeZone());
      best_model->WriteNetworkToFile(file::JoinPath(
          output_dir,
          absl::StrFormat("%s_%s_%s_train_%.2f_test_%.2f", initial_timestamp,
                          timestamp,
                          ProblemTypeToString(problem.problem_type()),
                          final_train_valid_accuracy, final_test_accuracy)));
      DumpArgs(output_dir, initial_timestamp,
               ProblemTypeToString(problem.problem_type()), argc, argv);
    }

    BasicStats<int64_t> timing_stats(timings);

    std::cout << "Mean (s.d.) elapsed time per epoch: " << timing_stats.mean()
              << " (" << timing_stats.stddev() << ") ms\n"
              << "Accuracy on train+validation sets: "
              << final_train_valid_accuracy
              << "%\nAccuracy on test set: " << test_errors.back() << "%\n"
              << std::endl;

    if (absl::GetFlag(FLAGS_wait_for_input_between_runs)) std::cin.get();
  }
  BasicStats<double> test_stats(test_errors);
  std::cout << "Mean (S.D.) accuracy on test sets: " << test_stats.mean()
            << " (" << test_stats.stddev() << ") %\n"
            << std::endl;
}

}  // namespace ihmehimmeli

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  ihmehimmeli::Main(argc, argv);
  return 0;
}
