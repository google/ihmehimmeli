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

#ifndef IHMEHIMMELI_TEMPCODING_TEMPCODER_H_
#define IHMEHIMMELI_TEMPCODING_TEMPCODER_H_

#include <math.h>

#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "common/data_parallel.h"
#include "common/util.h"

namespace ihmehimmeli {

struct Prediction {
  VectorXd outputs;
  bool is_correct;
  double error;
  double first_output_spike_time;

  // Stores (mean, standard deviation) of spike times of neurons in each layer.
  std::vector<BasicStats<double>> spike_stats_per_layer;
};

struct TempcoderOptions {
  bool update_all_datapoints = false;
  bool compute_gradients = false;
  bool compute_spike_stats = false;
  bool print_activations = false;
  bool print_trace = false;
  bool draw_trace = false;
  bool log_embeddings = false;  // only used when there is a single hidden layer
  bool check_gradient = false;
};

struct WeightInitializationOptions {
  // Uses glorot initialization if true, uniform initialization if false.
  bool use_glorot_initialization = true;

  // Bounds for uniform initialization.
  double weights_lower_bound = 0.8;
  double weights_upper_bound = 0.9;

  // Mean multipliers for glorot initialization.
  double nonpulse_weight_mean_multiplier = 0.0;
  double pulse_weight_mean_multiplier = 0.0;

  // Seed to be used for initialization, if set.
  absl::optional<int> seed;
};

// Generates evenly spaced pulses in the input range.
// Example: GeneratePulses(4, {0.0, 1.0}) -> {0.2, 0.4, 0.6, 0.8}
VectorXd GeneratePulses(int n_pulses, std::pair<double, double> input_range);

// Computes output correctness.
// `Outputs` is a vector of spike timings.
// `Targets` is a one-hot encoded vector. That is, only one value is 1
// (meaning it should spike first) and the others are 0.
// Returns true iff the min element in `outputs` is in the same location as the
// max element in `targets`.
// If no neuron spiked in `outputs`, returns false.
bool IsCorrectSpikingOutput(const VectorXd& outputs, const VectorXd& targets);

// Returns a vector `sorted_indices` of the same size as `activations`, such
// that `activations` indexed by `sorted_indices` gives an earliest-spike-first
// sorted vector. Example: {2.5, kNpSpike, 0.5} -> {1, 2, 0}
std::vector<size_t> GetSortedIndices(const VectorXd& activations);

// Returns a vector of the same size as `activations` containing
// exp(`decay_rate` * x) for every element x in `activations`.
VectorXd ExponentiateSortedValidSpikes(
    const VectorXd& activations, const std::vector<size_t>& sorted_indices,
    double decay_rate);

template <class T>
void PrintLayerStats(int layer, absl::string_view info,
                     const BasicStats<T>& stats) {
  std::cout << absl::StrFormat("Layer#%d %s (mean median sd): %.5f %.5f %.5f",
                               layer, info, stats.mean(), stats.median(),
                               stats.stddev())
            << std::endl;
}

// This class operates with decay parameters for Tempcoder.
class DecayParams {
 public:
  // The decay rate K of the potential function: f(t) = t * exp(-K * t)
  double rate() const { return rate_; }

  // Inverse of the decay rate (1 / decay_rate). Precomputed for speed.
  double rate_inverse() const { return rate_inverse_; }

  // Updates the decay rate and the minimum significant weight.
  void set_decay_rate(const double decay_rate) {
    rate_ = decay_rate;
    rate_inverse_ = 1.0 / rate_;  // allow zero decay rate for experiments
    dual_exponential_scale_ = 4 * rate_inverse_ / M_E;
  }

  // Factor to multiply the dual exponential by to make it have the same maximum
  // as the alpha function.
  double dual_exponential_scale() const { return dual_exponential_scale_; }

 private:
  double rate_ = 1.0;
  double rate_inverse_ = 1.0;
  double dual_exponential_scale_ = 4 / M_E;
};

struct JitteringParams {
  double sigma = 0.0;
  double mean = 0.0;
  double new_spike_probability = 0.0;
};

// A multi-layer network with spiking leaky "double-exponential" neurons
// that encodes information in the timing of its spikes (i.e. first spike in
// a layer is the winner). Every neuron is allowed to spike once during a run.
// If the flag `use_dual_exponential` is enabled, replaces the alpha function
// with a dual exponential.
class Tempcoder {
 public:
  // Neuron output if no spike.
  static constexpr double kNoSpike = std::numeric_limits<double>::max();

  // Clips val to the range [-limit, limit], if limit > 0.
  static double ClipDerivative(double val, double limit) {
    if (limit == 0.0) return val;
    if (val < -limit) val = -limit;
    if (val > limit) val = limit;
    return val;
  }

  static double ComputePotentialDualExponential(
      const VectorXd& weights, const VectorXd& inputs,
      const double point_in_time, const double decay_rate = 1.0,
      const double dual_exponential_scale = 4 / M_E);
  static double ComputePotentialAlpha(const VectorXd& weights,
                                      const VectorXd& inputs,
                                      const double point_in_time,
                                      const double decay_rate = 1.0);

  // Computes output spike time, as well as derivatives wrt. weights and inputs
  // if the dspike_dw/dspike_dt pointers are not nullptr.
  static double ActivateNeuronDualExponential(
      const VectorXd& weights, const VectorXd& inputs,
      const VectorXd& exp_inputs, const std::vector<size_t>& sorted_ind,
      double fire_threshold, const DecayParams& decay_params = DecayParams(),
      double* dspike_dw = nullptr, double* dspike_dt = nullptr);
  static double ActivateNeuronAlpha(
      const VectorXd& weights, const VectorXd& inputs,
      const VectorXd& exp_inputs, const std::vector<size_t>& sorted_ind,
      double fire_threshold, const DecayParams& decay_params = DecayParams(),
      double* dspike_dw = nullptr, double* dspike_dt = nullptr);

  // Loads a model in the format described by WriteNetworkToFile.
  static std::unique_ptr<Tempcoder> LoadTempcoderFromFile(
      const std::string& model_filename);

  // `layer_sizes` is an ordered vector holding the number of neurons in each
  // layer, starting with the input layer and finishing with the output layer,
  // without counting any pulses.
  // `pulses` is a number-of-layers sized vector that represents the initial
  // timings of the pulses on each layer.
  // `fire_threshold` is the membrane potential where the neuron spikes.
  // `weight_options` define weight initialisation.
  // `use_dual_exponential` specifies if the function e^{-k*t} - e^{-2*k*t}.
  // instead of the default alpha function t * e^{-k*t}.
  Tempcoder(const std::vector<int>& layer_sizes, const VectorXXd& pulses,
            double fire_threshold,
            WeightInitializationOptions weight_options =
                WeightInitializationOptions(),
            bool use_dual_exponential = false);
  Tempcoder(const std::vector<int>& layer_sizes, const VectorXXd& pulses,
            double fire_threshold, const VectorXXXd& weights);
  Tempcoder(const std::vector<int>& layer_sizes, const VectorXXd& pulses,
            double fire_threshold, const DecayParams decay_params,
            double latency, const VectorXXXd& weights);

  Tempcoder(const Tempcoder& other)
      : Tempcoder(other.layer_sizes_, other.pulses_, other.fire_threshold_,
                  other.decay_params_, other.latency_, other.weights_) {}

  const std::vector<int>& layer_sizes() const { return layer_sizes_; }
  int n_pulses() const { return n_pulses_; }
  int n_connections() const { return n_connections_; }

  // Performs a feedforward pass in inputs and optionally computes derivatives.
  // Returns a Prediction that includes the raw outputs (before softmax).
  const Prediction FeedforwardAndOptionallyComputeGradients(
      const VectorXd& inputs, const VectorXd& targets, int original_class = -1,
      absl::Mutex* mutex = nullptr,
      const TempcoderOptions& tempcoder_options = TempcoderOptions());

  // Compute cross-entropy (or MSE if `use_mse_`) loss based on the last
  // feedforward pass.
  double ComputeLossWithPenalty(const VectorXd& outputs,
                                const VectorXd& targets,
                                const VectorXd& spike_times);

  // Computes and applies training updates.
  void AccumulateAndApplyUpdates(ThreadPool* pool = nullptr);

  // Clear accumulated backprop updates.
  void ClearBatch();

  // Returns a vector of estimated gradients obtained by varying each parameter,
  // in the same order as d_weights and d_pulses (flattened and concatenated).
  VectorXd CheckGradient(const VectorXd& inputs, const VectorXd& targets);

  // Adjust a given set of `inputs` in order to minimize the spike time of a
  // `target_neuron` in `target_layer`. Indices are zero-based.
  // Input layer is 0. The spike times of non-target neurons in `target_layer`
  // are maximized unless `maximize_nontargets` is set to false.
  VectorXd AdjustInput(const VectorXd& inputs, int target_layer,
                       int target_neuron, double adjustment_strength = 1.0,
                       bool maximize_nontargets = true);

  double fire_threshold() const { return fire_threshold_; }
  double& learning_rate() { return learning_rate_; }
  double& learning_rate_pulses() { return learning_rate_pulses_; }
  double& penalty_no_spike() { return penalty_no_spike_; }
  double& penalty_output_spike_time() { return penalty_output_spike_time_; }
  double& clip_gradient() { return clip_gradient_; }
  double& clip_derivative() { return clip_derivative_; }
  JitteringParams& input_jittering_params() { return input_jittering_params_; }
  JitteringParams& noninput_jittering_params() {
    return noninput_jittering_params_;
  }
  DecayParams& decay_params() { return decay_params_; }
  const VectorXXd& pulses() const { return pulses_; }
  const VectorXXXd& weights() const { return weights_; }
  const std::vector<std::unique_ptr<VectorXXd>>& d_pulses_per_call() {
    return d_pulses_per_call_;
  }
  const std::vector<std::unique_ptr<VectorXXXd>>& d_weights_per_call() {
    return d_weights_per_call_;
  }
  bool& use_adam() { return use_adam_; }
  bool& use_mse() { return use_mse_; }

  void set_pulses(const VectorXXd& new_pulses) { pulses_ = new_pulses; }
  void set_weights(const VectorXXXd& new_weights) { weights_ = new_weights; }

  // Returns stats of flattened nonpulse weights between `layer`+1 and `layer`.
  BasicStats<double> MakeNonPulseWeightStats(int layer) const;
  // Returns stats of flattened pulse weights between `layer`+1 and `layer`.
  BasicStats<double> MakePulseWeightStats(int layer) const;

  void PrintNetwork() const;
  void PrintNetworkStats() const;
  void DrawNetwork() const;
  // Writes the network to the specified full filename.
  // Format: firing_threshold number_of_layers layer_size_values
  // number_of_pulses pulse_values weight_values.
  // All values are delimited by spaces.
  // Weights are in layer - post - pre order.
  void WriteNetworkToFile(const std::string& filename) const;

  // Writes embeddings_log_ to the specified full filename.
  void WriteEmbeddingsToFile(const std::string& filename) const;

  void ClearEmbeddings() { this->embeddings_log_ = ""; }
  std::string GetEmbeddings() const { return this->embeddings_log_; }

  // Only used if `use_mse_`, as a reference to align the outputs to.
  // Could also be *std::min_element(outputs.begin(), outputs.end()).
  double latency_ = 1.0;

 private:
  const double fire_threshold_;
  DecayParams decay_params_;
  double learning_rate_ = 0.0;
  double learning_rate_pulses_ = 0.0;
  double penalty_no_spike_ = 0.0;
  double penalty_output_spike_time_ = 0.0;
  double clip_gradient_ = 0.0;
  double clip_derivative_ = 0.0;
  JitteringParams input_jittering_params_;
  JitteringParams noninput_jittering_params_;
  bool use_adam_ = true;
  bool use_mse_ = false;
  const std::vector<int> layer_sizes_;

  // The number of synchronization pulses per layer.
  const int n_pulses_;

  const int n_connections_;  // store number of total connections in the net

  // String that logs middle layer representations if specified by
  // tempcoder_options and if the network has exactly one hidden layer. Can be
  // cleared, retrieved and logged with ClearEmbeddingsLog(),
  // GetEmbeddingsLog(), WriteEmbeddingsToFile().
  // Log format is one line per example, values are delimited by a space, the
  // last value is the original_class of the Example.
  std::string embeddings_log_ = "";

  void PrintActivations(const VectorXXd& activations) const;
  void PrintTrace(const VectorXXd& activations) const;
  void DrawTrace(const VectorXXd& activations) const;

  void ComputeGradients(const VectorXd& final_outputs,
                        const VectorXXd& activations, const VectorXd& targets,
                        const VectorXXXd& dspike_dw,
                        const VectorXXXd& dspike_dt,
                        VectorXXXd* d_weights_local, VectorXXd* d_pulses_local);

  void AddJittering(const JitteringParams& params, VectorXd* activations);

  // Each ParameterInfo identifies one vector of parameters - either weights or
  // pulses.
  struct ParameterInfo {
    enum { kWeightVector, kPulseVector } type;
    int layer;
    int post_neuron;
  };

  void AccumulateSingleDerivativeVector(const ParameterInfo& param_info,
                                        VectorXd* out) const;

  std::vector<ParameterInfo> ParameterIndices() const;

  // Weights of the network in the form weights_[layer][post][pre], with:
  // layer from 0 to n_layers-1
  // post from 0 to layer_sizes[layer+1]
  // pre from 0 to layer_sizes[layer]+n_pulses
  // The last n_pulse weights in each vector correspond to the connection
  // from node [layer][post] to the pulses in the previous layer.
  VectorXXXd weights_;

  // Synchronization pulses, with n_pulses_ per layer.
  // The last set of pulses are not used in practice, but we can later connect
  // them to the input layer to get a recursive topography.
  VectorXXd pulses_;

  // Storage for per-example derivatives. Accessing these vectors during
  // FeedforwardAndOptionallyComputeGradients is protected by a mutex (for
  // allocating the vector of data). During gradient computation, as the
  // vectors are not modified, no mutex is used.
  std::vector<std::unique_ptr<VectorXXXd>> d_weights_per_call_;
  std::vector<std::unique_ptr<VectorXXd>> d_pulses_per_call_;

  // If true, uses dual exponential (exp(-kx)-exp(-2kx)), otherwise leaky
  // exponential (x*exp(-kx)).
  bool use_dual_exponential_;

  // Adam optimizer.
  void InitAdam();

  VectorXXXd adam_momentum_weights_;
  VectorXXXd adam_var_weights_;
  VectorXXd adam_momentum_pulses_;
  VectorXXd adam_var_pulses_;
  int adam_epoch_;
};

}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_TEMPCODING_TEMPCODER_H_
