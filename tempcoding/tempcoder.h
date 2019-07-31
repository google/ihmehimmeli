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
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tempcoding/util.h"

namespace ihmehimmeli {

typedef std::vector<double> VectorXd;
typedef std::vector<VectorXd> VectorXXd;
typedef std::vector<VectorXXd> VectorXXXd;

typedef std::vector<char> VectorXb;
typedef std::vector<VectorXb> VectorXXb;
typedef std::vector<VectorXXb> VectorXXXb;

struct Prediction {
  VectorXd outputs;
  bool is_correct;
  double error;
  double first_output_spike_time;

  // Stores (mean, standard deviation) of spike times of neurons in each layer.
  std::vector<BasicStats<double>> spike_stats_per_layer;
};

struct TrainOptions {
  bool update_all_datapoints = false;
  bool accumulate_weight_updates = false;
  bool compute_spike_stats = false;
  bool print_activations = false;
  bool print_trace = false;
  bool draw_trace = false;
  bool print_causal_set = false;
};

struct TestOptions {
  bool compute_train_accuracy = false;
  bool print_train_spike_stats = false;
  bool print_test_spike_stats = false;
  bool print_network = false;
  bool print_network_stats = false;
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

// Formats the given VectorXXd to a string similar to: "[[1.0], [], [3.1, 4.1]]"
// Bundled with the Tempcoder because we operate with lots of VectorXX*.
inline std::string PPrintVectorXXd(const VectorXXd& v) {
  return absl::StrCat(
      "[",
      absl::StrJoin(v, ", ",
                    [](std::string* out, const VectorXd& w) {
                      absl::StrAppend(out, "[", absl::StrJoin(w, ", "), "]");
                    }),
      "]");
}

// Pretty printing, same as PPrintVectorXXd but for VectorXXXd.
inline std::string PPrintVectorXXXd(const VectorXXXd& v) {
  return absl::StrCat("[",
                      absl::StrJoin(v, ", ",
                                    [](std::string* out, const VectorXXd& w) {
                                      absl::StrAppend(out, PPrintVectorXXd(w));
                                    }),
                      "]");
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

  static double ActivateNeuronDualExponential(
      const VectorXd& weights, const VectorXd& inputs,
      const VectorXd& exp_inputs, const std::vector<size_t>& sorted_ind,
      double fire_threshold, VectorXb* causal_set, double* A, double* B,
      double* D, const DecayParams decay_params = DecayParams());
  static double ActivateNeuronAlpha(
      const VectorXd& weights, const VectorXd& inputs,
      const VectorXd& exp_inputs, const std::vector<size_t>& sorted_ind,
      double fire_threshold, VectorXb* causal_set, double* A, double* B,
      double* W, const DecayParams decay_params = DecayParams());

  // Loads a model in the format described by WriteNetworkToFile.
  static Tempcoder LoadTempcoderFromFile(const std::string& model_filename);

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

  const std::vector<int>& layer_sizes() const { return layer_sizes_; }
  int n_pulses() const { return n_pulses_; }
  int n_connections() const { return n_connections_; }

  // Performs a feedforward pass in inputs and optionally accumulates weight
  // updates. Returns a Prediction that includes the raw outputs (before
  // softmax).
  const Prediction FeedforwardAndOptionallyAccumulateUpdates(
      const VectorXd& inputs, const VectorXd& targets,
      absl::Mutex* mutex = nullptr,
      const TrainOptions& train_options = TrainOptions());

  // Compute cross-entropy loss based on the last feedforward pass.
  double ComputeCrossEntropyLossWithPenalty(const VectorXd& outputs,
                                            const VectorXd& targets,
                                            const VectorXd& spike_times);

  // A helper which handles adam updates in particular.
  std::tuple<VectorXXXd, VectorXXd> ComputeAdamUpdates();
  // Compute weight updates based on accumulated gradients.
  // Returns a tuple of updates to weights and updates to pulses.
  std::tuple<VectorXXXd, VectorXXd> ComputeUpdates();

  // Apply a set of updates to the weights and pulses within a training context.
  // Also updates the Adam epoch and calls ClearBatch().
  void ApplyTrainingUpdates(const VectorXXXd& weight_updates,
                            const VectorXXd& pulse_updates);

  // Apply a set of updates to the weights and pulses of the network. Can be
  // used to modify a network outside training.
  void ChangeNetwork(const VectorXXXd& weight_updates,
                     const VectorXXd& pulse_updates);

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
  const VectorXXd& d_pulses() const { return d_pulses_; }
  const VectorXXXd& d_weights() const { return d_weights_; }
  bool& use_adam() { return use_adam_; }

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

  const std::vector<int> layer_sizes_;

  // The number of synchronization pulses per layer.
  const int n_pulses_;

  const int n_connections_;  // store number of total connections in the net

  void PrintActivations(const VectorXXd& activations) const;
  void PrintTrace(const VectorXXd& activations) const;
  void DrawTrace(const VectorXXd& activations) const;
  void PrintCausalSets(const VectorXXd& activations,
                       const VectorXXXb& causal_sets) const;

  // Derivatives.
  double WeightDerivativeDualExponential(const VectorXXd& activations,
                                         const VectorXXXb& causal_sets,
                                         const VectorXXd& a, const VectorXXd& b,
                                         const VectorXXd& d, int layer,
                                         int post, int pre);
  double WeightDerivativeAlpha(const VectorXXd& activations,
                               const VectorXXXb& causal_sets,
                               const VectorXXd& a, const VectorXXd& b,
                               const VectorXXd& w, int layer, int post,
                               int pre);

  double ActivationDerivativeDualExponential(const VectorXXd& activations,
                                             const VectorXXXb& causal_sets,
                                             const VectorXXd& a,
                                             const VectorXXd& b,
                                             const VectorXXd& d, int layer,
                                             int post, int pre);
  double ActivationDerivativeAlpha(const VectorXXd& activations,
                                   const VectorXXXb& causal_sets,
                                   const VectorXXd& a, const VectorXXd& b,
                                   const VectorXXd& w, int layer, int post,
                                   int pre);

  void ComputeGradients(const VectorXd& exp_outputs,
                        const VectorXXd& activations, const VectorXd& targets,
                        const VectorXXXb& causal_sets, const VectorXXd& a,
                        const VectorXXd& b, const VectorXXd& w,
                        VectorXXXd* d_weights_local, VectorXXd* d_pulses_local);

  void AddJittering(const JitteringParams& params, VectorXd* activations);

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

  // Accumulated error for each weight, in the same form as weights_.
  VectorXXXd d_weights_;

  // Accumulated error for pulse spike times.
  VectorXXd d_pulses_;

  // The actual number of examples contributing to current updates. If
  // update_all_datapoints is true, then this is always equal to the batch size
  // (from runner.h). If update_all_datapoints is false, this is the number of
  // examples in the batch that were wrongly classified, i.e. anything between 0
  // and the batch size.
  int num_updates_;

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
