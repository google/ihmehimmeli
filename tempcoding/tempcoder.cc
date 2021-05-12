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

#include "tempcoding/tempcoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "common/data_parallel.h"
#include "common/file_passthrough.h"
#include "common/lambertw.h"
#include "common/util.h"

namespace ihmehimmeli {
VectorXd GeneratePulses(const int n_pulses,

                        const std::pair<double, double> input_range) {
  IHM_CHECK(n_pulses >= 0);
  IHM_CHECK(input_range.first <= input_range.second);
  VectorXd pulses_per_layer(n_pulses);
  const double pulse_spacing =
      (input_range.second - input_range.first) / (pulses_per_layer.size() + 1);
  for (int i = 0; i < pulses_per_layer.size(); ++i) {
    pulses_per_layer[i] = input_range.first + pulse_spacing * (i + 1);
  }
  return pulses_per_layer;
}

bool IsCorrectSpikingOutput(const VectorXd& outputs, const VectorXd& targets) {
  IHM_CHECK(outputs.size() == targets.size(),
            "Output and targets sizes must be equal.");
  IHM_CHECK(!outputs.empty(), "Targets and outputs must not be empty.");
  // We consider the minimum element in `outputs`, i.e. the earliest spike,
  // and the maximum element of `targets`, i.e. the value in the one-hot encoded
  // vector that marks the location of the neuron that should spike first.
  if (*min_element(outputs.begin(), outputs.end()) == Tempcoder::kNoSpike)
    return false;
  return (
      distance(begin(outputs), min_element(outputs.begin(), outputs.end())) ==
      distance(begin(targets), max_element(targets.begin(), targets.end())));
}

namespace {
// Earliest-first comparator for pairs of the form [neuron_info, spike_time].
template <typename T>
class CompareNeuronTiming {
 public:
  bool operator()(const std::pair<T, double>& neuron1,
                  const std::pair<T, double>& neuron2) {
    return neuron1.second > neuron2.second;
  }
};

}  // namespace

constexpr double Tempcoder::kNoSpike;

double Tempcoder::ComputePotentialAlpha(const VectorXd& weights,
                                        const VectorXd& inputs,
                                        const double point_in_time,
                                        const double decay_rate) {
  IHM_CHECK(weights.size() == inputs.size(),
            "Weights and inputs vectors must have the same size.");
  IHM_CHECK(point_in_time >= 0, "t must be nonnegative.");
  double potential = 0.0;
  for (int i = 0; i < inputs.size(); ++i) {
    double t_i = std::max(point_in_time - inputs[i], 0.0);
    potential += weights[i] * t_i * exp(-t_i * decay_rate);
  }
  return potential;
}

double Tempcoder::ComputePotentialDualExponential(
    const VectorXd& weights, const VectorXd& inputs, const double point_in_time,
    const double decay_rate, const double dual_exponential_scale) {
  IHM_CHECK(weights.size() == inputs.size(),
            "Weights and inputs vectors must have the same size.");
  IHM_CHECK(point_in_time >= 0, "t must be nonnegative.");
  double potential = 0.0;
  for (int i = 0; i < inputs.size(); ++i) {
    double t_i = std::max(point_in_time - inputs[i], 0.0);
    potential += weights[i] *
                 (exp(-t_i * decay_rate) - exp(-2 * t_i * decay_rate)) *
                 dual_exponential_scale;
  }
  return potential;
}

// Computes the set of neurons from previous layer that spiked before current
// neuron. Returns time of spike of output neuron, and updates dspike_dw and
// dspike_dt with the derivatives of the spike time with respect to each input.
// More precisely, dspike_dt[i] gets set to d spike_time / d inputs[i],
// and dspike_dw[i] gets set to d spike_time / d weights[i].
double Tempcoder::ActivateNeuronDualExponential(
    const VectorXd& weights, const VectorXd& inputs, const VectorXd& exp_inputs,
    const std::vector<size_t>& sorted_ind, double fire_threshold,
    const DecayParams& decay_params, double* dspike_dw, double* dspike_dt) {
  IHM_CHECK(weights.size() == inputs.size(),
            "Weights and inputs vectors must have the same size.");
  IHM_CHECK(exp_inputs.size() == inputs.size(),
            "Exponentiated inputs and inputs vectors must have the same size.");
  IHM_CHECK(sorted_ind.size() == inputs.size(),
            "Sorted_ind and inputs vectors must have the same size.");
  IHM_CHECK(fire_threshold > 0, "Firing threshold must be strictly positive.");

  double spike_time = kNoSpike;
  // A = sum of wi * exp(K * t_i)
  // B = sum of wi * exp(K * t_i) * t_i
  // where i represents causal indices causing firing.
  // D = determinant of the second-degree equation: sqrt(A*A-4*B*fire_threshold)
  double A = 0.0;
  double B = 0.0;
  double D = 0.0;

  // Process incoming spikes one by one.
  for (const size_t spike_ind : sorted_ind) {
    // Check if neuron spiked before incoming spike (or incoming is kNoSpike).
    if (spike_time <= inputs[spike_ind]) {
      break;
    }
    // Reset spike time, in case an inhibitory input cancels a potential spike.
    spike_time = kNoSpike;

    const double w_exp_z = weights[spike_ind] * exp_inputs[spike_ind] *
                           decay_params.dual_exponential_scale();
    A += w_exp_z;
    B += w_exp_z * exp_inputs[spike_ind];

    // Don't do anything if the coefficients are too small, or negative: this
    // would produce NaNs.
    if (A < 1e-20 && B < 1e-20) continue;

    // Compute square of determinant of the second degree polynomial.
    double D_square = A * A - 4 * B * fire_threshold;

    // If this argument is valid, we have a potential spike.
    if (D_square < 0) continue;

    D = std::sqrt(D_square);
    spike_time = (A - D) / (2 * B);
    if (spike_time < 0) {
      spike_time = kNoSpike;
      continue;
    }
    spike_time = -std::log(spike_time) * decay_params.rate_inverse();

    // For inhibitory weights, this might be a false alarm.
    // This is not the same as spike_time < inputs[spike_ind]: it is also true
    // for NaNs.
    if (!(spike_time >= inputs[spike_ind])) spike_time = kNoSpike;
  }

  // Compute derivatives of the spike time with respect to the inputs and the
  // weights, if requested. If there is no output spike, all derivatives are set
  // to 0.
  if (dspike_dt != nullptr && dspike_dw != nullptr) {
    if (spike_time == kNoSpike) {  // no output spike
      std::fill(dspike_dw, dspike_dw + inputs.size(), 0.0);
      std::fill(dspike_dt, dspike_dt + inputs.size(), 0.0);
      return kNoSpike;
    }
    for (size_t pre = 0; pre < inputs.size(); pre++) {
      const double tp = inputs[pre];
      if (tp > spike_time) {  // output precedes input
        dspike_dt[pre] = dspike_dw[pre] = 0;
        continue;
      }
      const double wp = weights[pre];
      const double e_K_tp = exp_inputs[pre];
      const double scale = decay_params.dual_exponential_scale();
      const double K_inverse = decay_params.rate_inverse();
      dspike_dw[pre] =
          scale * e_K_tp * K_inverse *
          (2 * fire_threshold * B * e_K_tp + (D - A) * (D * e_K_tp + B)) /
          (B * D * (D - A));
      dspike_dt[pre] =
          scale * e_K_tp * wp *
          (4 * fire_threshold * B * e_K_tp + (D - A) * (2 * D * e_K_tp + B)) /
          (B * D * (D - A));
    }
  }
  return spike_time;
}

double Tempcoder::ActivateNeuronAlpha(
    const VectorXd& weights, const VectorXd& inputs, const VectorXd& exp_inputs,
    const std::vector<size_t>& sorted_ind, double fire_threshold,
    const DecayParams& decay_params, double* dspike_dw, double* dspike_dt) {
  IHM_CHECK(weights.size() == inputs.size(),
            "Weights and inputs vectors must have the same size.");
  IHM_CHECK(exp_inputs.size() == inputs.size(),
            "Exponentiated inputs and inputs vectors must have the same size.");
  IHM_CHECK(sorted_ind.size() == inputs.size(),
            "Sorted_ind and inputs vectors must have the same size.");
  IHM_CHECK(fire_threshold > 0, "Firing threshold must be strictly positive.");

  double spike_time = kNoSpike;
  // A = sum of wi * exp(K * t_i)
  // B = sum of wi * exp(K * t_i) * t_i
  // where i represents causal indices causing firing.
  // W = LambertW(-K * fire_threshold / A * exp(K * B / A)
  double A = 0.0;
  double B = 0.0;
  double W = 0.0;
  double b_over_a = 0.0;

  // Process incoming spikes one by one.
  for (const size_t spike_ind : sorted_ind) {
    // Check if neuron spiked before incoming spike (or incoming is kNoSpike).
    if (spike_time <= inputs[spike_ind]) {
      break;
    }
    // Reset spike time, in case an inhibitory input cancels a potential spike.
    spike_time = kNoSpike;

    const double w_exp_z = weights[spike_ind] * exp_inputs[spike_ind];
    A += w_exp_z;
    B += w_exp_z * inputs[spike_ind];

    // The value of the first derivative of the activation function in the
    // intersection point with the fire threshold is given by *A multiplied by a
    // never-negative value. Thus, if *A is negative the intersection will be in
    // a decreasing-potential area, and thus not a spike.
    if (A < 0) continue;

    b_over_a = B / A;

    // Compute Lambert W argument for solving the threshold crossing.
    const double lambert_arg = -decay_params.rate() * fire_threshold / A *
                               exp(decay_params.rate() * b_over_a);

    if (lambert_arg >= kMinLambertArg && lambert_arg <= kMaxLambertArg) {
      IHM_CHECK(
          LambertW0(lambert_arg, &W),
          absl::StrFormat("Error computing Lambert W on: %f", lambert_arg));

      spike_time = b_over_a - W * decay_params.rate_inverse();

      // For inhibitory weights, this might be a false alarm.
      // This is not the same as spike_time < inputs[spike_ind]: it is also true
      // for NaNs.
      if (!(spike_time >= inputs[spike_ind])) spike_time = kNoSpike;
    }
  }

  // Compute derivatives of the spike time with respect to the inputs and the
  // weights, if requested. If there is no output spike, all derivatives are set
  // to 0.
  if (dspike_dt != nullptr && dspike_dw != nullptr) {
    if (spike_time == kNoSpike) {  // no output spike
      std::fill(dspike_dw, dspike_dw + inputs.size(), 0.0);
      std::fill(dspike_dt, dspike_dt + inputs.size(), 0.0);
      return kNoSpike;
    }
    for (size_t pre = 0; pre < inputs.size(); pre++) {
      const double tp = inputs[pre];
      if (tp > spike_time) {  // output precedes input
        dspike_dt[pre] = dspike_dw[pre] = 0;
        continue;
      }
      const double wp = weights[pre];
      const double K = decay_params.rate();
      const double e_K_tp = exp_inputs[pre];
      const double K_inverse = decay_params.rate_inverse();
      dspike_dw[pre] =
          e_K_tp * (tp - b_over_a + W * K_inverse) / (A * (1.0 + W));
      dspike_dt[pre] =
          wp * e_K_tp * (K * (tp - b_over_a) + W + 1) / (A * (1.0 + W));
    }
  }
  return spike_time;
}

// Compute the number of connections in the network.
int CountConnections(const std::vector<int>& layer_sizes, int n_pulses) {
  int n_conn = 0;
  for (int layer = 1; layer < layer_sizes.size(); ++layer)
    n_conn += layer_sizes[layer] * (layer_sizes[layer - 1] + n_pulses);
  return n_conn;
}

int CountNonInputNodes(const std::vector<int>& layer_sizes) {
  return std::accumulate(layer_sizes.begin() + 1, layer_sizes.end(), 0);
}

Tempcoder::Tempcoder(const std::vector<int>& layer_sizes,
                     const VectorXXd& pulses, double fire_threshold,
                     WeightInitializationOptions weight_options,
                     bool use_dual_exponential)
    : fire_threshold_(fire_threshold),
      layer_sizes_(layer_sizes),
      n_pulses_(pulses.front().size()),
      n_connections_(CountConnections(layer_sizes, n_pulses_)),
      pulses_(pulses),
      use_dual_exponential_(use_dual_exponential) {
  // Check network has proper architecture.
  IHM_CHECK(layer_sizes_.size() >= 2,
            "Need at least 2 (input & output) layers.");
  for (int layer = 0; layer < layer_sizes_.size(); ++layer)
    IHM_CHECK(layer_sizes_[layer] > 0, "The number of nodes must be positive.");

  // Note weights are of the form weights_[layer][post_node][pre_node].
  // There is a set of sets of weights between each layer, so weights_.size()
  // is equal to the number of layers - 1.
  // Examples:
  // weights_.front()[post][pre]: weights between first hidden to input layer
  // weights_[n][post][pre]: weights between layer n+1 and layer n.
  // weights_.back()[post][pre]: weights between last hidden and  output layer
  weights_.resize(layer_sizes_.size() - 1);
  const int seed = weight_options.seed.value_or(std::random_device{}());
  std::default_random_engine generator(seed);
  // std::normal_distribution doesn't have per-sample distribution guarantees
  // so it's much safer to scale to our desired distribution manually.
  std::normal_distribution<double> standard_normal(0, 1);

  if (weight_options.use_glorot_initialization) {
    for (int layer = 0; layer < layer_sizes_.size() - 1; ++layer) {
      weights_[layer].resize(layer_sizes_[layer + 1]);
      for (int j = 0; j < weights_[layer].size(); ++j) {
        weights_[layer][j].resize(layer_sizes_[layer] + n_pulses_);
        double sigma =
            sqrt(2.0 / (weights_[layer].size() + weights_[layer][j].size()));
        for (int i = 0; i < weights_[layer][j].size(); ++i) {
          double mean = 0.0;
          if (i < layer_sizes_[layer]) {
            mean = weight_options.nonpulse_weight_mean_multiplier * sigma;
          } else {
            mean = weight_options.pulse_weight_mean_multiplier * sigma;
          }
          weights_[layer][j][i] = standard_normal(generator) * sigma + mean;
        }
      }
    }
  } else {
    // Random starting weights, including backward connections to sync pulses.
    std::uniform_real_distribution<double> dist(
        weight_options.weights_lower_bound, weight_options.weights_upper_bound);

    for (int layer = 0; layer < layer_sizes_.size() - 1; ++layer) {
      weights_[layer].resize(layer_sizes_[layer + 1]);
      for (int j = 0; j < weights_[layer].size(); ++j) {
        weights_[layer][j].resize(layer_sizes_[layer] + n_pulses_);
        for (int i = 0; i < weights_[layer][j].size(); ++i) {
          weights_[layer][j][i] = dist(generator);
        }
      }
    }
  }

  InitAdam();
  ClearBatch();
}

Tempcoder::Tempcoder(const std::vector<int>& layer_sizes,
                     const VectorXXd& pulses, double fire_threshold,
                     const VectorXXXd& weights)
    : fire_threshold_(fire_threshold),
      layer_sizes_(layer_sizes),
      n_pulses_(pulses.front().size()),
      n_connections_(CountConnections(layer_sizes, n_pulses_)),
      weights_(weights),
      pulses_(pulses),
      use_dual_exponential_(false) {
  InitAdam();
  ClearBatch();
}

Tempcoder::Tempcoder(const std::vector<int>& layer_sizes,
                     const VectorXXd& pulses, double fire_threshold,
                     const DecayParams decay_params, double latency,
                     const VectorXXXd& weights)
    : latency_(latency),
      fire_threshold_(fire_threshold),
      decay_params_(decay_params),
      layer_sizes_(layer_sizes),
      n_pulses_(pulses.front().size()),
      n_connections_(CountConnections(layer_sizes, n_pulses_)),
      weights_(weights),
      pulses_(pulses),
      use_dual_exponential_(false) {
  InitAdam();
  ClearBatch();
}

std::unique_ptr<Tempcoder> Tempcoder::LoadTempcoderFromFile(
    const std::string& model_filename) {
  double fire_threshold;
  double decay_rate;
  DecayParams decay_params;
  double latency;
  std::vector<int> layer_sizes;
  VectorXXd pulses;
  VectorXXXd weights;
  bool pulses_per_layer = false;
  bool decay_rate_latency = false;
  std::string model_contents;

  file::IhmFile f = file::OpenOrDie(model_filename, "r");
  IHM_CHECK(f.ReadWholeFileToString(&model_contents));
  f.Close();

  std::vector<absl::string_view> lines = absl::StrSplit(model_contents, '\n');

  for (absl::string_view line_raw : lines) {
    absl::string_view line = absl::StripAsciiWhitespace(line_raw);
    std::vector<absl::string_view> vals = absl::StrSplit(line, ' ');
    int pos = 0;

    // Check file version. Currently the only versions are "decay_rate+latency",
    // "pulses_per_layer" or the original version, which starts directly with
    // numeric data.
    if (!absl::SimpleAtod(vals[pos], &fire_threshold)) {
      if (vals[pos] == "pulses_per_layer") {
        pulses_per_layer = true;
      } else if (vals[pos] == "decay_rate+latency") {
        decay_rate_latency = true;
        pulses_per_layer = true;
      }
      IHM_CHECK(pulses_per_layer || decay_rate_latency,
                "Unknown network file version.");
      ++pos;
    }

    // Read fire threshold.
    IHM_CHECK(absl::SimpleAtod(vals[pos++], &fire_threshold));
    IHM_LOG(LogSeverity::INFO,
            absl::StrFormat("Loaded fire threshold: %f", fire_threshold));

    if (decay_rate_latency) {
      // Read decay rate
      IHM_CHECK(absl::SimpleAtod(vals[pos++], &decay_rate));
      IHM_LOG(LogSeverity::INFO,
              absl::StrFormat("Loaded decay rate: %f", decay_rate));
      decay_params.set_decay_rate(decay_rate);

      // Read latency
      IHM_CHECK(absl::SimpleAtod(vals[pos++], &latency));
      IHM_LOG(LogSeverity::INFO,
              absl::StrFormat("Loaded latency: %f", latency));
    }

    // Read number of layers and layers.
    int n_layers;
    IHM_CHECK(absl::SimpleAtoi(vals[pos++], &n_layers));
    layer_sizes.resize(n_layers);
    for (auto& layer_size : layer_sizes) {
      IHM_CHECK(absl::SimpleAtoi(vals[pos++], &layer_size));
    }
    IHM_LOG(LogSeverity::INFO, absl::StrFormat("Loaded layer sizes: %s",
                                               VecToString(layer_sizes)));

    // Read number of pulses and pulses.
    int n_pulses;
    IHM_CHECK(absl::SimpleAtoi(vals[pos++], &n_pulses));
    pulses.assign(n_layers, VectorXd(n_pulses));
    if (pulses_per_layer) {
      for (auto& layer_pulses : pulses) {
        for (auto& pulse : layer_pulses) {
          IHM_CHECK(absl::SimpleAtod(vals[pos++], &pulse));
        }
      }
    } else {
      // If the saved file originally had one set of pulses for whole network,
      // copy it at each layer. The loaded weights already work in this set-up.
      for (auto& pulse : pulses.front()) {
        IHM_CHECK(absl::SimpleAtod(vals[pos++], &pulse));
      }
      for (int layer = 1; layer < n_layers; ++layer) {
        pulses[layer] = pulses.front();  // copy
      }
    }
    IHM_LOG(LogSeverity::INFO,
            absl::StrFormat("Loaded pulses: %s", PPrintVectorXXd(pulses)));

    // Read weights.
    weights.resize(n_layers - 1);
    for (int layer = 0; layer < n_layers - 1; ++layer) {
      weights[layer].resize(layer_sizes[layer + 1]);
      for (int j = 0; j < weights[layer].size(); ++j) {
        weights[layer][j].resize(layer_sizes[layer] + n_pulses);
        for (int i = 0; i < weights[layer][j].size(); ++i) {
          IHM_CHECK(absl::SimpleAtod(vals[pos++], &weights[layer][j][i]));
        }
      }
    }
  }
  return absl::make_unique<Tempcoder>(layer_sizes, pulses, fire_threshold,
                                      decay_params, latency, weights);
}

void Tempcoder::ComputeGradients(const VectorXd& final_outputs,
                                 const VectorXXd& activations,
                                 const VectorXd& targets,
                                 const VectorXXXd& dspike_dw,
                                 const VectorXXXd& dspike_dt,
                                 VectorXXXd* d_weights_local,
                                 VectorXXd* d_pulses_local) {
  // Store the derivative of activations of post- wrt pre-layer.
  VectorXd d_activations_post;
  VectorXd d_activations_pre(layer_sizes_.back());

  // Accumulate derivative of cross-entropy loss at each output node.

  // Align outputs and targets to `latency_`.
  // The alternative is aligning them to
  // std::min_element(final_outputs.begin(), final_outputs.end()), which will
  // require an extra derivative. Another alternative is an extra reference
  // neuron.
  double latency = use_mse_ ? latency_ : 0.0;
  // Invert signs for MSE versus cross-entropy, since in the latter we are
  // minimizing (not maximizing) spike times.
  double mse_mult = use_mse_ ? -1.0 : 1.0;

  for (int k = 0; k < layer_sizes_.back(); ++k) {
    double target = targets[k] == kNoSpike ? 1.0 : targets[k];
    d_activations_pre[k] =
        mse_mult * -(final_outputs[k] - target - latency +
                     penalty_output_spike_time_ * (final_outputs[k] - latency));
  }

  // Initialise d_weights_local.
  for (int layer = 0; layer < layer_sizes_.size() - 1; ++layer) {
    (*d_weights_local)[layer].assign(
        layer_sizes_[layer + 1],
        VectorXd(layer_sizes_[layer] + n_pulses_, 0.0));
  }

  // Backpropagate errors.
  for (int layer = layer_sizes_.size() - 2; layer >= 0; --layer) {
    d_activations_post = d_activations_pre;
    d_activations_pre.assign(layer_sizes_[layer] + n_pulses_, 0.0);
    // Update weights.
    for (int k = 0; k < layer_sizes_[layer + 1]; ++k) {
      for (int j = 0; j < layer_sizes_[layer] + n_pulses_; ++j) {
        // Update weight derivative between current layers.
        if (activations[layer + 1][k] == kNoSpike) {
          // no spike, negative grad
          (*d_weights_local)[layer][k][j] += -penalty_no_spike_;
        } else {
          double derivative =
              d_activations_post[k] *
              ClipDerivative(dspike_dw[layer][k][j], clip_derivative_);
          (*d_weights_local)[layer][k][j] +=
              ClipDerivative(derivative, clip_gradient_);
        }
        // Update activation derivative in presynaptic wrt postsynaptic layer.
        double derivative =
            d_activations_post[k] *
            ClipDerivative(dspike_dt[layer][k][j], clip_derivative_);
        d_activations_pre[j] += ClipDerivative(derivative, clip_gradient_);
      }
    }
    // Update pulses.
    for (int p = 0; p < n_pulses_; ++p) {
      (*d_pulses_local)[layer][p] = d_activations_pre[layer_sizes_[layer] + p];
    }
  }
}

void Tempcoder::AddJittering(const JitteringParams& params,
                             VectorXd* activations) {
  if (params.sigma == 0.0 && params.new_spike_probability == 0.0) {
    return;
  }

  // Compute the maximum time of a spike in this layer.
  // Initialized to a negative value (which is invalid as a spike time), as we
  // do not know if any activation time is not kNoSpike.
  double max_original_time = -kNoSpike;
  for (double spike : *activations) {
    if (spike != kNoSpike && spike > max_original_time) {
      max_original_time = spike;
    }
  }

  // If no neuron in this layer spiked, max_original_time will still be
  // negative: nothing cannot be done in this case.
  if (max_original_time < 0.0) return;

  std::default_random_engine generator(std::random_device{}());
  std::normal_distribution<double> dist(params.mean, params.sigma);
  std::bernoulli_distribution bernoulli(params.new_spike_probability);

  for (double& spike : *activations) {
    if (spike == kNoSpike) {
      if (bernoulli(generator)) {
        // Create a spike with the maximum seen spike time plus some random
        // noise.
        spike = max_original_time + dist(generator);
      }
      continue;
    }
    spike += dist(generator);
    // Avoid negative time.
    if (spike < 0.0) {
      spike = 0.0;
    }
  }
}

namespace {
BasicStats<double> MakeStats(const VectorXd& values) {
  VectorXd values_no_kNoSpike;
  std::remove_copy(values.cbegin(), values.cend(),
                   std::back_inserter(values_no_kNoSpike), Tempcoder::kNoSpike);
  return BasicStats<double>(values_no_kNoSpike);
}
}  // namespace

std::vector<size_t> GetSortedIndices(const VectorXd& activations) {
  std::vector<size_t> sorted_indices(activations.size());
  iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&activations](size_t i, size_t j) {
              return activations[i] < activations[j];
            });
  return sorted_indices;
}

VectorXd ExponentiateSortedValidSpikes(
    const VectorXd& activations, const std::vector<size_t>& sorted_indices,
    const double decay_rate) {
  VectorXd exp_activations(activations.size(), Tempcoder::kNoSpike);
  for (int i = 0; i < sorted_indices.size() &&
                  activations[sorted_indices[i]] < Tempcoder::kNoSpike;
       ++i) {
    exp_activations[sorted_indices[i]] =
        exp(decay_rate * activations[sorted_indices[i]]);
  }
  return exp_activations;
}

const Prediction Tempcoder::FeedforwardAndOptionallyComputeGradients(
    const VectorXd& inputs, const VectorXd& targets, int original_class,
    absl::Mutex* mutex, const TempcoderOptions& tempcoder_options) {
  IHM_CHECK(inputs.size() == layer_sizes_.front());
  IHM_CHECK(targets.size() == layer_sizes_.back());

  Prediction prediction;

  // Output at each layer, representing the time of each neuron's (first and
  // only) spike. Has the form activations[layer][node].
  VectorXXd activations;

  // Clear all activations and causal sets.
  activations.resize(layer_sizes_.size());

  // Stores derivatives of spike times with respect to input weights and
  // activations, if required by tempcoder_options.compute_gradients.
  // Same shapes as weights.
  VectorXXXd dspike_dw;
  VectorXXXd dspike_dt;

  if (tempcoder_options.compute_gradients) {
    dspike_dw.resize(layer_sizes_.size() - 1);
    for (int layer = 0; layer < layer_sizes_.size() - 1; ++layer) {
      dspike_dw[layer].resize(layer_sizes_[layer + 1]);
      for (int neuron = 0; neuron < dspike_dw[layer].size(); neuron++) {
        dspike_dw[layer][neuron].resize(layer_sizes_[layer] + n_pulses_);
      }
    }
    dspike_dt = dspike_dw;
  }

  // Activate neurons in all layers.
  activations.front() = inputs;
  // Only for training: add noise to all input spikes except pulses.
  if (tempcoder_options.compute_gradients) {
    AddJittering(input_jittering_params_, &activations.front());
  }
  activations.front().insert(activations.front().end(), pulses_.front().begin(),
                             pulses_.front().end());

  for (int layer = 0; layer < layer_sizes_.size() - 1; ++layer) {
    // Get sorted indices of spike times in the last activation layer.
    std::vector<size_t> sorted_indices = GetSortedIndices(activations[layer]);
    // Precompute exp(activations).
    VectorXd exp_activations = ExponentiateSortedValidSpikes(
        activations[layer], sorted_indices, decay_params_.rate());
    activations[layer + 1].resize(layer_sizes_[layer + 1]);
    for (int n = 0; n < layer_sizes_[layer + 1]; ++n) {
      double* dspike_dw_neuron = tempcoder_options.compute_gradients
                                     ? &dspike_dw[layer][n][0]
                                     : nullptr;
      double* dspike_dt_neuron = tempcoder_options.compute_gradients
                                     ? &dspike_dt[layer][n][0]
                                     : nullptr;
      if (use_dual_exponential_) {
        activations[layer + 1][n] = ActivateNeuronDualExponential(
            weights_[layer][n], activations[layer], exp_activations,
            sorted_indices, fire_threshold_, decay_params_, dspike_dw_neuron,
            dspike_dt_neuron);
      } else {
        activations[layer + 1][n] = ActivateNeuronAlpha(
            weights_[layer][n], activations[layer], exp_activations,
            sorted_indices, fire_threshold_, decay_params_, dspike_dw_neuron,
            dspike_dt_neuron);
      }
    }
    // Only for training: add noise to all non-input spikes except pulses.
    if (tempcoder_options.compute_gradients) {
      AddJittering(noninput_jittering_params_, &activations[layer + 1]);
    }

    if (layer < layer_sizes_.size() - 2) {  // add sync pulses
      activations[layer + 1].insert(activations[layer + 1].end(),
                                    pulses_[layer + 1].begin(),
                                    pulses_[layer + 1].end());
    }
  }

  // Stores either the softmaxed outputs for cross-entropy loss (default), or
  // the last outputs (copying them from `activations`) for MSE.
  // This is because the two cases then allow the same derivative computation.
  VectorXd final_outputs(layer_sizes_.back());

  if (use_mse_) {
    final_outputs = activations.back();
  } else {
    // Apply softmax to outputs (subtract max elem for numerical stability).
    double min_output =
        *min_element(activations.back().begin(), activations.back().end());

    // Very nonstandard and potentially confusing fragment. Since we want, for
    // biological plausibility, faster spikes to indicate more probability, the
    // inputs to the softmax below are inverted, and this includes derivatives.
    for (int i = 0; i < layer_sizes_.back(); ++i)
      final_outputs[i] = exp(-activations.back()[i] + min_output);
    double exp_sum =
        accumulate(final_outputs.begin(), final_outputs.end(), 0.0);
    for (int i = 0; i < final_outputs.size(); ++i) final_outputs[i] /= exp_sum;
  }

  prediction.outputs = use_mse_ ? final_outputs : activations.back();
  prediction.is_correct =
      use_mse_ ? true : IsCorrectSpikingOutput(activations.back(), targets);
  prediction.error =
      ComputeLossWithPenalty(final_outputs, targets, activations.back());

  prediction.first_output_spike_time =
      *min_element(activations.back().begin(), activations.back().end());

  if (tempcoder_options.compute_spike_stats) {
    prediction.spike_stats_per_layer.reserve(layer_sizes_.size());
    for (int layer = 0; layer < activations.size(); ++layer) {
      const VectorXd layer_spikes(
          activations[layer].begin(),
          activations[layer].begin() + layer_sizes_[layer]);
      prediction.spike_stats_per_layer.push_back(MakeStats(layer_spikes));
    }
  }

  if (tempcoder_options.print_activations) PrintActivations(activations);
  if (tempcoder_options.print_trace) PrintTrace(activations);
  if (tempcoder_options.draw_trace) DrawTrace(activations);
  if (!tempcoder_options.compute_gradients &&
      tempcoder_options.log_embeddings) {
    IHM_CHECK(activations.size() == 3);
    mutex->Lock();
    embeddings_log_ +=
        absl::StrCat("\n",
                     absl::StrJoin(activations[1].begin(),
                                   activations[1].end() - n_pulses_, " "),
                     " ", original_class);
    mutex->Unlock();
  }

  // Return if no need to compute updates.
  if (prediction.is_correct && !tempcoder_options.update_all_datapoints) {
    return prediction;
  }
  if (!(tempcoder_options.compute_gradients)) return prediction;

  VectorXXXd* d_weights_local;
  VectorXXd* d_pulses_local;
  {
    absl::MutexLock lock(mutex);
    d_weights_per_call_.emplace_back(
        absl::make_unique<VectorXXXd>(layer_sizes_.size() - 1));
    d_pulses_per_call_.emplace_back(absl::make_unique<VectorXXd>(
        layer_sizes_.size(), VectorXd(n_pulses_, 0.0)));
    d_weights_local = d_weights_per_call_.back().get();
    d_pulses_local = d_pulses_per_call_.back().get();
  }

  ComputeGradients(final_outputs, activations, targets, dspike_dw, dspike_dt,
                   d_weights_local, d_pulses_local);

  return prediction;
}

double Tempcoder::ComputeLossWithPenalty(const VectorXd& outputs,
                                         const VectorXd& targets,
                                         const VectorXd& spike_times) {
  IHM_CHECK(targets.size() == layer_sizes_.back());
  double total_loss = 0.0;
  if (use_mse_) {
    for (int i = 0; i < targets.size(); ++i) {
      double target = targets[i] == kNoSpike ? 1.0 : targets[i];
      double output = outputs[i] == kNoSpike ? 1.0 : outputs[i] - latency_;
      total_loss += (output - target) * (output - target) / targets.size();
      // we might want to change this function and discard this...
      total_loss += penalty_output_spike_time_ * output * output;
    }
  } else {
    const double kEps = 1e-8;
    for (int i = 0; i < targets.size(); ++i) {
      total_loss -= targets[i] * log(outputs[i] + kEps) / targets.size();
    }

    for (int i = 0; i < spike_times.size(); ++i) {
      total_loss +=  // ...and this
          penalty_output_spike_time_ * spike_times[i] * spike_times[i];
    }
  }
  return total_loss;
}

std::vector<Tempcoder::ParameterInfo> Tempcoder::ParameterIndices() const {
  std::vector<ParameterInfo> parameters;
  for (int layer = 0; layer < layer_sizes_.size() - 1; ++layer) {
    for (int k = 0; k < layer_sizes_[layer + 1]; ++k) {
      parameters.push_back(
          ParameterInfo{ParameterInfo::kWeightVector, layer, k});
    }
    parameters.push_back(
        ParameterInfo{ParameterInfo::kPulseVector, layer, /*unused*/ 0});
  }
  return parameters;
}

void Tempcoder::AccumulateSingleDerivativeVector(
    const ParameterInfo& param_info, VectorXd* out) const {
  int layer = param_info.layer;
  switch (param_info.type) {
    case ParameterInfo::kWeightVector: {
      int post = param_info.post_neuron;
      std::fill(out->begin(), out->end(), 0.0);
      for (int call = 0; call < d_weights_per_call_.size(); call++) {
        const std::unique_ptr<VectorXXXd>& d_weights_call =
            d_weights_per_call_[call];
        IHM_CHECK(out->size() >= d_weights_call->size());
        for (int pre = 0; pre < (*d_weights_call)[layer][post].size(); pre++) {
          (*out)[pre] += (*d_weights_call)[layer][post][pre];
        }
      }
      break;
    }
    case ParameterInfo::kPulseVector: {
      std::fill(out->begin(), out->end(), 0.0);
      for (int call = 0; call < d_pulses_per_call_.size(); call++) {
        const std::unique_ptr<VectorXXd>& d_pulses_call =
            d_pulses_per_call_[call];
        IHM_CHECK(out->size() >= d_pulses_call->size());
        for (int pulse = 0; pulse < (*d_pulses_call)[layer].size(); pulse++) {
          (*out)[pulse] += (*d_pulses_call)[layer][pulse];
        }
      }
      break;
    }
  }
}

void Tempcoder::AccumulateAndApplyUpdates(ThreadPool* pool) {
  // Compute the number of updates. This will be equal to batch_size for MSE
  // tasks, and when `update_all_datapoints` is set to true.
  int num_updates = d_weights_per_call_.size();
  // Nothing to do.
  if (num_updates == 0) return;

  // Constants for Adam, computed unconditionally.

  // Factor used to scale updates by the number of examples. This is used below
  // to ensure that an individual example contributes to the updates of the
  // model at the same scale, regardless of the actual number of examples in the
  // batch contributing to updates. This also means that an individual example
  // that happens to be batched with a lower number of other wrong examples has
  // a larger relative contribution to the updates per epoch, in particular to
  // the Adam momentum and variance.
  const double num_updates_inverse_ = 1.0 / num_updates;
  const int effective_adam_epoch = adam_epoch_ + 1;

  // Momentum coefficient, higher will stabilize learning, but can get worse
  // minima.
  const double kB1 = 0.9;

  // EMA constant for the uncentered variance estimate. Higher will likely
  // reduce effective learning rates, lower should approach the behavior of
  // momentum SGD.
  const double kB2 = 0.999;
  const double kEps = 1e-8;

  const double m_scale = 1.0 / (1.0 - IPow(kB1, effective_adam_epoch));
  const double v_scale = 1.0 / (1.0 - IPow(kB2, effective_adam_epoch));
  // We can now update the Adam epoch, as it will not get used in the rest of
  // this function.
  adam_epoch_ += use_adam_;

  // Functions to compute and apply updates, using either Adam or SGD.
  auto compute_and_apply_adam_update =
      [num_updates_inverse_, kB1, kB2, kEps, m_scale, v_scale](
          const VectorXd& deriv, double learning_rate, VectorXd& momentum,
          VectorXd& var, VectorXd& values) {
        for (int i = 0; i < values.size(); i++) {
          const double d_scaled = deriv[i] * num_updates_inverse_;
          momentum[i] = kB1 * momentum[i] + (1.0 - kB1) * d_scaled;
          var[i] = kB2 * var[i] + (1.0 - kB2) * d_scaled * d_scaled;
          values[i] -= learning_rate * momentum[i] * m_scale /
                       std::sqrt(var[i] * v_scale + kEps);
        }
      };
  auto compute_and_apply_sgd_update =
      [num_updates_inverse_](const VectorXd& deriv, double learning_rate,
                             VectorXd& values) {
        for (int i = 0; i < values.size(); i++) {
          values[i] -= learning_rate * num_updates_inverse_ * deriv[i];
        }
      };

  // Allocate storage for storing accumulated derivatives.
  int max_parameter_number =
      *std::max_element(layer_sizes_.begin(), layer_sizes_.end()) + n_pulses_;
  VectorXXd derivatives(NumThreads(pool), VectorXd(max_parameter_number));

  // Update all parameters.
  auto parameters = ParameterIndices();
  RunOnPool(
      pool, 0, parameters.size(),
      [this, &compute_and_apply_adam_update, &compute_and_apply_sgd_update,
       &parameters, &derivatives](int i, int thread) {
        const auto& param_info = parameters[i];
        int layer = param_info.layer;
        switch (param_info.type) {
          case ParameterInfo::kWeightVector: {
            int post = param_info.post_neuron;
            AccumulateSingleDerivativeVector(param_info, &derivatives[thread]);
            if (use_adam_) {
              compute_and_apply_adam_update(derivatives[thread], learning_rate_,
                                            adam_momentum_weights_[layer][post],
                                            adam_var_weights_[layer][post],
                                            weights_[layer][post]);
            } else {
              compute_and_apply_sgd_update(derivatives[thread], learning_rate_,
                                           weights_[layer][post]);
            }
            break;
          }
          case ParameterInfo::kPulseVector: {
            AccumulateSingleDerivativeVector(param_info, &derivatives[thread]);
            if (use_adam_) {
              compute_and_apply_adam_update(
                  derivatives[thread], learning_rate_pulses_,
                  adam_momentum_pulses_[layer], adam_var_pulses_[layer],
                  pulses_[layer]);
            } else {
              compute_and_apply_sgd_update(
                  derivatives[thread], learning_rate_pulses_, pulses_[layer]);
            }
            for (int pulse = 0; pulse < pulses_[layer].size(); pulse++) {
              // Avoid negative time.
              pulses_[layer][pulse] = std::max(0.0, pulses_[layer][pulse]);
            }
            break;
          }
        }
      });
  ClearBatch();
}

void Tempcoder::InitAdam() {
  adam_momentum_weights_.resize(layer_sizes_.size() - 1);
  adam_var_weights_.resize(layer_sizes_.size() - 1);

  adam_momentum_pulses_.assign(pulses_.size(), VectorXd(n_pulses_, 0.0));
  adam_var_pulses_.assign(pulses_.size(), VectorXd(n_pulses_, 0.0));

  for (int layer = 0; layer < layer_sizes_.size() - 1; ++layer) {
    adam_momentum_weights_[layer].assign(
        layer_sizes_[layer + 1],
        VectorXd(layer_sizes_[layer] + n_pulses_, 0.0));

    adam_var_weights_[layer].assign(
        layer_sizes_[layer + 1],
        VectorXd(layer_sizes_[layer] + n_pulses_, 0.0));
  }
  adam_epoch_ = 0;
}

void Tempcoder::ClearBatch() {
  d_weights_per_call_.clear();
  d_pulses_per_call_.clear();
}

VectorXd Tempcoder::CheckGradient(const VectorXd& inputs,
                                  const VectorXd& targets) {
  double eps = 1e-5;
  VectorXd estimated_grads;
  Tempcoder tempcoder_copy(*this);

  for (int layer = 0; layer < layer_sizes_.size() - 1; ++layer) {
    for (int k = 0; k < layer_sizes_[layer + 1]; ++k) {
      for (int j = 0; j < layer_sizes_[layer] + n_pulses_; ++j) {
        // Vary individual weight (+eps) and compute loss.
        tempcoder_copy.weights_[layer][k][j] += eps;
        const double ce_plus =
            tempcoder_copy
                .FeedforwardAndOptionallyComputeGradients(inputs, targets)
                .error;

        // Vary individual weight (-eps) and compute loss.
        tempcoder_copy.weights_[layer][k][j] -= 2 * eps;
        const double ce_minus =
            tempcoder_copy
                .FeedforwardAndOptionallyComputeGradients(inputs, targets)
                .error;

        estimated_grads.push_back((ce_plus - ce_minus) / (2 * eps));

        // Restore copy.
        tempcoder_copy.weights_[layer][k][j] += eps;
      }
    }
  }
  for (int layer = 0; layer < layer_sizes_.size() - 1; ++layer) {
    for (int p = 0; p < n_pulses_; ++p) {
      // Vary pulse (+eps).
      tempcoder_copy.pulses_[layer][p] += eps;
      const double ce_plus =
          tempcoder_copy
              .FeedforwardAndOptionallyComputeGradients(inputs, targets)
              .error;

      // Vary pulse (-eps).
      tempcoder_copy.pulses_[layer][p] -= 2 * eps;
      const double ce_minus =
          tempcoder_copy
              .FeedforwardAndOptionallyComputeGradients(inputs, targets)
              .error;

      estimated_grads.push_back((ce_plus - ce_minus) / (2 * eps));

      // Restore copy.
      tempcoder_copy.pulses_[layer][p] += eps;
    }
  }
  return estimated_grads;
}

void Tempcoder::PrintNetwork() const {
  std::cout << "T = " << fire_threshold() << "\n";
  std::cout << "weights = [";
  for (int layer = 0; layer < layer_sizes_.size() - 1; ++layer) {
    std::cout << "[";
    for (int j = 0; j < layer_sizes_[layer + 1]; ++j) {
      if (j > 0) std::cout << ",\n";
      std::cout << VecToString(weights_[layer][j]);
    }
    std::cout << "]";
    if (layer < layer_sizes_.size() - 2) std::cout << ",\n";
  }
  std::cout << "]" << std::endl;
  std::cout << "pulses = " << PPrintVectorXXd(pulses_) << std::endl;
}

BasicStats<double> Tempcoder::MakeNonPulseWeightStats(int layer) const {
  VectorXd flattened_nonpulse_weights;
  for (const auto& postsynaptic_weights : weights_[layer]) {
    flattened_nonpulse_weights.insert(flattened_nonpulse_weights.end(),
                                      postsynaptic_weights.begin(),
                                      postsynaptic_weights.end() - n_pulses_);
  }
  return BasicStats<double>(flattened_nonpulse_weights);
}

BasicStats<double> Tempcoder::MakePulseWeightStats(int layer) const {
  VectorXd flattened_pulse_weights;
  for (const auto& postsynaptic_weights : weights_[layer]) {
    flattened_pulse_weights.insert(flattened_pulse_weights.end(),
                                   postsynaptic_weights.end() - n_pulses_,
                                   postsynaptic_weights.end());
  }
  return BasicStats<double>(flattened_pulse_weights);
}

void Tempcoder::PrintNetworkStats() const {
  for (int layer = 0; layer < weights_.size(); ++layer) {
    PrintLayerStats(layer, "weights", MakeNonPulseWeightStats(layer));
  }
  for (int layer = 0; layer < weights_.size(); ++layer) {
    PrintLayerStats(layer, "pulse weights", MakePulseWeightStats(layer));
  }
  std::cout << "Pulses: " << PPrintVectorXXd(pulses_) << std::endl;
}

void Tempcoder::WriteNetworkToFile(const std::string& filename) const {
  // Writing dual-exponential models is not supported yet.
  if (use_dual_exponential_) {
    IHM_LOG(LogSeverity::WARNING,
            "Skip writing network with dual exponential.");
    return;
  }

  std::string s;

  // Write file version. Must not be a number. Currently the only versions are
  // "decay_rate+latency", "pulses_per_layer" or original version,
  // which starts directly with numeric data.
  absl::StrAppend(&s, "decay_rate+latency", " ");

  // Write threshold.
  absl::StrAppend(&s, fire_threshold_, " ");

  // Write decay rate
  absl::StrAppend(&s, decay_params_.rate(), " ");

  // Write latency
  absl::StrAppend(&s, latency_, " ");

  // Write number of layers and layer sizes.
  absl::StrAppend(&s, layer_sizes_.size(), " ");
  for (const auto& layer_size : layer_sizes_)
    absl::StrAppend(&s, layer_size, " ");

  // Write number of pulses and the pulses.
  absl::StrAppend(&s, n_pulses_, " ");
  for (const auto& layer_pulses : pulses_) {
    for (const auto& pulse : layer_pulses) absl::StrAppend(&s, pulse, " ");
  }

  // Write weights.
  for (const auto& layer : weights_)
    for (const auto& post : layer)
      for (const auto& value : post) absl::StrAppend(&s, value, " ");

  file::IhmFile my_file = file::OpenOrDie(filename, "w");
  IHM_CHECK(my_file.WriteString(s));
  IHM_CHECK(my_file.Close());
  IHM_LOG(LogSeverity::INFO, absl::StrFormat("Wrote network to: %s", filename));
}

void Tempcoder::WriteEmbeddingsToFile(const std::string& filename) const {
  file::IhmFile my_file = file::OpenOrDie(filename, "w");
  IHM_CHECK(my_file.WriteString(embeddings_log_));
  IHM_CHECK(my_file.Close());
  IHM_LOG(LogSeverity::INFO,
          absl::StrFormat("Wrote embeddings to: %s", filename));
}

void Tempcoder::PrintActivations(const VectorXXd& activations) const {
  for (int layer = 0; layer < layer_sizes_.size(); ++layer)
    std::cout << "Activations layer#" << layer << ": "
              << VecToString(activations[layer]) << std::endl;
  std::cout << std::endl;
}

void Tempcoder::PrintTrace(const VectorXXd& activations) const {
  if (activations.empty()) {
    IHM_LOG(LogSeverity::WARNING, "Network has not been active.");
    return;
  }
  // Presynaptic spikes ordered by timing.
  std::priority_queue<std::pair<std::string, double>,
                      std::vector<std::pair<std::string, double>>,
                      CompareNeuronTiming<std::string>>
      events;

  for (int layer = 0; layer < layer_sizes_.size(); ++layer)
    for (int k = 0; k < layer_sizes_[layer]; ++k)
      events.push(std::make_pair(absl::StrCat("layer#", layer, "_", k),
                                 activations[layer][k]));

  std::vector<std::string> trace;
  while (!events.empty()) {
    trace.push_back(
        absl::StrFormat("[%s@%.2f]", events.top().first, events.top().second));
    events.pop();
  }
  std::cout << "Trace: " << absl::StrJoin(trace, " ") << std::endl;
}

void Tempcoder::DrawNetwork() const {
  int node_height =
      *std::max_element(layer_sizes_.begin(), layer_sizes_.end()) + n_pulses_;
  std::vector<std::string> s;
  s.push_back("  [weight]\n");
  for (int i = 0; i < node_height; ++i) {
    for (int j = 0; j < node_height + 1 - n_pulses_; ++j) {
      for (int layer = 0; layer < layer_sizes_.size() - 1; ++layer) {
        if (j == 0 && i < weights_[layer][j].size())
          s.push_back("-@");
        else
          s.push_back("  ");

        if (layer < layer_sizes_.size() - 1) {
          if (j < weights_[layer].size() && i < weights_[layer][j].size())
            s.push_back(absl::StrFormat("[%+2.2f]    ", weights_[layer][j][i]));
          else
            s.push_back("           ");
        }
      }
      s.push_back("\n");
    }
  }
  std::cout << "\n" << absl::StrJoin(s, "") << std::endl;
}

void Tempcoder::DrawTrace(const VectorXXd& activations) const {
  if (activations.empty()) {
    IHM_LOG(LogSeverity::WARNING, "Network has not been active.");
    return;
  }
  int node_height =
      *std::max_element(layer_sizes_.begin(), layer_sizes_.end()) + n_pulses_;
  std::vector<std::string> s;
  s.push_back("  [activation time]\n");
  for (int i = 0; i < node_height; ++i) {
    for (int layer = 0; layer < layer_sizes_.size(); ++layer) {
      if (i < activations[layer].size())
        s.push_back("-@");
      else
        s.push_back("  ");
      if (i < activations[layer].size() && activations[layer][i] < kNoSpike)
        s.push_back(absl::StrFormat("[%+2.2f]    ", activations[layer][i]));
      else if (i < activations[layer].size())
        s.push_back("[large]    ");
      else
        s.push_back("           ");
    }
    s.push_back("\n");
  }
  std::cout << "\n" << absl::StrJoin(s, "") << std::endl;
}

VectorXd Tempcoder::AdjustInput(const VectorXd& inputs, const int target_layer,
                                const int target_neuron,
                                const double adjustment_strength,
                                const bool maximize_nontargets) {
  IHM_CHECK(inputs.size() == layer_sizes_.front());
  IHM_CHECK(target_layer > 0);
  IHM_CHECK(target_layer < layer_sizes_.size());
  IHM_CHECK(target_neuron >= 0);
  IHM_CHECK(target_neuron < layer_sizes_[target_layer]);

  // Initialisations. See FeedforwardAndOptionallyComputeGradients() for
  // explanations for each variable.
  VectorXXd activations;
  activations.resize(target_layer + 1);
  VectorXXXd dspike_dw;
  VectorXXXd dspike_dt;

  dspike_dw.resize(target_layer);
  for (int layer = 0; layer < target_layer; ++layer) {
    dspike_dw[layer].resize(layer_sizes_[layer + 1]);
    for (int neuron = 0; neuron < dspike_dw[layer].size(); neuron++) {
      dspike_dw[layer][neuron].resize(layer_sizes_[layer] + n_pulses_);
    }
  }
  dspike_dt = dspike_dw;

  // Feedfoward pass.
  activations.front() = inputs;
  activations.front().insert(activations.front().end(), pulses_.front().begin(),
                             pulses_.front().end());
  for (int layer = 0; layer < target_layer; ++layer) {
    std::vector<size_t> sorted_indices = GetSortedIndices(activations[layer]);
    VectorXd exp_activations = ExponentiateSortedValidSpikes(
        activations[layer], sorted_indices, decay_params_.rate());
    activations[layer + 1].resize(layer_sizes_[layer + 1]);
    for (int n = 0; n < layer_sizes_[layer + 1]; ++n) {
      double* dspike_dw_neuron = dspike_dw[layer][n].data();
      double* dspike_dt_neuron = dspike_dt[layer][n].data();
      if (use_dual_exponential_) {
        activations[layer + 1][n] = ActivateNeuronDualExponential(
            weights_[layer][n], activations[layer], exp_activations,
            sorted_indices, fire_threshold_, decay_params_, dspike_dw_neuron,
            dspike_dt_neuron);
      } else {
        activations[layer + 1][n] = ActivateNeuronAlpha(
            weights_[layer][n], activations[layer], exp_activations,
            sorted_indices, fire_threshold_, decay_params_, dspike_dw_neuron,
            dspike_dt_neuron);
      }
    }
    if (layer < target_layer - 1) {  // append sync pulses
      activations[layer + 1].insert(activations[layer + 1].end(),
                                    pulses_[layer + 1].begin(),
                                    pulses_[layer + 1].end());
    }
  }

  // Backpropagate errors.
  VectorXd d_activations_post;  // derivative of postsynaptic spike time
  VectorXd d_activations_pre;   // wrt presynaptic spike time
  d_activations_pre.resize(layer_sizes_[target_layer - 1]);
  for (int post = 0; post < layer_sizes_[target_layer]; ++post) {
    for (int pre = 0; pre < layer_sizes_[target_layer - 1]; ++pre) {
      if (post == target_neuron) {  // minimize target spike time
        d_activations_pre[pre] += dspike_dt[target_layer - 1][post][pre];
      } else if (maximize_nontargets) {  // maximize non-targets spike time
        d_activations_pre[pre] -= dspike_dt[target_layer - 1][post][pre];
      }
    }
  }
  for (int layer = target_layer - 2; layer >= 0; --layer) {
    d_activations_post = d_activations_pre;
    d_activations_pre.assign(layer_sizes_[layer], 0.0);
    for (int post = 0; post < layer_sizes_[layer + 1]; ++post) {
      for (int pre = 0; pre < layer_sizes_[layer]; ++pre) {
        d_activations_pre[pre] +=
            d_activations_post[post] * dspike_dt[layer][post][pre];
      }
    }
  }

  // Adjust inputs.
  VectorXd adjusted_inputs = inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    adjusted_inputs[i] -= d_activations_pre[i] * adjustment_strength;
  }
  return adjusted_inputs;
}

}  // namespace ihmehimmeli
