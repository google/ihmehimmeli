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

#include "tempcoding/gradient_landscape_helper.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "common/util.h"

namespace ihmehimmeli {

std::tuple<VectorXXXd, VectorXXd> LinearCombinationWeights(
    const VectorXXXd& start_weights, const VectorXXd& start_pulses,
    const std::vector<VectorXXXd*>& weight_deltas,
    const std::vector<VectorXXd*>& pulse_deltas, const VectorXd& factors,
    double batch_scale) {
  VectorXXXd probe_weights(start_weights);
  VectorXXd probe_pulses(start_pulses);

  for (int i = 0; i < factors.size(); ++i) {
    const VectorXXXd& factor_weights = *weight_deltas[i];
    const VectorXXd& factor_pulses = *pulse_deltas[i];

    for (int l = 0; l < probe_weights.size(); ++l) {
      for (int j = 0; j < probe_weights[l].size(); ++j) {
        for (int k = 0; k < probe_weights[l][j].size(); ++k) {
          probe_weights[l][j][k] +=
              factors[i] * factor_weights[l][j][k] * batch_scale;
        }
      }
    }
    for (int layer = 0; layer < probe_pulses.size(); ++layer) {
      for (int k = 0; k < probe_pulses[layer].size(); ++k) {
        probe_pulses[layer][k] +=
            factors[i] * factor_pulses[layer][k] * batch_scale;
      }
    }
  }

  for (int layer = 0; layer < probe_pulses.size(); ++layer) {
    for (int k = 0; k < probe_pulses[layer].size(); ++k) {
      probe_pulses[layer][k] = std::max(0.0, probe_pulses[layer][k]);
    }
  }

  return std::tie(probe_weights, probe_pulses);
}

double ParameterNorm(const VectorXXXd& weights, const VectorXXd& pulses) {
  double norm = 0;

  for (const auto& vv : weights) {
    for (const auto& v : vv) {
      for (double g : v) {
        norm += g * g;
      }
    }
  }

  for (const auto& v : pulses) {
    for (double g : v) norm += g * g;
  }
  return std::sqrt(norm);
}

// Samples isotropically distributed parameter vector of length norm.
// exp(-x[0]**2) * exp(-x[1]**2) * exp(-x[2]**2) * ... = exp(-r**2)
std::tuple<VectorXXXd, VectorXXd> RandomParametersLike(
    const VectorXXXd& weights, const VectorXXd& pulses, double norm) {
  VectorXXXd random_weights(weights);
  VectorXXd random_pulses(pulses);

  std::default_random_engine generator(std::random_device{}());
  std::normal_distribution<double> standard_normal_dist(0.0, 1.0);

  for (int l = 0; l < weights.size(); ++l) {
    for (int j = 0; j < weights[l].size(); ++j) {
      for (int k = 0; k < weights[l][j].size(); ++k) {
        random_weights[l][j][k] = standard_normal_dist(generator);
      }
    }
  }

  for (int layer = 0; layer < pulses.size(); ++layer) {
    for (int k = 0; k < pulses[layer].size(); ++k) {
      random_pulses[layer][k] = standard_normal_dist(generator);
    }
  }

  double random_vec_norm = ParameterNorm(random_weights, random_pulses);

  for (int l = 0; l < random_weights.size(); ++l) {
    for (int j = 0; j < random_weights[l].size(); ++j) {
      for (int k = 0; k < random_weights[l][j].size(); ++k) {
        random_weights[l][j][k] *= norm / random_vec_norm;
      }
    }
  }

  for (int layer = 0; layer < pulses.size(); ++layer) {
    for (int k = 0; k < pulses[layer].size(); ++k) {
      random_pulses[layer][k] *= norm / random_vec_norm;
    }
  }

  return std::tie(random_weights, random_pulses);
}

double ParameterDotProduct(const VectorXXXd& w0, const VectorXXd& p0,
                           const VectorXXXd& w1, const VectorXXd& p1) {
  double dot = 0;
  for (int l = 0; l < w0.size(); ++l) {
    for (int j = 0; j < w0[l].size(); ++j) {
      for (int k = 0; k < w0[l][j].size(); ++k) {
        dot += w0[l][j][k] * w1[l][j][k];
      }
    }
  }
  for (int layer = 0; layer < p0.size(); ++layer) {
    for (int k = 0; k < p0[layer].size(); k++) {
      dot += p0[layer][k] * p1[layer][k];
    }
  }
  return dot;
}

void PrintGradientLandscape(
    Tempcoder* tempcoder, SpikingProblem* problem,
    ihmehimmeli::ThreadPool* thread_pool, double total_train_error,
    int train_ind, int train_ind_high_bound, const VectorXXXd& weight_updates,
    const VectorXXd& pulse_updates,
    const GradientLandscapeOptions& gradient_landscape_options) {
  // This contains negative points as a sanity check on our gradients.
  std::vector<double> kLineSearchPoints(
      {-0.0025, -0.00214286, -0.00178571, -0.00142857, -0.00107143, -0.00071429,
       -0.00035714, 0., 0.00035714, 0.00071429, 0.00107143, 0.00142857,
       0.00178571, 0.00214286, 0.0025});
  for (auto& elem : kLineSearchPoints) {
    elem *= gradient_landscape_options.line_points_multiplier;
  }
  const int kNumLineSearchPoints = kLineSearchPoints.size();
  VectorXd line_search_errors(kNumLineSearchPoints * kNumLineSearchPoints, 0.0);
  std::vector<int> line_search_correct(
      kNumLineSearchPoints * kNumLineSearchPoints, 0);
  VectorXXd line_search_outputs(
      tempcoder->layer_sizes().back(),
      std::vector<double>(kNumLineSearchPoints * kNumLineSearchPoints, 0));

  VectorXXXd start_weights(tempcoder->weights());
  VectorXXd start_pulses(tempcoder->pulses());
  VectorXXXd d_weights(tempcoder->d_weights());
  VectorXXd d_pulses(tempcoder->d_pulses());

  double gradient_norm = ParameterNorm(d_weights, d_pulses);
  VectorXXXd random_weights;
  VectorXXd random_pulses;
  std::tie(random_weights, random_pulses) =
      RandomParametersLike(d_weights, d_pulses, gradient_norm);

  double updates_on_gradient =
      ParameterDotProduct(weight_updates, pulse_updates, d_weights, d_pulses) /
      gradient_norm;

  // There's no bug here as ||random|| = ||gradient||.
  double updates_on_random =
      ParameterDotProduct(weight_updates, pulse_updates, random_weights,
                          random_pulses) /
      gradient_norm;

  double batch_scale = 1.0 / (train_ind_high_bound - train_ind);

  int target_sample_size = train_ind_high_bound - train_ind;
  int custom_num_objective_samples =
      gradient_landscape_options.custom_num_objective_samples;
  if (custom_num_objective_samples > 0) {
    target_sample_size =
        std::min(custom_num_objective_samples,
                 static_cast<int>(problem->train_examples().size()));
  }

  // Index balancing magic. Since 0 < target_sample_size <= |train set|,
  // there is a valid interval of target_sample_size. We prefer expanding
  // fowards (so as to maximize current batch overlap), and will only expand
  // backward when that would be impossible.
  int sample_ind_high_bound =
      std::min(static_cast<int>(problem->train_examples().size()),
               train_ind + target_sample_size);
  int sample_ind_low_bound = train_ind;
  if (sample_ind_high_bound - sample_ind_low_bound < target_sample_size)
    sample_ind_low_bound = sample_ind_high_bound - target_sample_size;

  for (int random_point = 0; random_point < kNumLineSearchPoints;
       ++random_point) {
    for (int line_point = 0; line_point < kNumLineSearchPoints; ++line_point) {
      VectorXXXd probe_weights;
      VectorXXd probe_pulses;

      std::tie(probe_weights, probe_pulses) = LinearCombinationWeights(
          start_weights, start_pulses, {&d_weights, &random_weights},
          {&d_pulses, &random_pulses},
          {kLineSearchPoints[line_point], kLineSearchPoints[random_point]},
          batch_scale);

      tempcoder->set_weights(probe_weights);
      tempcoder->set_pulses(probe_pulses);

      VectorXd probe_errors(problem->train_examples().size());
      std::vector<int> probe_correct(problem->train_examples().size());
      VectorXd probe_outputs(problem->train_examples().size(), 0);
      thread_pool->Run(
          sample_ind_low_bound, sample_ind_high_bound,
          [&tempcoder, &problem, &probe_errors, &probe_correct, &probe_outputs,
           &sample_ind_low_bound](const int i, const int num_thread) {
            const Prediction prediction =
                tempcoder->FeedforwardAndOptionallyAccumulateUpdates(
                    problem->train_examples()[i].inputs,
                    problem->train_examples()[i].targets);
            probe_errors[i] = prediction.error;
            probe_correct[i] = prediction.is_correct;
            // save output spike times for the first example in the batch only
            if (i == sample_ind_low_bound) {
              probe_outputs = prediction.outputs;
            }
          });
      double total_probe_error = 0;
      int total_correct = 0;
      for (int i = sample_ind_low_bound; i < sample_ind_high_bound; ++i) {
        total_probe_error += probe_errors[i];
        total_correct += probe_correct[i];
      }

      int line_search_ind = random_point * kNumLineSearchPoints + line_point;
      line_search_errors[line_search_ind] = total_probe_error;
      line_search_correct[line_search_ind] = total_correct;
      for (int output_ind = 0; output_ind < line_search_outputs.size();
           ++output_ind) {
        line_search_outputs[output_ind][line_search_ind] =
            probe_outputs[output_ind];
      }
    }
  }
  tempcoder->set_weights(start_weights);
  tempcoder->set_pulses(start_pulses);

  std::cout << "Starting point: " << total_train_error << std::endl
            << "Projections: Gradient: " << updates_on_gradient
            << ", Random: " << updates_on_random << std::endl
            << "BB: " << VecToString(kLineSearchPoints) << std::endl
            << "data = [" << VecToString(line_search_errors) << ", "
            << VecToString(line_search_correct) << "]" << std::endl;

  if (gradient_landscape_options.print_spikes_along_gradient) {
    std::cout << "output_spike_times = [";
    for (int output_ind = 0; output_ind < line_search_outputs.size();
         ++output_ind) {
      if (output_ind) std::cout << " ,";
      std::cout << VecToString(line_search_outputs[output_ind]);
    }
    std::cout << "]" << std::endl;
    std::cout
        << "target = "
        << std::distance(
               problem->train_examples()[sample_ind_low_bound].targets.begin(),
               std::max_element(problem->train_examples()[sample_ind_low_bound]
                                    .targets.begin(),
                                problem->train_examples()[sample_ind_low_bound]
                                    .targets.end()))
        << std::endl;
  }
}

}  // namespace ihmehimmeli
