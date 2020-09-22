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

#ifndef IHMEHIMMELI_TEMPCODING_GRADIENT_LANDSCAPE_HELPER_H_
#define IHMEHIMMELI_TEMPCODING_GRADIENT_LANDSCAPE_HELPER_H_

#include "common/data_parallel.h"
#include "tempcoding/spiking_problem.h"
#include "tempcoding/tempcoder.h"

namespace ihmehimmeli {

struct GradientLandscapeOptions {
  bool check_gradient = false;
  bool print_gradient_landscape = false;
  int batch_num = 0;
  double line_points_multiplier = 1.0;
  int custom_num_objective_samples = 0;
  int show_objectives_every_n_batches = 1;
  bool print_spikes_along_gradient = false;

  bool ShouldPrint() {
    return print_gradient_landscape &&
           (batch_num % show_objectives_every_n_batches == 0);
  }
};

void PrintGradientLandscape(
    Tempcoder* tempcoder, SpikingProblem* problem,
    ihmehimmeli::ThreadPool* thread_pool, double total_train_error,
    int train_ind, int train_ind_high_bound, const VectorXXXd& weight_updates,
    const VectorXXd& pulse_updates,
    const GradientLandscapeOptions& gradient_landscape_options);

}  // namespace ihmehimmeli

#endif  // IHMEHIMMELI_TEMPCODING_GRADIENT_LANDSCAPE_HELPER_H_
