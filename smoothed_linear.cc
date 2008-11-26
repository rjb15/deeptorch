// Copyright 2008 Google Inc.
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
//
#include "smoothed_linear.h"
#include <cassert>

namespace Torch {

SmoothedLinear::SmoothedLinear(int n_inputs_, int n_outputs_) : Linear(n_inputs_, n_outputs_)
{
  warning("SmoothedLinear - Assuming input is a square image!");
  input_sub_unit_size = (int) sqrt(n_inputs);
  input_n_sub_units = input_sub_unit_size;
  assert( (input_sub_unit_size*input_n_sub_units) == n_inputs);

  addROption("l1 smoothing weight decay", &l1_smoothing_weight_decay, 0., "l1_smoothing weight decay");
  addROption("l2 smoothing weight decay", &l2_smoothing_weight_decay, 0., "l2_smoothing weight decay");
}

void SmoothedLinear::frameBackward(int t, real *f_inputs, real *beta_, real *f_outputs, real *alpha_)
{
  Linear::frameBackward(t, f_inputs, beta_, f_outputs, alpha_);

  // Apply smoothing weight decay.
  if (l1_smoothing_weight_decay != 0. || l2_smoothing_weight_decay != 0.)  {
    real *src_ = params->data[0];
    real *dest_ = der_params->data[0];

    int offset=0;
    real delta=0.0;

    bool is_l1_decayed = false;
    bool is_l2_decayed = false;
    if (l1_smoothing_weight_decay > 0.0)
      is_l1_decayed = true;
    if (l2_smoothing_weight_decay > 0.0)
      is_l2_decayed = true;

    // the gradient will be SUBSTRACTED. We add negative the correction to it.
    for(int i=0; i<n_outputs; i++) {
      for (int j=0; j<input_n_sub_units; j++) {
        for (int k=0; k<input_sub_unit_size; k++) {

          // not considering the equal cases...
          // from left - not when k==0
          if (k) {
            delta = src_[offset] - src_[offset-1];

            if (is_l1_decayed)  {
              if(delta<0.0)
                dest_[offset] -= l1_smoothing_weight_decay;
              else
                dest_[offset] += l1_smoothing_weight_decay;
            }

            if (is_l2_decayed)
              dest_[offset] += l2_smoothing_weight_decay * delta;
          }

          // from right
          if (k != (input_sub_unit_size-1)) { // not on right border
            delta = src_[offset] - src_[offset+1];

            if(is_l1_decayed) {
              if(delta<0.0)
                dest_[offset] -= l1_smoothing_weight_decay;
              else
                dest_[offset] += l1_smoothing_weight_decay;
            }

            if (is_l2_decayed)
              dest_[offset] += l2_smoothing_weight_decay * delta;
          }

          // from top
          if (j) { // not on top border
            delta = src_[offset] - src_[offset-input_sub_unit_size];

            if(is_l1_decayed) {
              if(delta<0.0)
                dest_[offset] -= l1_smoothing_weight_decay;
              else
                dest_[offset] += l1_smoothing_weight_decay;
            }

            if (is_l2_decayed)
              dest_[offset] += l2_smoothing_weight_decay * delta;
          }

          // from bottom
          if (j<input_n_sub_units-1) { // not on bottom border 
            delta = src_[offset] - src_[offset+input_sub_unit_size];

            if (is_l1_decayed) {
              if (delta < 0.0)
                dest_[offset] -= l1_smoothing_weight_decay;
              else
                dest_[offset] += l1_smoothing_weight_decay;
            }

            if (is_l2_decayed)
              dest_[offset] += l2_smoothing_weight_decay * delta;
          }

          //
          offset++;
        }
      }
    } // for n_outputs
  } // if
}

SmoothedLinear::~SmoothedLinear()
{
}

}
