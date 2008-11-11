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

namespace Torch {

SmoothedLinear::SmoothedLinear(int n_inputs_, int n_outputs_) : Linear(n_inputs_, n_outputs_)
{
  addROption("smoothing weight decay", &smoothing_weight_decay, 0., "smoothing weight decay");
}

void SmoothedLinear::frameBackward(int t, real *f_inputs, real *beta_, real *f_outputs, real *alpha_)
{
  Linear::frameBackward(t, f_inputs, beta_, f_outputs, alpha_);

  // Apply smoothing weight decay.
  if (smoothing_weight_decay != 0.)  {
    real *src_ = params->data[0];
    real *dest_ = der_params->data[0];

    int n_weights = n_inputs*n_outputs;

    for(int i=0; i<n_weights; i++) {
      // not considering the equal cases...
      // from left 
      if ((i%n_inputs) != 0) { // not on W(:,0)
        if(src_[i]<src_[i-1])
          dest_[i] += smoothing_weight_decay;
        else
          dest_[i] -= smoothing_weight_decay;
      }
      // from right
      if ((i%n_inputs) != (n_inputs-1)) { // not on W(:,n_inputs-1)
        if(src_[i]<src_[i+1])
          dest_[i] += smoothing_weight_decay;
        else
          dest_[i] -= smoothing_weight_decay;
      }

      // from top
      if (i >= n_inputs) { // not on W(0,:)
        if(src_[i]<src_[i-n_inputs])
          dest_[i] += smoothing_weight_decay;
        else
          dest_[i] -= smoothing_weight_decay;
      }

      // from bottom
      if (i<n_weights-n_inputs) { // not on W(,:)
        if(src_[i]<src_[i+n_inputs])
          dest_[i] += smoothing_weight_decay;
        else
          dest_[i] -= smoothing_weight_decay;
      }

    }
  }

}

SmoothedLinear::~SmoothedLinear()
{
}

}
