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
#ifndef TORCH_SMOOTHED_LINEAR_H_
#define TORCH_SMOOTHED_LINEAR_H_

#include "Linear.h"

namespace Torch {

// A modified Linear Layer with a weight decay that tries to maintain
// neighbouring weights of a neuron close. Assumes the input is a square
// (NxN) image.
class SmoothedLinear : public Linear
{
  public:
    int input_sub_unit_size;
    int input_n_sub_units;
    real l1_smoothing_weight_decay;
    real l2_smoothing_weight_decay;

    ///
    SmoothedLinear(int n_inputs_, int n_outputs_);

    //-----
    virtual void frameBackward(int t, real *f_inputs, real *beta_, real *f_outputs, real *alpha_);

    virtual ~SmoothedLinear();
};


}

#endif // TORCH_SMOOTHED_LINEAR_H_
