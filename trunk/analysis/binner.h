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
#ifndef TORCH_BINNER_H_
#define TORCH_BINNER_H_

#include <string>
#include "Object.h"
#include "XFile.h"

namespace Torch {

// A Binner represents a distribution from which you can draw.
//
// TODO consider deriving this from Distribution.
//
class Binner : public Object
{
  public:

   int n_bins;
   int *bin_n_samples;
   int *bin_cumulative_n_samples;
   real *bin_lowers;
   real *bin_uppers;


   Binner();

   // Does the binning. Watch out, samples will get ordered!
   virtual void init(int the_n_bins, int n_samples, real *samples);
   virtual real draw();

   virtual void loadXFile(XFile *file);
   virtual void saveXFile(XFile *file);

   virtual ~Binner();

};

}
#endif  // TORCH_CODER_H_
