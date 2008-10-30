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
#include "binner.h"

#include <algorithm>
#include <cassert>
#include "Random.h"


namespace Torch {

Binner::Binner(int n_bins_)
{
  n_bins = n_bins_;
}

void Binner::init(int n_samples, real *samples)
{
  // Start by sorting the numbers. This alters samples!
  std::sort(samples, samples+n_samples);

  // Get the range of the data and the bin width
  real range_lower = samples[0];
  real range_upper = samples[n_samples-1];
  real range_width = range_upper - range_lower;
  real bin_width = range_width / n_bins;

  // Allocate and prepare variables
  bin_n_samples = (int*) allocator->alloc(sizeof(int)*(n_bins));
  bin_cumulative_n_samples = (int*) allocator->alloc(sizeof(int)*(n_bins));
  bin_lowers = (real*) allocator->alloc(sizeof(real)*(n_bins));
  bin_uppers = (real*) allocator->alloc(sizeof(real)*(n_bins));

  for (int i=0; i<n_bins; i++)  {
    bin_n_samples[i] = 0;
    bin_cumulative_n_samples[i] = 0;
    bin_lowers[i] = range_lower + i * bin_width;
    bin_uppers[i] = range_lower + (i+1) * bin_width; 
  }

  // Do the binning.
  int current_bin = 0;
  for (int i=0; i<n_samples; i++) {
    // Find the right bin. For the last sample the test should be false
    // for the last bin as they will be equal...
    while(samples[i]>bin_uppers[current_bin])
      current_bin++;
    assert(current_bin < n_bins);

    bin_n_samples[current_bin]++;
  }

  // Compute the cumulative number of samples
  bin_cumulative_n_samples[0] = bin_n_samples[0];
  for (int i=1; i<n_bins; i++)
    bin_cumulative_n_samples[i] = bin_cumulative_n_samples[i-1] + bin_n_samples[i];

  assert(bin_cumulative_n_samples[n_bins-1] == n_samples);
}

real Binner::draw()
{
  // Draw a bin based on its weight. Draw from uniform over [0,n_samples[
  // and see which bin that falls into.
  real bin_selector = Random::boundedUniform(0.0, bin_cumulative_n_samples[n_bins-1]);
  bin_selector = floor(bin_selector);

  int the_bin = 0;
  while (bin_selector > bin_cumulative_n_samples[the_bin])
    the_bin++;

  assert(the_bin<n_bins);

  // Draw a uniform over that bin's range
  return Random::boundedUniform(bin_lowers[the_bin], bin_uppers[the_bin]);
}

void Binner::loadXFile(XFile *file)
{
}

void Binner::saveXFile(XFile *file)
{
}

Binner::~Binner()
{
}

}
