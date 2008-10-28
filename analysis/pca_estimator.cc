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

#include <cassert>
#include <iostream>
#include "pca_estimator.h"

namespace Torch {

PcaEstimator::PcaEstimator(int n_dim_, int n_eigen_, int minibatch_size_, real gamma_) : Object()
{
  n_dim = n_dim_;
  n_eigen = n_eigen_;
  minibatch_size = minibatch_size_;
  gamma = gamma_;
  lambda = 1e-3;

  n_observations = 0;
  minibatch_index = 0;

  Initialize();
}

void PcaEstimator::Initialize()
{
  Xt = new(allocator) Mat(n_eigen + minibatch_size, n_dim);
  Xt->zero();

  x_sum = new(allocator) Vec(n_dim);
  for (int i=0; i<x_sum->n; i++)
    x_sum->ptr[i] = 0.0;

  G = new(allocator) Mat(n_eigen + minibatch_size, n_eigen + minibatch_size);
  G->zero();
  for (int i=0; i<n_eigen; i++)
    G->ptr[i][i] = lambda;

  d = new(allocator) Vec(n_eigen + minibatch_size);
  for (int i=0; i<n_eigen+minibatch_size; i++)
    d->ptr[i] = 0.0;

  V = new(allocator)  Mat(n_eigen + minibatch_size, n_eigen + minibatch_size);
  V->zero();

  Ut = new(allocator)  Mat(n_eigen, n_dim);
  Ut->zero();
}

void PcaEstimator::Observe(Vec *x)
{
  assert(x->n == n_dim);

  n_observations++;

  // Add the *non-centered* observation to Xt.
  int row = n_eigen + minibatch_index;
  Xt->setRow(row, x);

  // Update the discounted sum of the observations.
  for (int i=0; i<x_sum->n; i++)
    x_sum->ptr[i] = x_sum->ptr[i] * gamma + x->ptr[i];

  // To get the mean, we must normalize the sum by:
  // /gamma^(n_observations-1) + /gamma^(n_observations-2) + ... + 1
  real normalizer = (1 - pow(gamma, (real)n_observations)) /(1 - gamma);
  real inv_normalizer = 1.0 / normalizer;

  // Now center the observation.
  // We will lose the first observation as it is the only one in the mean.
  Vec new_x(Xt->ptr[row], n_dim);
  for (int i=0; i<new_x.n; i++)
    new_x.ptr[i] -= x_sum->ptr[i] * inv_normalizer;

  // Multiply the centered observation by the discount compensator. Basically
  // we make this observation look "younger" than the previous ones. The actual
  // discount is applied in the reevaluation (and when solving the equations in 
  // the case of TONGA) by multiplying every direction with the same aging factor.
  real rn = pow(gamma, -0.5*(minibatch_index+1));
  for (int i=0; i<n_dim; i++)
    Xt->ptr[row][i] *= rn;

  // Update the Gram Matrix. Xkt represents the currently used portion
  // of Xt (first rows).
  Vec new_g(G->ptr[row], row + 1);
  Mat *Xkt =  Xt->subMat(0,0,row,n_dim-1);
  mxMatMulVec(Xkt, x, &new_g); 
  delete Xkt;

  // Now copy G(row,:) to G(:,row).
  // There are row+1 values, but the diag doesn't need to get copied.
  for (int i=0; i<row; i++)
    G->ptr[i][row] =  G->ptr[row][i];

  minibatch_index++;

  if(minibatch_index==minibatch_size)
    Reevaluate();

}

void PcaEstimator::Reevaluate()
{
  // TODO do the modifications to handle when this is not true.
  assert(minibatch_index==minibatch_size);

  // Regularize - not necessary but in case
  //for (int i=0; i<n_eigen+minibatch_size; i++)
  //  G->ptr[i][i] += lambda;

  // The Gram matrix is up to date. Get its low rank eigendecomposition.
  // TODO How to get only #n_eigen# elements?
  mxSymEig(G, V, d);

  // The eigen values and vectors are *NOT SORTED*. Furthermore, the vectors
  // are on the columns.
  // Find and sort the n_eigen first values and vectors using Dumb Sort
  for (int i=0; i<n_eigen; i++) {
    // Initialize the max search
    int max_index = i;
    real max_value = d->ptr[i];

    // Compare with remaining possibilities
    for (int j=i+1; j<n_eigen+minibatch_size; j++) {
      if (max_value < d->ptr[j])  {
        max_value = d->ptr[j];
        max_index = j;
      }
    }

    // Swap!
    d->ptr[max_index] = d->ptr[i];
    d->ptr[i] = max_value;
    mxSwapColsMat(V, i, max_index, -1, -1);
  }

  // Convert the n_eigen first eigenvectors of the Gram matrix contained in V 
  // into *unnormalized* eigenvectors U of the covariance.
  Mat *Vk = V->subMat(0, 0, n_eigen+minibatch_index-1, n_eigen-1);
  mxTrMatMulMat(Vk, Xt, Ut);
  delete Vk;

  // Take into account the discount factor.
  // Here, minibatch index is minibacth_size. We age everyone. Because of the 
  // previous multiplications to make some observations "younger" we multiply
  // everyone by the same factor.
  // TODO VERIFY THIS!
  real rn = pow(gamma, -0.5*(minibatch_index+1));
  real inv_rn2 = 1.0/(rn*rn);

  mxRealMulMat(1.0/rn, Ut, Ut);
  for (int i=0; i<d->n; i++)
    d->ptr[i] *= inv_rn2;

  //---------------------
  /*
  std::cout << "*** Reevaluate! ***" << std::endl;
  real normalizer = (1.0 - pow(gamma, (real)n_observations)) /(1.0 - gamma);
  real inv_normalizer = 1.0 / normalizer;
  std::cout << "inv_normalizer: " << inv_normalizer << std::endl;
  for (int i=0; i<n_eigen; i++)
    std::cout << d->ptr[i] * inv_normalizer << std::endl;
  */
  //---------------------

  // Update Xt, G and minibatch_index
  Mat *Xkt =  Xt->subMat(0,0,n_eigen-1,n_dim-1);
  Xkt->copy(Ut);
  delete Xkt;

  for (int i=0; i<n_eigen; i++)
    G->ptr[i][i] = d->ptr[i];

  minibatch_index=0;
}

void PcaEstimator::GetLeadingEigen(Vec *the_d, Mat *the_Vt)
{

  // Copy the eigen values and normalize them by (1 - pow(gamma,real(t+1)))/(1 - gamma);
  the_d->copy(d);

  real normalizer = (1.0 - pow(gamma, (real)n_observations)) /(1.0 - gamma);
  real inv_normalizer = 1.0 / normalizer;

  for (int i=0; i<the_d->n; i++)
    the_d->ptr[i] *= inv_normalizer;

  // Copy the unnormalized eigen vectors and normalize them
  the_Vt->copy(Ut);

  warning("I haven't normalized the eigen vectors");

}

PcaEstimator::~PcaEstimator()
{
}

}
