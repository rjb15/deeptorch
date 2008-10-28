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
#ifndef TORCH_PCA_ESTIMATOR_H_
#define TORCH_PCA_ESTIMATOR_H_

#include "Object.h"
#include "matrix.h"

namespace Torch {

// The PCA estimator estimates the main (largest) eigen values and vectors
// of the covariance matrix of some samples.
//
// A moving low rank (#n_eigen#) estimate of the covariance is reevaluated
// after #minibatch_size# samples.
//
class PcaEstimator : public Object
{
  public:
    PcaEstimator(int n_dim_, int n_eigen_, int minibatch_size_, real gamma_);

    // Dimensionality of the observations
    int n_dim;
    // Number of eigen values-vectors to keep in each reevaluation
    int n_eigen;
    // Number of observations between each reevaluation
    int minibatch_size;
    // Discount factor in moving average estimator of mean and covariance.
    // We use:
    //  foo_{t+1} = gamma * foo_t + new 
    // and then we normalise using the geometric serie. 
    real gamma;
    // The regularizer
    real lambda;

    // Total number of observations: to compute the normalizer for the mean and
    // the covariance.
    int n_observations;
    // Index in the current minibatch
    int minibatch_index;

    // Matrix containing on its *rows*:
    // - the current unnormalized eigen vector estimates
    // - the observations since the last reevaluation
    Mat *Xt;

    // The discounted sum of the observations. 
    Vec *x_sum;

    // The Gram matrix of the observations, ie Xt Xt' (since Xt is rowwise)
    Mat *G;

    // Hold the results of the eigendecomposition of the Gram matrix G
    // (eigen vectors on columns of V).
    Vec *d;
    Mat *V; 

    // Holds the unnormalized eigenvectors of the covariance matrix before
    // they're copied back to Xt.
    Mat *Ut;

    virtual void Initialize();
    virtual void Observe(Vec *x);
    virtual void Reevaluate();

    // Copies the current estimate to already allocated the_d and the_Vt
    // (vectors on rows)
    virtual void GetLeadingEigen(Vec *the_d, Mat *the_Vt);

    virtual ~PcaEstimator();

};

}
#endif  // TORCH_PCA_ESTIMATOR_H_
