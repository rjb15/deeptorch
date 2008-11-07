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
const char *help = "\
pca_tester\n\
\n\
This program tests the pca_estimator. \n";

#include <string>
#include <sstream>
#include <iostream>

#include "Allocator.h"
#include "CmdLine.h"
#include "DiskXFile.h"
#include "MatDataSet.h"
#include "matrix.h"

#include "analysis/pca_estimator.h"

using namespace Torch;

// ************
// *** MAIN ***
// ************
int main(int argc, char **argv)
{

  // The command-line
  int flag_n_dim;
  char *flag_data_filename;

  int flag_n_eigen;
  int flag_minibatch_size;
  real flag_gamma;
  int flag_iterations;

  int flag_max_load;
  bool flag_binary_mode;

  CmdLine cmd;
  cmd.info(help);

  cmd.addICmdArg("-n_dim", &flag_n_dim, "Dimensionality of the samples.");
  cmd.addSCmdArg("-data_filename", &flag_data_filename, "Filename for the data.");

  cmd.addICmdOption("-n_eigen", &flag_n_eigen, 10, "number of eigen values in the low rank estimate", true);
  cmd.addICmdOption("-minibatch_size", &flag_minibatch_size, 10, "number of observations before a reevaluation", true);
  cmd.addRCmdOption("-gamma", &flag_gamma, 0.999, "discount factor", true);
  cmd.addICmdOption("-iterations", &flag_iterations, 1, "number of iterations over the data", true);

  cmd.addICmdOption("-max_load", &flag_max_load, -1, "max number of examples to load for train", true);
  cmd.addBCmdOption("-binary_mode", &flag_binary_mode, false, "binary mode for files", true);

  cmd.read(argc, argv);

  // Allocator
  Allocator *allocator = new Allocator;

  // data
  MatDataSet data(flag_data_filename, flag_n_dim, 0, false,
                                flag_max_load, flag_binary_mode);

  // Estimator
  PcaEstimator the_estimator(flag_n_dim, flag_n_eigen, flag_minibatch_size, flag_gamma);

  // Iterate over the data
  Vec sample(NULL, flag_n_dim);
  for (int it=0; it<flag_iterations; it++)  {
    for (int i=0; i<data.n_examples; i++)  {
      data.setExample(i);
      sample.ptr = data.inputs->frames[0];
      the_estimator.Observe(&sample);
    }
  }

  // Grab and print the eigen values vectors
  Vec eigenvals(flag_n_eigen);
  Mat eigenvecs(flag_n_eigen, flag_n_dim);

  the_estimator.GetLeadingEigen(&eigenvals, &eigenvecs);

  for (int i=0; i<eigenvals.n; i++)
    std::cout << eigenvals.ptr[i] << std::endl;

  // -----
  // A small test of mxSymEig...
  /*
  // do the eigendecomposition of that matrix and print out
  Mat the_mat(data.n_examples, flag_n_dim);
  Vec the_vec(NULL, flag_n_dim);
  for (int i=0; i<data.n_examples; i++)  {
    data.setExample(i);
    the_vec.ptr = data.inputs->frames[0];
    the_mat.setRow(i, &the_vec);
  }

  Mat Vt(data.n_examples, data.n_examples);
  Vec d(data.n_examples);

  mxSymEig(&the_mat, &Vt, &d);

  for (int i=0; i<data.n_examples; i++)
    std::cout << d.ptr[i] << std::endl;

  for (int i=0; i<the_mat.m; i++) {
    for (int j=0; j<the_mat.n; j++)
      std::cout << Vt.ptr[i][j] << " ";
    std::cout << std::endl;
  }
  */
  // -----

  delete allocator;
  return(0);
}

