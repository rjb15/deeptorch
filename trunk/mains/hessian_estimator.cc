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
hessian_estimator\n\
\n\
This program estimates the hessian's leading (largest) eigen values-vectors\n\
using the covariance approximation. To do so, we use the pca_estimator on the\n\
gradients.\n";

#include <string>
#include <sstream>
#include <iostream>
#include <cassert>

#include "Allocator.h"
#include "CmdLine.h"
#include "DiskXFile.h"
#include "MatDataSet.h"
#include "ClassFormatDataSet.h"
#include "OneHotClassFormat.h"
#include "ClassNLLCriterion.h"
#include "matrix.h"
#include "Parameters.h"
#include  "communicating_stacked_autoencoder.h"
#include "pca_estimator.h"
#include "helpers.h"

using namespace Torch;

// ************
// *** MAIN ***
// ************
int main(int argc, char **argv)
{

  // The command-line
  int flag_n_inputs;
  int flag_n_classes;
  char *flag_data_filename;
  char *flag_model_filename;

  int flag_n_eigen;
  int flag_minibatch_size;
  real flag_gamma;
  int flag_iterations;

  int flag_max_load;
  bool flag_binary_mode;

  CmdLine cmd;
  cmd.info(help);

  cmd.addICmdArg("-n_inputs", &flag_n_inputs, "number of inputs");
  cmd.addICmdArg("-n_classes", &flag_n_classes, "number of targets");
  cmd.addSCmdArg("-data_filename", &flag_data_filename, "Filename for the data.");
  cmd.addSCmdArg("-model_filename", &flag_model_filename, "the model filename");

  cmd.addICmdOption("-n_eigen", &flag_n_eigen, 10, "number of eigen values in the low rank estimate", true);
  cmd.addICmdOption("-minibatch_size", &flag_minibatch_size, 10, "number of observations before a reevaluation", true);
  cmd.addRCmdOption("-gamma", &flag_gamma, 0.999, "discount factor", true);
  cmd.addICmdOption("-iterations", &flag_iterations, 1, "number of iterations over the data", true);

  cmd.addICmdOption("-max_load", &flag_max_load, -1, "max number of examples to load for train", true);
  cmd.addBCmdOption("-binary_mode", &flag_binary_mode, false, "binary mode for files", true);

  cmd.read(argc, argv);

  // Allocator
  Allocator *allocator = new Allocator;

  // Data
  MatDataSet matdata(flag_data_filename, flag_n_inputs, 1, false,
                                flag_max_load, flag_binary_mode);
  ClassFormatDataSet data(&matdata,flag_n_classes);
  OneHotClassFormat class_format(&data);  // Not sure about this... what if not
                                          // all classes are in the test set?

  // Load the model
  CommunicatingStackedAutoencoder *csae = LoadCSAE(allocator, flag_model_filename);

  // Criterion
  ClassNLLCriterion criterion(&class_format);
  
  // Measurer

  // The parameters of the model
  Parameters *der_params = csae->der_params;
  int n_param_groups = der_params->n_data;
  std::cout << n_param_groups << " groups of parameters." << std::endl;
  assert(n_param_groups == csae->n_hidden_layers+1);

  // Estimator - get one per layer / parameter group
  PcaEstimator **estimators = (PcaEstimator**) allocator->alloc(sizeof(PcaEstimator*)*n_param_groups); 
  for (int i=0; i<n_param_groups; i++)  {
    estimators[i] = new(allocator) PcaEstimator(der_params->size[i], flag_n_eigen, flag_minibatch_size, flag_gamma);
  }

  // Iterate over the data
  csae->setDataSet(&data);
  criterion.setDataSet(&data);

  int tick = 1;

  for (int it=0; it<flag_iterations; it++)  {
    for (int i=0; i<data.n_examples; i++)  {

      data.setExample(i);

      // fbprop
      csae->forward(data.inputs);
      criterion.forward(csae->outputs);

      criterion.backward(csae->outputs, NULL);
      csae->backward(data.inputs, criterion.beta);
    
      // Observe the gradients
      for(int j=0; j<der_params->n_data; j++) {
        Vec sample(der_params->data[j], der_params->size[j]);
        estimators[j]->Observe(&sample);
      }
 
      // Progress
      if ( (real)i/data.n_examples > tick/100.0)  {
        std::cout << ".";
        flush(std::cout);
        tick++;
      }
   
    }
  }

  for (int i=0; i<der_params->n_data; i++)  {
    // Grab and print the eigen values vectors
    Vec eigenvals(flag_n_eigen);
    Mat eigenvecs(flag_n_eigen, der_params->size[i]);

    estimators[i]->GetLeadingEigen(&eigenvals, &eigenvecs);

    std::cout << der_params->size[i] << " parameters." << std::endl;
    for (int j=0; j<eigenvals.n; j++)
      std::cout << eigenvals.ptr[j] << std::endl;

    // Print first eigen vector
    /*for (int j=0; j<der_params->size[i]; j++)
      std::cout << eigenvecs.ptr[0][j] << " ";
    std::cout << std::endl;*/
  }

  delete allocator;
  return(0);
}

