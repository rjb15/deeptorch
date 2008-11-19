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
hessian_estimator_full-eigen\n\
\n\
This program computes the full eigendecomposition of the covariance\n\
of the gradients on a model's parameters.\n";

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
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
#include "GradientMachine.h"
#include "communicating_stacked_autoencoder.h"
#include "pca_estimator.h"
#include "helpers.h"
#include "analysis_utilities.h"

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
  char *flag_model_type;
  char *flag_criterion_type;

  char *flag_model_label;
  int flag_max_load;
  bool flag_binary_mode;

  CmdLine cmd;
  cmd.info(help);

  cmd.addICmdArg("-n_inputs", &flag_n_inputs, "number of inputs");
  cmd.addICmdArg("-n_classes", &flag_n_classes, "number of targets");
  cmd.addSCmdArg("-data_filename", &flag_data_filename, "Filename for the data.");
  cmd.addSCmdArg("-model_filename", &flag_model_filename, "the model filename");
  cmd.addSCmdArg("-model_type", &flag_model_type, "the type of the model: csae or linear.");
  cmd.addSCmdArg("-criterion_type", &flag_criterion_type, "the type of the criterion: 'mse' or 'class-nll'.");

  cmd.addSCmdOption("-model_label", &flag_model_label, "", "label used to describe the model", true);
  cmd.addICmdOption("-max_load", &flag_max_load, -1, "max number of examples to load for train", true);
  cmd.addBCmdOption("-binary_mode", &flag_binary_mode, false, "binary mode for files", true);

  cmd.read(argc, argv);

  // Allocator
  Allocator *allocator = new Allocator;

  // Data
  MatDataSet matdata(flag_data_filename, flag_n_inputs, 1, false,
                                flag_max_load, flag_binary_mode);
  ClassFormatDataSet data(&matdata,flag_n_classes);
  OneHotClassFormat class_format(&data);

  // Load the model
  GradientMachine *model = NULL;
  if (!strcmp(flag_model_type, "csae"))
    model = LoadCSAE(allocator, flag_model_filename);
  else if (!strcmp(flag_model_type, "linear"))
    model = LoadCoder(allocator, flag_model_filename);
  else
    error("model type %s is not supported.", flag_model_type);

  // Criterion
  Criterion *criterion = NULL;
  if (!strcmp(flag_criterion_type, "mse"))
    criterion = new(allocator) MSECriterion(model->n_outputs);
  else if (!strcmp(flag_criterion_type, "class-nll"))
    criterion = new(allocator) ClassNLLCriterion(&class_format);
  else
    error("criterion type %s is not supported.", flag_criterion_type);

  // Get the number of parameters
  int n_params = GetNParams(model);
  std::cout << n_params << " parameters." << std::endl;

  // Allocate the mat to save the gradients
  Mat *gradients = new(allocator) Mat(data.n_examples, n_params);
  Mat *covariance = new(allocator) Mat(n_params, n_params);

  // Set the dataset
  model->setDataSet(&data);
  criterion->setDataSet(&data);
  ClearDerivatives(model);

  // Iterate over the data
  int tick = 1;
  Parameters *der_params = model->der_params;
  for (int i=0; i<data.n_examples; i++)  {
    data.setExample(i);

    // fbprop
    model->forward(data.inputs);
    criterion->forward(model->outputs);
    criterion->backward(model->outputs, NULL);
    model->backward(data.inputs, criterion->beta);
    
    // Copy and clear der_params.
    real *ptr = gradients->ptr[i];
    for(int j=0; j<der_params->n_data; j++) {
      memcpy(ptr, der_params->data[j], der_params->size[j] * sizeof(real));
      memset(der_params->data[j], 0, sizeof(real)*der_params->size[j]);
      ptr += der_params->size[j];
    }

    // Progress
    if ( (real)i/data.n_examples > tick/10.0)  {
      std::cout << ".";
      flush(std::cout);
      tick++;
    }
  }

  // Compute the mean gradient norm
  real mean_norm2 = 0.0;
  Vec gradient(NULL, n_params);
  for (int i=0; i<data.n_examples; i++)  {
    gradient.ptr = gradients->ptr[i];
    mean_norm2 += gradient.norm2();
  }
  mean_norm2 /= data.n_examples;
  std::cout << "mean_norm2 = " << mean_norm2 << std::endl;

  // Compute the mean gradient
  message("Computing the mean of the gradients.");
  Vec *gradient_mean = new(allocator) Vec(n_params);
  for (int i=0; i<n_params; i++)
    gradient_mean->ptr[i] = 0.0;

  for (int i=0; i<data.n_examples; i++)  {
    for (int j=0; j<n_params; j++)
      gradient_mean->ptr[j] += gradients->ptr[i][j];
  }

  real inv_n_examples = 1.0 / data.n_examples;
  for (int i=0; i<n_params; i++)
    gradient_mean->ptr[i] *= inv_n_examples;

  // Center the gradients
  /*message("Centering the gradients.");
  for (int i=0; i<data.n_examples; i++)
    for (int j=0; j<n_params; j++)
      gradients->ptr[i][j] -= gradient_mean->ptr[j];
  */

  // Compute the covariance
  message("Computing the covariance.");
  mxTrMatMulMat(gradients, gradients, covariance);
  mxRealMulMat(1.0/(data.n_examples-1.0), covariance, covariance);

  // Free up some memory!
  allocator->free(gradients);

  // Memory for eigendecomp.
  message("Performing the eigendecomposition.");
  Vec *d = new(allocator) Vec(n_params);
  Mat *V = new(allocator) Mat(n_params, n_params);

  // The eigen values and vectors are SORTED. Furthermore, the vectors
  // are on the columns.
  mxSymEig(covariance, V, d);

  mxTrMat(V, V);

  // Save the eigenvals
  message("Saving the results");
  std::ofstream fd_eigenvals;

  std::stringstream command;
  command << "mkdir hessian" << flag_model_label;
  system(command.str().c_str());

  // ASCII
  std::stringstream ss_filename;
  ss_filename << "hessian" << flag_model_label << "/eigenvals_full.txt";
  fd_eigenvals.open(ss_filename.str().c_str());
  if (!fd_eigenvals.is_open())
    error("Can't open file for eigenvals");
  for (int j=0; j<d->n; j++)
    fd_eigenvals << d->ptr[j] << std::endl;
  fd_eigenvals.close();

  // Save the eigenvecs - RIGHT NOW THEY'RE on the rows!
  std::ofstream fd_eigenvecs;

  // ASCII
  ss_filename.str("");
  ss_filename.clear();
  ss_filename << "hessian" << flag_model_label << "/eigenvecs_full.txt";
  fd_eigenvecs.open(ss_filename.str().c_str());
  if (!fd_eigenvecs.is_open())
    error("Can't open eigenvecs file");
  for (int j=0; j<V->m; j++)  {
    for (int k=0; k<V->n; k++)
      fd_eigenvecs << V->ptr[j][k] << " ";
    fd_eigenvecs << std::endl;
  }
  fd_eigenvecs.close();

  delete allocator;
  return(0);
}



