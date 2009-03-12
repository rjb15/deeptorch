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
gradient_covariance_full-eigen\n\
\n\
This program loads a model and:\n\
  - computes all the gradients on the dataset\n\
  - computes the covariance of the gradients\n\
  - performs the the full eigendecomposition of that covariance\n\
  - saves the eigenvecs / eigenvals.\n";

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
#include "input_as_target_data_set.h"
#include "dynamic_data_set.h"
#include "concat_criterion.h"


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
  int flag_is_centered;

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
  cmd.addICmdArg("-is_centered", &flag_is_centered, "second moment (0) or covariance (1).");

  cmd.addSCmdOption("-model_label", &flag_model_label, "", "label used to describe the model", true);
  cmd.addICmdOption("-max_load", &flag_max_load, -1, "max number of examples to load for train", true);
  cmd.addBCmdOption("-binary_mode", &flag_binary_mode, false, "binary mode for files", true);

  cmd.read(argc, argv);
  assert ( flag_is_centered == 0 || flag_is_centered == 1 );

  // Allocator
  Allocator *allocator = new Allocator;

  // Data
  MatDataSet matdata(flag_data_filename, flag_n_inputs, 1, false,
                                flag_max_load, flag_binary_mode);
  ClassFormatDataSet data(&matdata,flag_n_classes);
  OneHotClassFormat class_format(&data);

  DataSet *the_data = &data;

  // Load the model
  CommunicatingStackedAutoencoder *csae = NULL;
  GradientMachine *model = NULL;

  if (!strcmp(flag_model_type, "csae")) {
    csae = LoadCSAE(allocator, flag_model_filename);
    model = csae;
  }
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
  else if (!strcmp(flag_criterion_type, "unsup-xentropy")) {
    // Must have a csae
    assert(csae);

    // The model!
    model = csae->unsup_machine;

    // Set up a ConcatCriterion
    DataSet **unsup_datasets = (DataSet**) allocator->alloc(sizeof(DataSet*)*(csae->n_hidden_layers));
    Criterion **the_criterions = (Criterion**) allocator->alloc(sizeof(Criterion *)*(csae->n_hidden_layers));

    for(int i=0; i<csae->n_hidden_layers; i++)     {
      // Dataset
      if (i == 0)
        unsup_datasets[0] = new(allocator) InputAsTargetDataSet(&data);
      else
        unsup_datasets[i] = new(allocator) DynamicDataSet(&data, (Sequence*)NULL, csae->encoders[i-1]->outputs);
  
      // Criterion
      the_criterions[i] = NewUnsupCriterion(allocator, "xentropy", csae->decoders[i]->n_outputs);
      the_criterions[i]->setBOption("average frame size", true);  // averaging frame size!
      the_criterions[i]->setDataSet(unsup_datasets[i]);
    }

    //
    Criterion *concat_criterion;
    concat_criterion = new(allocator) ConcatCriterion(csae->unsup_machine->n_outputs,
                                                 csae->n_hidden_layers,
                                                 the_criterions,
                                                 NULL);

    criterion = concat_criterion;
    the_data = unsup_datasets[0];
  }
  else
    error("criterion type %s is not supported.", flag_criterion_type);

  // Get the number of parameters
  int n_params = GetNParams(model);
  std::cout << n_params << " parameters." << std::endl;

  // Allocate the mat to save the gradients
  Mat *gradients = new(allocator) Mat(the_data->n_examples, n_params);
  Mat *covariance = new(allocator) Mat(n_params, n_params);

  // Set the dataset
  model->setDataSet(the_data);
  criterion->setDataSet(the_data);
  ClearDerivatives(model);

  // Iterate over the data
  int tick = 1;
  Parameters *der_params = model->der_params;
  for (int i=0; i<the_data->n_examples; i++)  {
    the_data->setExample(i);

    // fbprop
    model->forward(the_data->inputs);
    criterion->forward(model->outputs);
    criterion->backward(model->outputs, NULL);
    model->backward(the_data->inputs, criterion->beta);
    
    // Copy and clear der_params.
    real *ptr = gradients->ptr[i];
    for(int j=0; j<der_params->n_data; j++) {
      memcpy(ptr, der_params->data[j], der_params->size[j] * sizeof(real));
      memset(der_params->data[j], 0, sizeof(real)*der_params->size[j]);
      ptr += der_params->size[j];
    }

    // Progress
    if ( (real)i/the_data->n_examples > tick/10.0)  {
      std::cout << ".";
      flush(std::cout);
      tick++;
    }
  }

  // Compute the mean gradient norm
  real mean_norm2 = 0.0;
  Vec gradient(NULL, n_params);
  for (int i=0; i<the_data->n_examples; i++)  {
    gradient.ptr = gradients->ptr[i];
    mean_norm2 += gradient.norm2();
  }
  mean_norm2 /= the_data->n_examples;
  std::cout << "mean_norm2 = " << mean_norm2 << std::endl;

  // Compute the mean gradient
  message("Computing the mean of the gradients.");
  Vec *gradient_mean = new(allocator) Vec(n_params);
  for (int i=0; i<n_params; i++)
    gradient_mean->ptr[i] = 0.0;

  for (int i=0; i<the_data->n_examples; i++)  {
    for (int j=0; j<n_params; j++)
      gradient_mean->ptr[j] += gradients->ptr[i][j];
  }

  real inv_n_examples = 1.0 / the_data->n_examples;
  for (int i=0; i<n_params; i++)
    gradient_mean->ptr[i] *= inv_n_examples;

  // Center the gradients
  if (flag_is_centered) {
    message("Centering the gradients.");
    for (int i=0; i<the_data->n_examples; i++)
      for (int j=0; j<n_params; j++)
        gradients->ptr[i][j] -= gradient_mean->ptr[j];
  } else  {
    message("*NOT* Centering the gradients.");
  }

  // Compute the covariance
  message("Computing the covariance.");
  mxTrMatMulMat(gradients, gradients, covariance);
  mxRealMulMat(1.0/(the_data->n_examples-1.0), covariance, covariance);

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
  std::string savedir;
  if (flag_is_centered)
    savedir = "covariance";
  else
    savedir = "second-moment";
  savedir += flag_model_label;

  std::stringstream command;
  command << "mkdir " << savedir;
  system(command.str().c_str());

  // ASCII
  std::ofstream fd_eigenvals;
  std::stringstream ss_filename;
  ss_filename << savedir << "/eigenvals_full.txt";
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
  ss_filename << savedir << "/eigenvecs_full.txt";
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



