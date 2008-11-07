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
hessian_evaluator\n\
\n\
This program evaluates a model around its current parameter values using\n\
the eigen vectors given in input (usually the hessian's eigen values-vectors or their approximation).\n\
We expect to find a folder './hessian' containing the eigen-info.\n";

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
#include "communicating_stacked_autoencoder.h"
#include "pca_estimator.h"
#include "helpers.h"

using namespace Torch;

void LoadEigen(char *hessian_folder, Allocator *allocator, GradientMachine *machine, int n_eigen, Vec *eigenvals, Mat *eigenvecs);
void StepInParameterSpace(GradientMachine *machine, int id_eigen,
                          Vec *eigenvals, Mat *eigenvecs, real stepsize,
                          bool eigen_step);
void EvaluateCostAlongDirection(GradientMachine *machine,
                                char *data_label,
                                ClassFormatDataSet *data,
                                OneHotClassFormat *class_format,
                                int id_eigen,
                                Vec *eigenvals,
                                Mat *eigenvecs,
                                int n_steps_oneside,
                                real stepsize,
                                bool eigenstep);

// ************
// *** MAIN ***
// ************
int main(int argc, char **argv)
{

  // The command-line
  int flag_n_inputs;
  int flag_n_classes;
  char *flag_data_filename;
  char *flag_data_label;
  char *flag_model_filename;
  char *flag_hessian_folder;

  int flag_n_eigen;
  int flag_n_steps_oneside;
  real flag_stepsize;
  bool flag_eigenstep;

  int flag_max_load;
  bool flag_binary_mode;

  CmdLine cmd;
  cmd.info(help);

  cmd.addICmdArg("-n_inputs", &flag_n_inputs, "number of inputs");
  cmd.addICmdArg("-n_classes", &flag_n_classes, "number of targets");
  cmd.addSCmdArg("-data_filename", &flag_data_filename, "Filename for the data.");
  cmd.addSCmdArg("-data_label", &flag_data_label, "Label for the data, ie train/test. Used for naming the measurer files.");
  cmd.addSCmdArg("-model_filename", &flag_model_filename, "the model filename");
  cmd.addSCmdArg("-hessian_folder", &flag_hessian_folder, "folder where to find eigenvals and eigenvecs");

  cmd.addICmdOption("-n_eigen", &flag_n_eigen, 10, "number of eigen directions to explore", true);
  cmd.addICmdOption("-n_steps_oneside", &flag_n_steps_oneside, 10, "How many evaluations to perform on each side of a direction.", true);
  cmd.addRCmdOption("-stepsize", &flag_stepsize, 1e-4, "Stepsize in parameter space.", true);
  cmd.addBCmdOption("-eigenstep", &flag_eigenstep, false, "True if the step is based on (stepsize multiplied by) the eigenvalue).", true);

  cmd.addICmdOption("-max_load", &flag_max_load, -1, "max number of examples to load for train", true);
  cmd.addBCmdOption("-binary_mode", &flag_binary_mode, false, "binary mode for files", true);

  cmd.read(argc, argv);

  // Allocator
  Allocator *allocator = new Allocator;

  // Load the data
  MatDataSet matdata(flag_data_filename, flag_n_inputs, 1, false,
                                flag_max_load, flag_binary_mode);
  ClassFormatDataSet data(&matdata,flag_n_classes);
  OneHotClassFormat class_format(&data);  // Not sure about this... what if not
                                          // all classes are in the test set?

  // Load the model
  CommunicatingStackedAutoencoder *csae = LoadCSAE(allocator, flag_model_filename);

  // Load the eigen values vectors
  int n_params = 0;
  for (int i=0; i<csae->params->n_data; i++)  {
    n_params += csae->params->size[i];
  }
  Vec *eigenvals = new(allocator) Vec(flag_n_eigen);
  Mat *eigenvecs = new(allocator) Mat(flag_n_eigen, n_params);

  LoadEigen(flag_hessian_folder, allocator, csae, flag_n_eigen, eigenvals, eigenvecs);

  // Evaluate cost along the eigen directions
  for (int i=0; i<flag_n_eigen; i++) 
    EvaluateCostAlongDirection(csae, flag_data_label, &data, &class_format, i, eigenvals,
                                     eigenvecs, flag_n_steps_oneside, flag_stepsize,
                                     flag_eigenstep);

  delete allocator;
  return(0);
}

void LoadEigen(char *hessian_folder, Allocator *allocator, GradientMachine *machine, int n_eigen, Vec *eigenvals, Mat *eigenvecs)
{
  std::stringstream ss_filename;
  std::string line;
  std::stringstream tokens;

  int n_params = 0;
  for (int i=0; i<machine->params->n_data; i++)  {
    n_params += machine->params->size[i];
  }


  // load the eigen values
  std::ifstream fd_eigenvals;
  ss_filename << hessian_folder << "eigenvals.txt";
  fd_eigenvals.open(ss_filename.str().c_str());
  if (!fd_eigenvals.is_open())
    error("Can't open yourpath/eigenvals.txt");

  int n_val = 0;
  while (getline(fd_eigenvals, line))  {
    tokens.str(line);
    tokens.clear();

    tokens >> eigenvals->ptr[n_val];
    n_val++;
  }

  fd_eigenvals.close();
  assert (n_val == n_eigen);

  // load the eigen vectors
  std::ifstream fd_eigenvecs;
  ss_filename.str("");
  ss_filename.clear();
  ss_filename << hessian_folder << "eigenvecs.txt";
  fd_eigenvecs.open(ss_filename.str().c_str());
  if (!fd_eigenvecs.is_open())
    error("Can't open yourpath/eigenvecs.txt");

  int n_vecs=0;
  real value;
  while (getline(fd_eigenvecs, line))  {
    tokens.str(line);
    tokens.clear();
    n_val = 0;

    while (tokens >> value) {
      eigenvecs->ptr[n_vecs][n_val] = value;
      n_val++;
    }
    assert (n_val == n_params);
    n_vecs++;
  }

  fd_eigenvecs.close();
  assert (n_vecs == n_eigen);

}

// Move in parameter space.
void StepInParameterSpace(GradientMachine *machine, int id_eigen,
Vec *eigenvals, Mat *eigenvecs, real stepsize, bool eigen_step)
{
  int eig_offset = 0;

  for (int i=0; i<machine->params->n_data; i++)  {
    real *ptr = machine->params->data[i];
    for (int j=0; j<machine->params->size[i]; j++)
      ptr[j] += stepsize * eigenvecs->ptr[id_eigen][eig_offset+j];
    eig_offset += machine->params->size[i];
  }
}

void EvaluateCostAlongDirection(GradientMachine *machine,
                                char *data_label,
                                ClassFormatDataSet *data,
                                OneHotClassFormat *class_format,
                                int id_eigen,
                                Vec *eigenvals,
                                Mat *eigenvecs,
                                int n_steps_oneside,
                                real stepsize,
                                bool eigenstep)
{
  Allocator *allocator = new Allocator;

  // Build a list of measurers
  MeasurerList measurers;

  std::stringstream command;
  command << "mkdir stepsize=" << stepsize;
  system(command.str().c_str());

  // NLL measurer
  std::stringstream measurer_filename;
  measurer_filename << "./stepsize=" << stepsize << "/" << data_label << "_nll_eigen" << id_eigen << ".txt";
  DiskXFile *file_nll = new(allocator) DiskXFile(measurer_filename.str().c_str(),"w");
  ClassNLLMeasurer *measurer_nll = new(allocator) ClassNLLMeasurer(machine->outputs,
                                                                   data, class_format, file_nll);
  measurers.addNode(measurer_nll);

  // Class measurer
  measurer_filename.str("");
  measurer_filename.clear();
  measurer_filename << "./stepsize=" << stepsize << "/" << data_label << "_class_eigen" << id_eigen << ".txt";
  DiskXFile *file_class = new(allocator) DiskXFile(measurer_filename.str().c_str(),"w");
  ClassMeasurer *measurer_class = new(allocator) ClassMeasurer(machine->outputs, 
                                                               data, class_format, file_class);
  measurers.addNode(measurer_class);

  // Trainer
  StochasticGradient trainer(machine, NULL);

  // Move to the most "negative" point in parameter space
  StepInParameterSpace(machine, id_eigen, eigenvals, eigenvecs, -n_steps_oneside*stepsize, eigenstep);

  // Test and move on the "positive side"
  for (int i=0; i<2*n_steps_oneside+1; i++) {
    trainer.test(&measurers);
    StepInParameterSpace(machine, id_eigen, eigenvals, eigenvecs, stepsize, eigenstep);
  }

  // Return to the initial point!
  StepInParameterSpace(machine, id_eigen, eigenvals, eigenvecs, -(n_steps_oneside+1)*stepsize, eigenstep);

  delete allocator;
}




