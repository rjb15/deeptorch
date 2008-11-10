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
evaluate_cost_along_directions\n\
\n\
This program evaluates a model around its current parameter values in\n\
directions given in input (usually the hessian's eigen values-vectors\n\
or their approximation).\n";

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

#include "analysis_utilities.h"

using namespace Torch;

void EvaluateCostAlongDirection(GradientMachine *machine,
                                char *data_label,
                                ClassFormatDataSet *data,
                                OneHotClassFormat *class_format,
                                int id_direction,
                                Mat *directions,
                                int n_steps_oneside,
                                real stepsize);

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
  char *flag_directions_filename;

  int flag_n_directions;
  int flag_n_steps_oneside;
  real flag_stepsize;

  int flag_max_load;
  bool flag_binary_mode;

  CmdLine cmd;
  cmd.info(help);

  cmd.addICmdArg("-n_inputs", &flag_n_inputs, "number of inputs");
  cmd.addICmdArg("-n_classes", &flag_n_classes, "number of targets");
  cmd.addSCmdArg("-data_filename", &flag_data_filename, "Filename for the data.");
  cmd.addSCmdArg("-data_label", &flag_data_label, "Label for the data, ie train/test. Used for naming the measurer files.");
  cmd.addSCmdArg("-model_filename", &flag_model_filename, "the model filename");
  cmd.addSCmdArg("-directions_filename", &flag_directions_filename, "Name of the file containing the directions.");

  cmd.addICmdOption("-n_directions", &flag_n_directions, 6, "number of directions to explore (first from the file)", true);
  cmd.addICmdOption("-n_steps_oneside", &flag_n_steps_oneside, 10, "How many evaluations to perform on each side of a direction.", true);
  cmd.addRCmdOption("-stepsize", &flag_stepsize, 1e-4, "Stepsize in parameter space.", true);

  cmd.addICmdOption("-max_load", &flag_max_load, -1, "max number of examples to load for train", true);
  cmd.addBCmdOption("-binary_mode", &flag_binary_mode, false, "binary mode for files", true);

  cmd.read(argc, argv);

  // Allocator
  Allocator *allocator = new Allocator;

  // Load the data
  MatDataSet matdata(flag_data_filename, flag_n_inputs, 1, false,
                                flag_max_load, flag_binary_mode);
  ClassFormatDataSet data(&matdata,flag_n_classes);
  OneHotClassFormat class_format(&data);

  // Load the model
  CommunicatingStackedAutoencoder *csae = LoadCSAE(allocator, flag_model_filename);

  // Load the directions
  int n_params = 0;
  for (int i=0; i<csae->params->n_data; i++)  {
    n_params += csae->params->size[i];
  }
  Mat *directions = new(allocator) Mat(flag_n_directions, n_params);
  LoadDirections(flag_directions_filename, flag_n_directions, directions);

  // Evaluate cost along the directions
  for (int i=0; i<flag_n_directions; i++) 
    EvaluateCostAlongDirection(csae, flag_data_label, &data, &class_format, i, directions,
                                     flag_n_steps_oneside, flag_stepsize);

  delete allocator;
  return(0);
}


void EvaluateCostAlongDirection(GradientMachine *machine,
                                char *data_label,
                                ClassFormatDataSet *data,
                                OneHotClassFormat *class_format,
                                int id_direction,
                                Mat *directions,
                                int n_steps_oneside,
                                real stepsize)
{
  Allocator *allocator = new Allocator;

  // Build a list of measurers
  MeasurerList measurers;

  std::stringstream command;
  command << "mkdir stepsize=" << stepsize;
  system(command.str().c_str());

  // NLL measurer
  std::stringstream measurer_filename;
  measurer_filename << "./stepsize=" << stepsize << "/" << data_label << "_nll_dir" << id_direction << ".txt";
  DiskXFile *file_nll = new(allocator) DiskXFile(measurer_filename.str().c_str(),"w");
  ClassNLLMeasurer *measurer_nll = new(allocator) ClassNLLMeasurer(machine->outputs,
                                                                   data, class_format, file_nll);
  measurers.addNode(measurer_nll);

  // Class measurer
  measurer_filename.str("");
  measurer_filename.clear();
  measurer_filename << "./stepsize=" << stepsize << "/" << data_label << "_class_dir" << id_direction << ".txt";
  DiskXFile *file_class = new(allocator) DiskXFile(measurer_filename.str().c_str(),"w");
  ClassMeasurer *measurer_class = new(allocator) ClassMeasurer(machine->outputs, 
                                                               data, class_format, file_class);
  measurers.addNode(measurer_class);

  // Trainer
  StochasticGradient trainer(machine, NULL);

  // Move to the most "negative" point in parameter space
  Vec direction(NULL, directions->n);
  direction.ptr = directions->ptr[id_direction];
  StepInParameterSpace(machine, &direction, -n_steps_oneside*stepsize);

  // Test and move on the "positive side"
  for (int i=0; i<2*n_steps_oneside+1; i++) {
    trainer.test(&measurers);
    StepInParameterSpace(machine, &direction, stepsize);
  }

  // Return to the initial point!
  StepInParameterSpace(machine, &direction, -(n_steps_oneside+1)*stepsize);

  delete allocator;
}




