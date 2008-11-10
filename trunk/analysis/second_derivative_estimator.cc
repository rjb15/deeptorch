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
second_derivative_estimator\n\
\n\
This program estimates by finite difference the second derivative of a cost\n\
function wrt a model's parameters, in a few directions.\n";

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


// ************
// *** MAIN ***
// ************
int main(int argc, char **argv)
{
  // The command-line flags
  int flag_n_inputs;
  int flag_n_classes;
  char *flag_data_filename;
  char *flag_data_label;
  char *flag_model_filename;
  char *flag_model_label;
  char *flag_directions_filename;

  int flag_n_directions;
  real flag_epsilon;

  int flag_max_load;
  bool flag_binary_mode;

  // The actual command line
  CmdLine cmd;
  cmd.info(help);

  cmd.addICmdArg("-n_inputs", &flag_n_inputs, "number of inputs");
  cmd.addICmdArg("-n_classes", &flag_n_classes, "number of targets");
  cmd.addSCmdArg("-data_filename", &flag_data_filename, "Filename for the data.");
  cmd.addSCmdArg("-data_label", &flag_data_label, "Label for the data, ie train/test.");
  cmd.addSCmdArg("-model_filename", &flag_model_filename, "the model filename");
  cmd.addSCmdArg("-model_label", &flag_model_label, "Label for describing the model.");
  cmd.addSCmdArg("-directions_filename", &flag_directions_filename, "the name of the file containing the directions");

  cmd.addICmdOption("-n_directions", &flag_n_directions, 7, "number directions to load from the file", true);
  cmd.addRCmdOption("-epsilon", &flag_epsilon, 1e-6, "stepsize for finite difference", true);
  cmd.addICmdOption("-max_load", &flag_max_load, -1, "max number of examples to load for train", true);
  cmd.addBCmdOption("-binary_mode", &flag_binary_mode, false, "binary mode for files", true);

  cmd.read(argc, argv);

  // Allocator
  Allocator *allocator = new Allocator;

  // Load the data
  MatDataSet matdata(flag_data_filename, flag_n_inputs, 1, false,
                                flag_max_load, flag_binary_mode);
  ClassFormatDataSet *data = new(allocator) ClassFormatDataSet(&matdata,flag_n_classes);
  OneHotClassFormat class_format(data);  // Not sure about this... what if not
                                          // all classes are in the test set?

  // Load the model
  CommunicatingStackedAutoencoder *csae = LoadCSAE(allocator, flag_model_filename);

  // Determine the number of parameters
  int n_params = 0;
  for (int i=0; i<csae->params->n_data; i++)  {
    n_params += csae->params->size[i];
  }

  // Criterion
  ClassNLLCriterion *criterion = new(allocator) ClassNLLCriterion(&class_format);

  // Load the directions
  Mat *directions = new(allocator) Mat(flag_n_directions, n_params);
  LoadDirections(flag_directions_filename, flag_n_directions, directions);

  // Evaluate the gradient
  Vec *gradient = new(allocator) Vec(n_params);
  EvaluateGradient(csae, criterion, data, gradient);

  // For each direction:
  //    - project the gradient in the direction.
  //    - step in the direction
  //    - reevaluate the gradient and project it in the direction
  //    - return to initial position
  //    - compute the second derivative
  Vec* direction = new(allocator) Vec(NULL, n_params);
  Vec* gradient_after_step = new(allocator) Vec(n_params);
  real gradient_in_direction, gradient_in_direction_after_step;
  real second_derivative;

  for (int i=0; i<flag_n_directions; i++) {
    direction->ptr = directions->ptr[i];
    assert( (direction->norm2()-1.0) < 1e-6 );

    gradient_in_direction = direction->iP(gradient);

    StepInParameterSpace(csae, direction, flag_epsilon);
    EvaluateGradient(csae, criterion, data, gradient_after_step);

    gradient_in_direction_after_step = direction->iP(gradient_after_step);

    second_derivative = fabs(gradient_in_direction_after_step-gradient_in_direction) / flag_epsilon;

    std::cout << second_derivative << std::endl;
  } 

  delete allocator;
  return(0);
}
