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
  char *flag_model_type;
  char *flag_criterion_type;
  int flag_is_centered;

  int flag_n_directions;
  real flag_epsilon;

  int flag_max_load;
  bool flag_binary_mode;
  char *flag_out_filename;

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
  cmd.addSCmdArg("-model_type", &flag_model_type, "the type of the model: csae or linear.");
  cmd.addSCmdArg("-criterion_type", &flag_criterion_type, "the type of the criterion: 'mse' or 'class-nll'.");
  cmd.addICmdArg("-is_centered", &flag_is_centered, "second moment (0) or variance (1).");

  cmd.addICmdOption("-n_directions", &flag_n_directions, 7, "number directions to load from the file", true);
  cmd.addRCmdOption("-epsilon", &flag_epsilon, 1e-6, "stepsize for finite difference", true);
  cmd.addICmdOption("-max_load", &flag_max_load, -1, "max number of examples to load for train", true);
  cmd.addBCmdOption("-binary_mode", &flag_binary_mode, false, "binary mode for files", true);
  cmd.addSCmdOption("-out_filename", &flag_out_filename, "second_derivatives.txt", "Name of the file to output to.", true);

  cmd.read(argc, argv);
  assert ( flag_is_centered == 0 || flag_is_centered == 1 );

  // Allocator
  Allocator *allocator = new Allocator;

  // Load the data
  MatDataSet matdata(flag_data_filename, flag_n_inputs, 1, false,
                                flag_max_load, flag_binary_mode);
  ClassFormatDataSet *data = new(allocator) ClassFormatDataSet(&matdata,flag_n_classes);
  OneHotClassFormat class_format(data);  // Not sure about this... what if not
                                          // all classes are in the test set?

  // Load the model
  GradientMachine *model = NULL;
  if (!strcmp(flag_model_type, "csae"))
    model = LoadCSAE(allocator, flag_model_filename);
  else if (!strcmp(flag_model_type, "linear"))
    model = LoadCoder(allocator, flag_model_filename);
  else
    error("model type %s is not supported.", flag_model_type);

  // Determine the number of parameters
  int n_params = GetNParams(model);
  std::cout << n_params << " parameters." << std::endl;

  // Criterion
  Criterion *criterion = NULL;
  if (!strcmp(flag_criterion_type, "mse"))
    criterion = new(allocator) MSECriterion(model->n_outputs);
  else if (!strcmp(flag_criterion_type, "class-nll"))
    criterion = new(allocator) ClassNLLCriterion(&class_format);
  else
    error("criterion type %s is not supported.", flag_criterion_type);

  // Load the directions
  Mat *directions = new(allocator) Mat(flag_n_directions, n_params);
  LoadDirections(flag_directions_filename, flag_n_directions, directions);

  // Evaluate the gradient
  Vec *gradient = new(allocator) Vec(n_params);
  EvaluateGradient(model, criterion, data, gradient);

  // For each direction:
  //    - project the gradient in the direction.
  //    - step in the direction
  //    - reevaluate the gradient and project it in the direction
  //    - return to initial position
  //    - compute the second derivative
  Vec* direction = new(allocator) Vec(NULL, n_params);
  Vec* gradient_pos_step = new(allocator) Vec(n_params);
  //Vec* gradient_neg_step = new(allocator) Vec(n_params);
  real gradient_in_direction, gradient_in_direction_pos_step;
  //real gradient_in_direction_neg_step;
  real second_derivative;
  //real variance;

  std::ofstream fd_second_der;
  fd_second_der.open(flag_out_filename);
  if (!fd_second_der.is_open())
    error("Could not open %s", flag_out_filename);

  for (int i=0; i<flag_n_directions; i++) {
    direction->ptr = directions->ptr[i];
    if( !(fabs(direction->norm2()-1.0) < 1e-6) )  {
      error("direction norm is not 1, but %d", direction->norm2());      
    }

    gradient_in_direction = direction->iP(gradient);

    // Compute the variance in the direction
    //variance = EvaluateGradientVarianceInDirection(model, criterion, data, direction, flag_is_centered); 

    // Positive step
    StepInParameterSpace(model, direction, flag_epsilon);
    EvaluateGradient(model, criterion, data, gradient_pos_step);
    gradient_in_direction_pos_step = direction->iP(gradient_pos_step);

    // Return to original position
    StepInParameterSpace(model, direction, -flag_epsilon);

    // Compute second derivative
    second_derivative = fabs(gradient_in_direction_pos_step-gradient_in_direction) / flag_epsilon;

    // Negative step
    //StepInParameterSpace(model, direction, -2 * flag_epsilon);
    //EvaluateGradient(model, criterion, data, gradient_neg_step);
    //gradient_in_direction_neg_step = direction->iP(gradient_neg_step);
    // Return to original position
    //StepInParameterSpace(model, direction, flag_epsilon);
    // Compute second derivative
    //second_derivative = fabs(gradient_in_direction_pos_step-gradient_in_direction_neg_step) / (2*flag_epsilon);

    //std::cout << "variance (or 2nd moment): " << variance << " 2nd derivative: " << second_derivative << std::endl;
    std::cout << second_derivative << std::endl;
    fd_second_der << second_derivative << std::endl;
  } 

  fd_second_der.close();

  delete allocator;
  return(0);
}
