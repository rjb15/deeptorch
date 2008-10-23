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
sae_visualizer\n\
\n\
This program will load a model output information useful for\n\
visualizing it: the weights matrices as well as hidden unit\n\
activities on the given dataset.\n\
\n";

#include <string>
#include <sstream>

#include "CmdLine.h"
#include "Allocator.h"
#include "ClassFormatDataSet.h"
#include "OneHotClassFormat.h"
#include "Measurer.h"
#include "MSEMeasurer.h"
#include "ClassMeasurer.h"
#include "ClassNLLMeasurer.h"
#include "MatDataSet.h"
#include "DiskXFile.h"
#include "helpers.h"


using namespace Torch;

// ************
// *** MAIN ***
// ************
int main(int argc, char **argv)
{

  // === The command-line ===

  int flag_n_inputs;
  int flag_n_classes;
  char *flag_expdir;
  char *flag_testdata_filename;
  int flag_is_tied;

  char *flag_task;
  int flag_max_load;
  bool flag_binary_mode;

  // Construct the command line
  CmdLine cmd;

  // Put the help line at the beginning
  cmd.info(help);

  cmd.addText("\nArguments:");

  cmd.addICmdArg("-n_inputs", &flag_n_inputs, "number of inputs");
  cmd.addICmdArg("-n_classes", &flag_n_classes, "number of targets");
  cmd.addSCmdArg("-expdir", &flag_expdir, "location where to find model.save and write out.");
  cmd.addSCmdArg("-testdata_filename", &flag_testdata_filename, "name of the test file");
  cmd.addICmdArg("-is_tied", &flag_is_tied, "Are the weights tied?");

  cmd.addText("\nOptions:");
  cmd.addSCmdOption("-task", &flag_task, "", "name of the task", true);
  cmd.addICmdOption("max_load", &flag_max_load, -1, "max number of examples to load for train", true);
  cmd.addBCmdOption("binary_mode", &flag_binary_mode, false, "binary mode for files", true);

  // Read the command line
  cmd.read(argc, argv);

  Allocator *allocator = new Allocator;
  warning("Assuming tied_weights=%d", flag_is_tied);

  // Ensure directory structure exists
  std::string str_expdir = flag_expdir;

  warning("Calling non portable mkdir!");
  std::stringstream command;

  command << "mkdir " << str_expdir << "visualization/";
  system(command.str().c_str());

  command.str("");
  command.clear();
  std::string str_weights_dir = str_expdir + "visualization/weights/"; 
  command << "mkdir " << str_weights_dir; 
  system(command.str().c_str());

  command.str("");
  command.clear();
  std::string str_representations_dir = str_expdir + "visualization/representations/"; 
  command << "mkdir " << str_representations_dir; 
  system(command.str().c_str());

  // data
  MatDataSet test_matdata(flag_testdata_filename, flag_n_inputs,1,false,
                                flag_max_load, flag_binary_mode);
  ClassFormatDataSet test_data(&test_matdata,flag_n_classes);
  OneHotClassFormat class_format(&test_data);   // Not sure about this... what if not all classes were in the test set?

  // model
  std::string model_filename = str_expdir + "model.save";
  CommunicatingStackedAutoencoder *csae = LoadCSAE(allocator, model_filename);

  // Output the weight matrices
  saveWeightMatrices(csae, str_weights_dir, flag_is_tied);

  // Produce representations
  saveRepresentations(csae, str_representations_dir, &test_data, 100);

  delete allocator;
  return(0);
}
