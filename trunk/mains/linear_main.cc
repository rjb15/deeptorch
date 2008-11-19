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
linear_main\n\
\n\
This program will train a linear model with a squared error criterion.\n";

#include <string>
#include <sstream>

#include "Allocator.h"
#include "Random.h"
#include "DiskXFile.h"
#include "CmdLine.h"

#include "MeanVarNorm.h"
#include "MatDataSet.h"
#include "ClassFormatDataSet.h"
#include "OneHotClassFormat.h"
#include "Measurer.h"
#include "MSEMeasurer.h"
#include "ClassMeasurer.h"
#include "ClassNLLMeasurer.h"
#include "Trainer.h"         // for MeasurerList!
#include "ClassNLLCriterion.h"
#include "MSECriterion.h"
#include "ConnectedMachine.h"
#include "Linear.h"

#include "coder.h"
#include "input_as_target_data_set.h"
#include "dynamic_data_set.h"
#include "stacked_autoencoder.h"
#include "communicating_stacked_autoencoder.h"
#include "stacked_autoencoder_trainer.h"
#include "helpers.h"
#include "binner.h"


using namespace Torch;

// ************
// *** MAIN ***
// ************
int main(int argc, char **argv)
{

  //=================== The command-line ==========================

  char *flag_expdir_prefix;

  // --- Task ---
  char *flag_task;
  int flag_n_inputs;
  int flag_n_classes;
  char *flag_train_data_file;
  char *flag_valid_data_file;
  char *flag_test_data_file;

  // --- Training ---
  int flag_max_iter;
  real flag_accuracy;
  real flag_lr;
  real flag_lrate_decay;
  real flag_l1_decay;
  real flag_l2_decay;
  real flag_bias_decay;
  bool flag_criter_avg_framesize;

  // --- Stuff ---
  int flag_start_seed;
  int flag_model_seed;
  int flag_max_load;
  int flag_max_train_load;
  bool flag_binary_mode;
  bool flag_save_model;
  bool flag_single_results_file;
  bool flag_multiple_results_files;

  // Construct the command line
  CmdLine cmd;

  // Put the help line at the beginning
  cmd.info(help);
  cmd.addSCmdOption("-expdir_prefix", &flag_expdir_prefix, "./", "Location where to write the expdir folder.", true);

  // Task
  cmd.addText("\nTask Arguments:");
  cmd.addSCmdArg("-task", &flag_task, "name of the task");
  cmd.addICmdArg("-n_inputs", &flag_n_inputs, "number of inputs");
  cmd.addICmdArg("-n_classes", &flag_n_classes, "number of targets");
  cmd.addSCmdArg("-train_data_file", &flag_train_data_file, "name of the training file");
  cmd.addSCmdArg("-valid_data_file", &flag_valid_data_file, "name of the valid file");
  cmd.addSCmdArg("-test_data_file", &flag_test_data_file, "name of the test file");

  // Training
  cmd.addText("\nTraining options:");
  cmd.addICmdOption("-max_iter", &flag_max_iter, 2, "max number of iterations with only supervised cost (4th phase)", true);
  cmd.addRCmdOption("-accuracy", &flag_accuracy, 1e-5, "end accuracy", true);
  cmd.addRCmdOption("-lr", &flag_lr, 1e-3, "learning rate layerwise unsup phase", true);
  cmd.addRCmdOption("-lrate_decay", &flag_lrate_decay, 0.0, "learning rate decay", true);
  cmd.addRCmdOption("-l1_decay", &flag_l1_decay, 0.0, "l1 weight decay", true);
  cmd.addRCmdOption("-l2_decay", &flag_l2_decay, 0.0, "l2 weight decay", true);
  cmd.addRCmdOption("-bias_decay", &flag_bias_decay, 0.0, "bias decay", true);
  cmd.addBCmdOption("-criter_avg_framesize", &flag_criter_avg_framesize, false, "if true, costs of unsup criterions are divided by number of inputs", true);

  // Stuff
  cmd.addICmdOption("start_seed", &flag_start_seed, 1, "the random seed used in the beginning (-1 to for random seed)", true);
  cmd.addICmdOption("model_seed", &flag_model_seed, 2, "the random seed used just before model initialization (-1 to for random seed)", true);
  cmd.addICmdOption("max_load", &flag_max_load, -1, "max number of examples to load for valid and test", true);
  cmd.addICmdOption("max_train_load", &flag_max_train_load, -1, "max number of examples to load for train", true);
  cmd.addBCmdOption("binary_mode", &flag_binary_mode, false, "binary mode for files", true);
  cmd.addBCmdOption("save_model", &flag_save_model, true, "if true, save the model", true);
  cmd.addBCmdOption("single_results_file", &flag_single_results_file, false, "if true, saves the results into a single file (1 for sup, 1 for unsup, 1 for supunsup)", true);
  cmd.addBCmdOption("multiple_results_files", &flag_multiple_results_files, true, "if true, save results into different files, depending on the cost", true);

  // Read the command line
  cmd.read(argc, argv);

  Allocator *allocator = new Allocator;

  std::string str_train_data_file = flag_train_data_file;
  std::string str_valid_data_file = flag_valid_data_file;
  std::string str_test_data_file = flag_test_data_file;

  // Formats the expdir name, where the results and models will be saved
  std::stringstream ss;
  ss << flag_expdir_prefix << "linear-task=" << flag_task
     << "-ne=" << flag_max_iter
     << "-lr=" << flag_lr 
     << "-dc=" << flag_lrate_decay << "-l1=" << flag_l1_decay
     << "-l2=" << flag_l2_decay << "-bdk=" << flag_bias_decay
     << "-cFs=" << flag_criter_avg_framesize
     << "-ss=" << flag_start_seed << "-ms=" << flag_model_seed;

  if (flag_multiple_results_files)
     ss << "/";
  else
     ss << "_";

  std::string expdir = ss.str();

  if (!flag_single_results_file)        {
    warning("Calling non portable mkdir!");
    std::string command = "mkdir " + expdir;
    system(command.c_str());
  }

  // To be changed if you want reproducible results for operations that use
  // random numbers BEFORE instantiating the models.
  if(flag_start_seed == -1)
    Random::seed();
  else
    Random::manualSeed((long)flag_start_seed);

  // === Create the DataSets ===
  MatDataSet train_matdata(flag_train_data_file, flag_n_inputs,1,false,
                                 flag_max_train_load, flag_binary_mode);
  MatDataSet valid_matdata(flag_valid_data_file, flag_n_inputs,1,false,
                                 flag_max_load, flag_binary_mode);
  MatDataSet test_matdata(flag_test_data_file, flag_n_inputs,1,false,
                                flag_max_load, flag_binary_mode);
  message("Data loaded\n");
  message("Data was loaded as is and was NOT normalized\n");

  ClassFormatDataSet train_data(&train_matdata,flag_n_classes);
  ClassFormatDataSet valid_data(&valid_matdata,flag_n_classes);
  ClassFormatDataSet test_data(&test_matdata,flag_n_classes);

  OneHotClassFormat class_format(&train_data);


  // === Create the model ===

  // Seed before model init. 
  if(flag_model_seed == -1)
    Random::seed();
  else
    Random::manualSeed((long)flag_model_seed);
  
  // Last two parameters: communication type and n_communication_layers
  //Coder model(flag_n_inputs, flag_n_classes, false, NULL, false, false, "logsoftmax");
  Coder model(flag_n_inputs, flag_n_classes, false, NULL, false, false, "none");

  model.linear_layer->setROption("l1 weight decay", flag_l1_decay);
  model.linear_layer->setROption("weight decay", flag_l2_decay);
  model.linear_layer->setROption("bias decay", flag_bias_decay);

  message("Model instanciated.\n");

  // === Measurers ===
  MeasurerList measurers;
  //AddClassificationMeasurers(allocator, expdir, &measurers, &model,
  //                           &train_data, &valid_data, &test_data,
  //                           &class_format, flag_multiple_results_files);
  std::stringstream ss_filename;

  // train
  ss_filename.str("");
  ss_filename.clear();
  ss_filename << expdir << "train_mse.txt";
  DiskXFile *file_train_mse = new(allocator) DiskXFile(ss_filename.str().c_str(),"w");
  MSEMeasurer *measurer_train_mse = new(allocator) MSEMeasurer(model.outputs, &train_data, file_train_mse);
  measurers.addNode(measurer_train_mse);

  // valid
  ss_filename.str("");
  ss_filename.clear();
  ss_filename << expdir << "valid_mse.txt";
  DiskXFile *file_valid_mse = new(allocator) DiskXFile(ss_filename.str().c_str(),"w");
  MSEMeasurer *measurer_valid_mse = new(allocator) MSEMeasurer(model.outputs, &valid_data, file_valid_mse);
  measurers.addNode(measurer_valid_mse);

  // test
  ss_filename.str("");
  ss_filename.clear();
  ss_filename << expdir << "test_mse.txt";
  DiskXFile *file_test_mse = new(allocator) DiskXFile(ss_filename.str().c_str(),"w");
  MSEMeasurer *measurer_test_mse = new(allocator) MSEMeasurer(model.outputs, &test_data, file_test_mse);
  measurers.addNode(measurer_test_mse);

  // === Criterion ===
  //ClassNLLCriterion supervised_criterion(&class_format);
  warning("Using MSE criterion!");
  MSECriterion criterion(model.n_outputs);

  // === Train the csae ===
  StochasticGradientPlus trainer(&model, &criterion, NULL);

  trainer.setROption("end accuracy", flag_accuracy);
  trainer.setROption("learning rate decay", flag_lrate_decay);

  DiskXFile* resultsfile = NULL;
  
  if(flag_save_model) {
    SaveCoder(expdir, "linear-after-init.save", &model);
  }

  // --- train with only supervised cost ---
  
  if (flag_max_iter) {
    trainer.setROption("learning rate", flag_lr);
    trainer.setIOption("max iter", flag_max_iter);
 
    if (flag_single_results_file) {
      resultsfile = InitResultsFile(allocator,expdir,"sup");
      trainer.resultsfile = resultsfile;
    }

    trainer.train(&train_data, &measurers);
  }

  // === Save model ===
  if(flag_save_model) {
    SaveCoder(expdir, "linear-final.save", &model);
  }

  delete allocator;
  return(0);
}
