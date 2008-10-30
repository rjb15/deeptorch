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
bin_sae_weights\n\
\n\
This program will use binners to represent a sae's weights and bias\n\
 marginals. The binners are saved.\n";

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
#include "binner.h"

using namespace Torch;

// ************
// *** MAIN ***
// ************
int main(int argc, char **argv)
{

  // === The command-line ===

  char *flag_model_filename;

  int flag_n_bins;
  char *flag_outdir;
  bool flag_binary_mode;

  // Construct the command line
  CmdLine cmd;

  // Put the help line at the beginning
  cmd.info(help);

  cmd.addText("\nArguments:");
  cmd.addSCmdArg("-model_filename", &flag_model_filename, "the model filename");

  cmd.addText("\nOptions:");
  cmd.addICmdOption("-n_bins", &flag_n_bins, 10, "Number of bins to use.", true);
  cmd.addSCmdOption("-outdir", &flag_outdir, "./", "Location where to save the binners.", true);
  cmd.addBCmdOption("binary_mode", &flag_binary_mode, false, "binary mode for files", true);

  // Read the command line
  cmd.read(argc, argv);

  Allocator *allocator = new Allocator;

  std::string str_outdir = flag_outdir;
  if(str_outdir != "./")   {
    warning("Calling non portable mkdir!");
    std::stringstream command;
    command << "mkdir " << flag_outdir;
    system(command.str().c_str());
  }

  // model
  CommunicatingStackedAutoencoder *csae = LoadCSAE(allocator, flag_model_filename);

  // binners
  Binner **w_binners = (Binner**) allocator->alloc(sizeof(Binner*)*csae->n_hidden_layers);
  Binner **b_binners = (Binner**) allocator->alloc(sizeof(Binner*)*csae->n_hidden_layers);

  // Do it!
  int n_samples;
  real *samples;
  std::stringstream filename;
  XFile *the_xfile;

  for (int i=0; i<csae->n_hidden_layers; i++) {
    Linear *linear_layer = csae->encoders[i]->linear_layer;
    real *weights_ = linear_layer->weights;
    real *bias_ = linear_layer->bias;

    // *** weights ***
    // Copy the weights. Binner will sort them.
    n_samples = linear_layer->n_outputs * linear_layer->n_inputs;
    samples = (real*) allocator->alloc(sizeof(real)*n_samples);
    memcpy(samples, weights_, sizeof(real)*n_samples);

    // Bin
    w_binners[i] = new(allocator) Binner();
    w_binners[i]->init(flag_n_bins, n_samples, samples);
    delete samples;    

    // Save the binner
    filename.str("");
    filename.clear();
    filename << flag_outdir << "binner_w" << i << ".save";
    the_xfile = new(allocator) DiskXFile(filename.str().c_str(), "r");
    w_binners[i]->saveXFile(the_xfile);
    delete the_xfile;
    
    // *** Do the same for the biases ***
    // Copy the weights. Binner will sort them.
    n_samples = linear_layer->n_outputs;
    samples = (real*) allocator->alloc(sizeof(real)*n_samples);
    memcpy(samples, bias_, sizeof(real)*n_samples);

    // Bin
    b_binners[i] = new(allocator) Binner();
    b_binners[i]->init(flag_n_bins, n_samples, samples);
    delete samples;    

    // Save the binner
    filename.str("");
    filename.clear();
    filename << flag_outdir << "binner_b" << i << ".save";
    the_xfile = new(allocator) DiskXFile(filename.str().c_str(), "r");
    w_binners[i]->saveXFile(the_xfile);
    delete the_xfile;
  }

  delete allocator;
  return(0);

}
