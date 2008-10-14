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
#include "communicating_stacked_autoencoder.h"

//#include "Linear.h"
//#include "Tanh.h"
//#include "destructive.h"
//#include "nonlinear.h"
//#include "machines/identity.h"
//#include "transposed_tied_linear.h"
//#include <iostream>

namespace Torch {

CommunicatingStackedAutoencoder::CommunicatingStackedAutoencoder(std::string name_,
                                                                 std::string nonlinearity_,
                                                                 bool tied_weights_,
                                                                 int n_inputs_,
                                                                 int n_hidden_layers_,
                                                                 int *n_hidden_units_per_layer_,
                                                                 int n_outputs_,
                                                                 bool is_noisy_,
                                                                 int *n_speech_units_,
                                                                 int communication_type_
                                                                )
    : StackedAutoencoder( name_, nonlinearity_, tied_weights_, n_inputs_,
                          n_hidden_layers_, n_hidden_units_per_layer_, n_outputs_,
                          is_noisy_)
{
  communication_type = communication_type_;

  n_speech_units = (int*) allocator->alloc(sizeof(int)*(n_hidden_layers));
  for(int i=0; i<n_hidden_layers; i++)  {
    n_speech_units[i] = n_speech_units_[i];
  }

  // We're building what's needed for all 3 modes of communication, though
  // that is not necessary.

  BuildCommunicationCoders();
  BuildCommunicationAutoencoders();

  hidden_handles = (Identity**) allocator->alloc(sizeof(Identity*)*n_hidden_layers);
  speaker_handles = (Identity**) allocator->alloc(sizeof(Identity*)*n_hidden_layers);
  for(int i=0; i<n_hidden_layers; i++)  {
    hidden_handles[i] = new(allocator) Identity(encoders[i]->n_outputs);
    speaker_handles[i] = new(allocator) Identity(speakers[i]->n_outputs);
  }

  // The machine constructs
  sup_unsup_comA_machine = NULL;
  sup_unsup_comB_machine = NULL;
  sup_unsup_comC_machine = NULL;
  mentor = NULL;
  mentor_communicator = NULL;

  BuildSupUnsupComA();
  BuildSupUnsupComB();
  BuildSupUnsupComC();

  //BuildSupUnsupCsupCunsupMachine();
  //BuildMentor();

}

void CommunicatingStackedAutoencoder::BuildCommunicationCoders()
{
  // speakers
  speakers = (Coder**) allocator->alloc(sizeof(Coder*)*(n_hidden_layers));
  for(int i=0; i<n_hidden_layers; i++)    {
    speakers[i] = new(allocator)Coder(encoders[i]->n_outputs, n_speech_units[i],
                                      false, NULL, false, nonlinearity);
  }

  // noisy speaker
  if(is_noisy)  {
    noisy_speakers = (Coder**) allocator->alloc(sizeof(Coder*)*(n_hidden_layers));
    for(int i=0; i<n_hidden_layers; i++)    {
      noisy_speakers[i] = new(allocator)Coder(encoders[i]->n_outputs, n_speech_units[i],
                                              true, speakers[i], false, nonlinearity);
    }
  }     else    {
    noisy_speakers = NULL;
  }

  // listener
  listeners = (Coder**) allocator->alloc(sizeof(Coder*)*(n_hidden_layers));
  for(int i=0; i<n_hidden_layers; i++)    {
    if(tied_weights)      {
      listeners[i] = new(allocator)Coder(n_speech_units[i], encoders[i]->n_outputs,
                                     true, speakers[i], true, nonlinearity);
    }   else    {
      listeners[i] = new(allocator)Coder(n_speech_units[i], encoders[i]->n_outputs,
                                         false, NULL, false, nonlinearity);
    }
  }

}

void CommunicatingStackedAutoencoder::BuildCommunicationAutoencoders()
{
  speakerlisteners = (ConnectedMachine**) allocator->alloc(sizeof(ConnectedMachine*)*(n_hidden_layers));

  for(int i=0; i<n_hidden_layers; i++) {
    speakerlisteners[i] = new(allocator)ConnectedMachine();

    if(is_noisy)
      speakerlisteners[i]->addFCL(noisy_speakers[i]);
    else
      speakerlisteners[i]->addFCL(speakers[i]);

    speakerlisteners[i]->addFCL(listeners[i]);
    speakerlisteners[i]->build();
  }
}

// TODO there may be something lighter when not noisy if we don't always use
// the autoencoder but instead plug an identity machine and the listener in
// the speaker
void CommunicatingStackedAutoencoder::AddCommunicationMachines(ConnectedMachine *mch)
{
  for(int i=0; i<n_hidden_layers; i++) {
    mch->addMachine(speakers[i]);
    mch->connectOn(encoders[i]);

    mch->addMachine(speakerlisteners[i]);
    mch->connectOn(encoders[i]);
  }
}

void CommunicatingStackedAutoencoder::AddMachines(ConnectedMachine *mch,
                                                  GradientMachine **addees,
                                                  GradientMachine **connectees)
{
  for(int i=0; i<n_hidden_layers; i++) {
    mch->addMachine(addees[i]);
    mch->connectOn(connectees[i]);
  }
}

void CommunicatingStackedAutoencoder::BuildSupUnsupComA()
{
  sup_unsup_comA_machine = new(allocator) ConnectedMachine();
  AddCoreMachines(sup_unsup_comA_machine);

  sup_unsup_comA_machine->addMachine(outputer);
  sup_unsup_comA_machine->connectOn(encoders[n_hidden_layers-1]);

  AddUnsupMachines(sup_unsup_comA_machine);

  AddMachines(sup_unsup_comA_machine,
              (GradientMachine**) hidden_handles, (GradientMachine**) encoders);

  sup_unsup_comA_machine->build();
}

void CommunicatingStackedAutoencoder::BuildSupUnsupComB()
{
  sup_unsup_comB_machine = new(allocator) ConnectedMachine();

  AddCoreMachines(sup_unsup_comB_machine);

  sup_unsup_comB_machine->addMachine(outputer);
  sup_unsup_comB_machine->connectOn(encoders[n_hidden_layers-1]);

  AddUnsupMachines(sup_unsup_comB_machine);

  AddMachines(sup_unsup_comB_machine,
              (GradientMachine**) speakers, (GradientMachine**) encoders);

  sup_unsup_comB_machine->build();
}

void CommunicatingStackedAutoencoder::BuildSupUnsupComC()
{
  sup_unsup_comC_machine = new(allocator) ConnectedMachine();

  AddCoreMachines(sup_unsup_comC_machine);

  // We'll plug the listeners in the speakers. the speakers must be on a
  // lower layer.
  if(!is_noisy) {
    AddMachines(sup_unsup_comC_machine,
                (GradientMachine**) speakers, (GradientMachine**) encoders);
    sup_unsup_comC_machine->addLayer();

    sup_unsup_comC_machine->addMachine(outputer);
    sup_unsup_comC_machine->connectOn(encoders[n_hidden_layers-1]);

    AddUnsupMachines(sup_unsup_comC_machine);

    AddMachines(sup_unsup_comC_machine,
                (GradientMachine**) speaker_handles, (GradientMachine**) speakers);

    AddMachines(sup_unsup_comC_machine,
                (GradientMachine**) listeners, (GradientMachine**) speakers);
  }
  // Since we can't plug the listeners on the speakers, we'll not waste time
  // with identity handles, and not add a layer. Speakers directly on last
  // layer.
  else  {
    sup_unsup_comC_machine->addMachine(outputer);
    sup_unsup_comC_machine->connectOn(encoders[n_hidden_layers-1]);

    AddUnsupMachines(sup_unsup_comC_machine);

    AddMachines(sup_unsup_comC_machine,
                (GradientMachine**) speakers, (GradientMachine**) encoders);

    AddMachines(sup_unsup_comC_machine,
                (GradientMachine**) speakerlisteners, (GradientMachine**) encoders);
  }
  sup_unsup_comC_machine->build();
}

/*
void CommunicatingStackedAutoencoder::BuildSupUnsupCsupCunsupMachine()
{
  sup_unsup_csup_cunsup_machine = new(allocator) ConnectedMachine();

  AddCoreMachines(sup_unsup_csup_cunsup_machine);

  // Add the output - Can't do FCL because if only 1 layer, then
  // there might be an identity layer.
  sup_unsup_csup_cunsup_machine->addMachine(outputer);
  sup_unsup_csup_cunsup_machine->connectOn(encoders[n_hidden_layers-1]);

  // add the hidden reconstruction and communicating components
  AddUnsupMachines(sup_unsup_csup_cunsup_machine);
  AddCommunicationMachines(sup_unsup_csup_cunsup_machine);

  sup_unsup_csup_cunsup_machine->build();
}
*/

void CommunicatingStackedAutoencoder::BuildMentor()
{
  // Mentor
  mentor = new(allocator) ConnectedMachine();

  // Construct the core machine
  // we could use AddCoreMachines, but we don't need the
  // identity machine that would be put on the first layer
  for(int i=0; i<n_hidden_layers; i++)    {
    mentor->addMachine(encoders[i]);
    // connect it if not the first layer
    if(i>0)     {
      mentor->connectOn(encoders[i-1]);
    }
    mentor->addLayer();
  }

  // These are the sole outputs
  AddCommunicationMachines(mentor);

  mentor->build();

  // Mentor communicator
  mentor_communicator  = new(allocator) ConnectedMachine();
  for(int i=0; i<n_hidden_layers; i++)    {
    mentor_communicator->addMachine(speakers[i]);
    mentor_communicator->addMachine(speakerlisteners[i]);
  }
  mentor_communicator->build();

}

void CommunicatingStackedAutoencoder::setL1WeightDecay(real weight_decay)
{
  StackedAutoencoder::setL1WeightDecay(weight_decay);
  warning("CommunicatingStackedAutoencoder::setL1WeightDecay - fixme");
}

void CommunicatingStackedAutoencoder::setL2WeightDecay(real weight_decay)
{
  StackedAutoencoder::setL2WeightDecay(weight_decay);
  warning("CommunicatingStackedAutoencoder::setL2WeightDecay - fixme");
}

void CommunicatingStackedAutoencoder::setDestructionOptions(real destruct_prob, real destruct_value)
{
  StackedAutoencoder::setDestructionOptions(destruct_prob, destruct_value);
  warning("CommunicatingStackedAutoencoder::setDestructionOptions - fixme");
}

void CommunicatingStackedAutoencoder::loadXFile(XFile *file)
{
  sup_unsup_comC_machine->loadXFile(file);
}

void CommunicatingStackedAutoencoder::saveXFile(XFile *file)
{
  sup_unsup_comC_machine->saveXFile(file);
}

CommunicatingStackedAutoencoder::~CommunicatingStackedAutoencoder()
{
}

}
