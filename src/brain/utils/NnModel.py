import numpy as np
import os
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """ Initialize the hidden layer weights with random noise. """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim

class NnModel(nn.Module):
    def __init__(self, config, inputs_size, seed):
        """Initialize parameters and build model. Initialize the torch functions contained in the defintion.
        
        Args:
            config (dict): model definiton.
            inputs_size (dict): Contains the sizes of input and output.
                state_size (int): Dimension of each state.
                action_size (int): Dimension of each action.
            seed (int): Random seed. """
        super(NnModel, self).__init__()
        self.config = config
        self.seed = torch.manual_seed(seed)
        if type(self.config) == str:
            with open(os.path.join(os.path.dirname(__file__),"..", "rl_agents",'predefined_models','{}.json'.format(self.config)), 'r') as j:
                self.config = json.load(j)

        self.layers = nn.ModuleList()
        for i, layer_config in enumerate(self.config["layers"]):
            # if i == 0:
            #     input_size = state_size
            # else:
            #     input_size = self.config[model_type]["layers"][i-1]["size"]

            if "size" in layer_config.keys() and type(layer_config["size"]) == str:
                    layer_config["size"] = inputs_size[layer_config["size"]]

            if layer_config["type"] == "BatchNorm1d":
                self.layers.append(nn.BatchNorm1d(layer_config["size"]))
            if layer_config["type"] == "linear":
                input_size = self.config["layers"][i-1]["size"]
                if "concat" in layer_config.keys():
                    for input in layer_config["concat"]:
                        input_size += inputs_size[input]
                self.layers.append(nn.Linear(input_size, layer_config["size"]))

            if layer_config["type"] == "conv2d":
                in_channels = self.config["layers"][i]["in_channels"]
                out_channels = self.config["layers"][i]["out_channels"]
                kernel_size = tuple(self.config["layers"][i]["kernel_size"])
                stride = tuple(self.config["layers"][i]["stride"])
                padding = tuple(self.config["layers"][i]["padding"])
                # size = ((in_channels - kernel_size + 2*padding)/stride) + 1
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))

            if layer_config["type"] == "rnn":
                input_size = self.config["layers"][i-1]["size"]
                hidden_size = self.config["layers"][i]["hidden_size"]
                num_layers = self.config["layers"][i]["num_layers"]
                nonlinearity = self.config["layers"][i]["nonlinearity"]
                self.config["layers"][i]["size"] = hidden_size
                self.layers.append(nn.RNN(input_size = input_size, hidden_size = hidden_size,
                                                     num_layers = num_layers, nonlinearity = nonlinearity,
                                                     batch_first = True))

            if layer_config["type"] == "maxpool2d":
                kernel_size = self.config["layers"][i]["kernel_size"]
                stride = self.config["layers"][i]["stride"]
                self.layers.append(nn.MaxPool2d(kernel_size, stride))

            if layer_config["type"] == "softmax":
                self.layers.append(nn.Softmax(dim= 1))

            if layer_config["type"] == "clamp": ### TODO
                self.layers.append(torch.clamp(min= 1))

        self.reset_parameters()

    def reset_parameters(self):
        """ Randomly itinialize the weights of the linear layers. """
        for i, layer_config in enumerate(self.config["layers"]):
            if layer_config["type"] == "linear":
                self.layers[i].weight.data.uniform_(*hidden_init(self.layers[i]))

    def forward(self, inputs):
        """ Process all the sequential operations that define the current model.
        There can be concatenations after the main operations of each layer.
        There can be auxiliar operations after the main operation.
        
        Args:
            inputs (dict): Current state and action. """
        x = inputs["state"]
        for i, layer_config in enumerate(self.config["layers"]):
            ## Prior concatenations
            if "concat" in layer_config.keys():
                for input in layer_config["concat"]:
                    x = torch.cat((x, inputs[input]), dim=1)

            ## Main layers
            if layer_config["type"] == "rnn":
                x, _ = self.layers[i](x)
            else:
                x = self.layers[i](x)

            ## Post processing
            if "features" in layer_config.keys():
                for feature in layer_config["features"]:
                    if feature == "unsqueeze":
                        x = x.unsqueeze(1)  ### TODO: enter as feature parameter
                    elif feature == "squeeze":
                        x = x.squeeze(1)  ### TODO: enter as feature parameter
                    elif feature == "leaky_relu":
                        x = F.leaky_relu(x)
                    elif feature == "relu":
                        x = F.relu(x)
                    elif feature == "tanh":
                        x = torch.tanh(x)
                    elif feature == "sigmoid":
                        x = torch.sigmoid(x)
                    elif feature == "flatten":
                        x = torch.flatten(x, 1)
            if "clip" in layer_config.keys():
                x = torch.clip(x, layer_config["clip"][0], layer_config["clip"][1])
        return x