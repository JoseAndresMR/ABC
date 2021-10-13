import numpy as np
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    # for initializing the hidden layer weights with random noise
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.bn1(state)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128+action_size, 128)
        self.fc3 = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.bn1(state)
        xs = F.leaky_relu(self.fcs1(xs))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3([[ 1. ,        -0.92331517 , 0.98709023 ,-0.92626536]])

class NnModel(nn.Module):
    def __init__(self, config, inputs_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(NnModel, self).__init__()
        self.config = config
        self.seed = torch.manual_seed(seed)
        if type(self.config) == str:
            with open(os.path.join(os.path.dirname(__file__),'predefined_models.json'), 'r') as j:
                self.config = json.load(j)[self.config]

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

            if layer_config["type"] == "maxpool2d":
                kernel_size = self.config["layers"][i]["kernel_size"]
                stride = self.config["layers"][i]["stride"]
                self.layers.append(nn.MaxPool2d(kernel_size, stride))

            if layer_config["type"] == "softmax":
                self.layers.append(nn.Softmax(dim= 1))

        self.reset_parameters()

    def reset_parameters(self):
        for i, layer_config in enumerate(self.config["layers"]):
            if layer_config["type"] == "linear":
                self.layers[i].weight.data.uniform_(*hidden_init(self.layers[i]))

            # self.fc3.weight.data.uniform_(-3e-3, 3e-3)  #### How to implement this?

    def forward(self, inputs):
        # inputs = {"state" : state, "action" : action}
        # x = self.bn1(inputs["state"])
        x = inputs["state"]
        for i, layer_config in enumerate(self.config["layers"]):
            if "concat" in layer_config.keys():
                for input in layer_config["concat"]:
                    x = torch.cat((x, inputs[input]), dim=1)
            x = self.layers[i](x)
            if "features" in layer_config.keys():
                for feature in layer_config["features"]:
                    if feature == "leaky_relu":
                        x = F.leaky_relu(x)
                    if feature == "flatten":
                        x = torch.flatten(x, 1)
        return x