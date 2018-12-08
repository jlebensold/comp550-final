# -*- coding: utf-8 -*-
"""
Original network described in

Network from https://github.com/1991viet/Character-level-cnn-pytorch
(MIT License)
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn
import torch
class CharacterLevelCNN(nn.Module):

    def update_weights(self, averaged_weights):
        """ here we initialize the weights from a custom dictionary """
        self.conv1[0].weight += averaged_weights['module.conv1.0.weight']
        self.conv1[0].bias += averaged_weights['module.conv1.0.bias']
        self.conv2[0].weight += averaged_weights['module.conv2.0.weight']
        self.conv2[0].bias += averaged_weights['module.conv2.0.bias']
        self.conv3[0].weight += averaged_weights['module.conv3.0.weight']
        self.conv3[0].bias += averaged_weights['module.conv3.0.bias']
        self.conv4[0].weight += averaged_weights['module.conv4.0.weight']
        self.conv4[0].bias += averaged_weights['module.conv4.0.bias']
        self.conv5[0].weight += averaged_weights['module.conv5.0.weight']
        self.conv5[0].bias += averaged_weights['module.conv5.0.bias']
        self.conv6[0].weight += averaged_weights['module.conv6.0.weight']
        self.conv6[0].bias += averaged_weights['module.conv6.0.bias']
        self.fc1[0].weight += averaged_weights['module.fc1.0.weight']
        self.fc1[0].bias += averaged_weights['module.fc1.0.bias']
        self.fc2[0].weight += averaged_weights['module.fc2.0.weight']
        self.fc2[0].bias += averaged_weights['module.fc2.0.bias']
        self.fc3.weight += averaged_weights['module.fc3.weight']
        self.fc3.bias += averaged_weights['module.fc3.bias']


    def __init__(self, n_classes=14, input_length=1014, input_dim=68,
                 n_conv_filters=256,
                 n_fc_neurons=1024):
        super(CharacterLevelCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))

        dimension = int((input_length - 96) / 27 * n_conv_filters)
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)

        if n_conv_filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif n_conv_filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):
#        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
