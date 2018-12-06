from collections import Counter
import numpy as np
import torch
import torch.optim as optim

from datetime import datetime
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from models.symbol_dataset import SymbolDataset
from constants import *
from preprocess.utils import (load_struct,
        save_struct,
        valid,
        train,
        train_validation_split,
        write_csv_submission,
        prediction_misclassifications_by_label
)
from models.resnet import ResNet

# print("TacoNet Network (with PyTorch)")

class TorchHarness:
    def __init__(self, path, epochs, model_factory, model_name, batch_size):
        self.path = path
        self.epochs = epochs
        self.model_factory = model_factory
        self.model = None
        self.model_name = model_name
        self.batch_size = batch_size
        self.all_labels = []

    def train(self):
        all_labels, train_images, train_labels, test_images, test_labels = train_validation_split(self.path)
        train_transform = transforms.Compose([ transforms.ToTensor(), ])
        test_transform = transforms.Compose([ transforms.ToTensor(), ])

        dl_args = { 'batch_size': self.batch_size, 'shuffle': True, 'num_workers':1 }
        b1 = 0.9
        b2 = 0.999
        lr = 0.001
        epochs = self.epochs
        dataset_train = SymbolDataset(train_images, train_labels, all_labels=all_labels, transform=train_transform)
        dataset_test = SymbolDataset(test_images, test_labels, all_labels=all_labels, transform=test_transform)
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, **dl_args)
        valid_loader = torch.utils.data.DataLoader(dataset=dataset_test, **dl_args)

        torch.manual_seed(1)

        arch = self.model_name
        model = self.model_factory().to(device)

        # lr = learning rate
        suffix = "lr={},b1={},b2={},bsize={}".format(lr, b1, b2, dl_args['batch_size'])
        time = datetime.now().strftime("%I_%M%S_{}_".format(arch))

        writer = SummaryWriter('training_logs/{}_{}'.format(time,suffix))
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

        # after we hit > 75% accuracy, we begin saving model checkpoints:
        current_accuracy = 75.
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, writer)
            acc = valid(model, device, valid_loader, epoch, writer)
            if acc > current_accuracy:
                current_accuracy = acc
                torch.save(model, '{}.model.arch.acc.{}'.format(time,str(current_accuracy)))

        transform = transforms.Compose([ transforms.ToTensor(), ])
        prediction_loader = torch.utils.data.DataLoader(dataset=dataset_test)
        result = prediction_misclassifications_by_label(all_labels, model, device, prediction_loader)
        total = len(test_images)
        acc = (total - np.sum([freq for (k, freq) in result.most_common()]) )  / total
        print("Misclassification counts by label:")
        print(result.most_common())

        # required for writing the Kaggle CSV
        self.model = model
        self.all_labels = all_labels


