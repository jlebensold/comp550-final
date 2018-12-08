import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from char_cnn import CharacterLevelCNN
from datetime import datetime
from utils import build_train_loader, build_test_loader, valid_epoch, train_epoch
from federator import Federator
from worker import Worker
import time

def perform_federated_training():
    def optimizer_factory(model):
        return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def model_factory():
        return torch.nn.DataParallel(CharacterLevelCNN(), [0,1,2,3]).cuda()
    model1 = model_factory()
    model2 = model_factory()
    model3 = model_factory()

    # lets maintain the same initialization state for all models:
    model2.load_state_dict(model1.state_dict())
    model3.load_state_dict(model1.state_dict())
    workers = [
            Worker(name="Jane", cuda_num=0, model=model1, optimizer=optimizer_factory(model1)),
            Worker(name="Sally", cuda_num=0, model=model2, optimizer=optimizer_factory(model2)),
            Worker(name="Bob", cuda_num=0, model=model3, optimizer=optimizer_factory(model3))
            ]
    federator = Federator(workers=workers, optimizer_factory=optimizer_factory, model_factory=model_factory)
    federator.train_rounds()

if __name__ == '__main__':
    fire.Fire()
