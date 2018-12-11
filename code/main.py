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

from constants import *

def perform_federated_training(with_replacement, classes_per_worker, same_initilization):
    def optimizer_factory(model):
        return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def model_factory():
        return torch.nn.DataParallel(CharacterLevelCNN(), [0,1,2,3]).cuda()
        #return CharacterLevelCNN().cuda()

    worker_categories = [
        np.take(CATEGORIES,[0,1,2]),
        np.take(CATEGORIES,[0,1,3]),
        np.take(CATEGORIES,[0,1,4]),
        np.take(CATEGORIES,[0,2,3]),
        np.take(CATEGORIES,[0,2,4]),
        np.take(CATEGORIES,[0,3,4]),
        np.take(CATEGORIES,[1,2,3]),
        np.take(CATEGORIES,[1,3,4]),
        np.take(CATEGORIES,[2,3,4]),
        np.take(CATEGORIES,[2,1,4]),
    ]
    experiment= "wr.{}_cpw.{}_init.{}".format(with_replacement, classes_per_worker, same_initilization)
    def new_worker(idx):
        return Worker(name="W{}".format(idx), train_categories=worker_categories[idx],
                      experiment=experiment, 
                      model_factory=model_factory, 
                      optimizer_factory=optimizer_factory)


    NUM_WORKERS = 10
    workers = [new_worker(wid) for wid in np.arange(0,NUM_WORKERS)]

    federator = Federator(workers=workers, 
                         optimizer_factory=optimizer_factory, 
                         model_factory=model_factory,
                         experiment=experiment)
    federator.train_rounds(with_replacement, classes_per_worker, same_initilization)

if __name__ == '__main__':
    fire.Fire()
