from utils import build_train_loader, build_test_loader
from utils import iteration_number
import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
import threading

ROUNDS = 20
CATEGORIES = ['EducationalInstitution', 'Artist', 'Company', 'MeanOfTransportation', 'OfficeHolder']
TEST_SET_SIZE_PER_CLASS = 1_000
WORKER_SET_SIZE_PER_CLASS = 5_000

class Federator:
    def __init__(self, workers, optimizer_factory, model_factory, experiment="Undefined"):
        self.workers = workers
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory

        # federator keeps a copy of the model lying around
        # but does no training
        self.model = model_factory()
        time = datetime.now().strftime("%I_%M%S_{}".format(experiment))
        self.writer = SummaryWriter('../training_logs/{}/{}'.format("Federator", time))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_rounds(self, with_replacement, classes_per_worker, same_initilization):
        test_set, indexes= build_test_loader(test_categories=CATEGORIES, size=TEST_SET_SIZE_PER_CLASS)


        for comm_round in np.arange(0, ROUNDS):
            exclude_ids = []
            for idx, worker in enumerate(self.workers):
                train_categories = np.random.choice(CATEGORIES, classes_per_worker, replace=False)
                train_loader, indexes = build_train_loader(train_categories, WORKER_SET_SIZE_PER_CLASS, exclude_ids)
                worker.train_communication_round(train_loader, comm_round)
                worker.valid_comm_round(test_set, comm_round)

                if not with_replacement:
                    exclude_ids = np.concatenate([exclude_ids, indexes])

            new_model = self.average_worker_models()

            # update the model for the federator 
            self.model = new_model()
            self.valid_comm_round(test_set, comm_round)



            for worker in self.workers:
                worker.model = new_model()
                worker.optimizer = self.optimizer_factory(worker.model)

    def valid_comm_round(self, test_loader, comm_round):
        self.model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target) 
                test_loss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
                self.writer.add_scalar('valid/loss', loss.data.item(),           
                        iteration_number(comm_round, test_loader, idx))

        test_loss /= len(test_loader.dataset)

        acc = 100. * correct / len(test_loader.dataset)
        self.writer.add_scalar('valid/accuracy', acc, (comm_round * len(test_loader)))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), acc))

    def average_worker_models(self):
        acc = {}
        averaged_weights = {}
        for worker in self.workers:
            for name, param in worker.current_model_weights.items():
                if name not in acc.keys():
                    acc[name] = []
                acc[name].append(param.data)
        for key, weights in acc.items():
            # average the weights across the models
            averaged = torch.stack(weights).mean(0)
            averaged_weights[key] = torch.nn.Parameter(averaged)


        def new_model_factory():
            with torch.no_grad():
                self.model.module.update_weights(averaged_weights)
            return self.model
        return new_model_factory
