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
from constants import *

class Federator:
    def __init__(self, workers, optimizer_factory, model_factory, experiment="Undefined"):
        """ The Federator dispatches training tasks to each worker, simulating a
        server in the FederatedAveraging setting """

        self.workers = workers
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.experiment=experiment

        # federator keeps a copy of the model lying around
        # but does no training
        self.model = model_factory()
        time = datetime.now().strftime("%b_%d_%I_%M%S")
        self.writer = SummaryWriter('../training_logs/{}/{}/{}'.format(time,experiment,"Federator"))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.test_set, _ = build_test_loader(test_categories=CATEGORIES, size=TEST_SET_SIZE_PER_CLASS)

    def train_rounds(self, with_replacement=True):

        for comm_round in np.arange(0, ROUNDS):
            exclude_ids = []
            for idx, worker in enumerate(self.workers):
                train_loader, indexes = build_train_loader(worker.train_categories, WORKER_SET_SIZE_PER_CLASS, exclude_ids)
                worker.train_communication_round(train_loader, comm_round)

                worker.valid_comm_round(self.test_set, comm_round)

                if not with_replacement:
                    exclude_ids = np.concatenate([exclude_ids, indexes])

            averaged_weights  = self.average_worker_models()

            with torch.no_grad():
                # update the model for the federator 
                self.model.module.update_weights(averaged_weights)

                federator_weights = self.model.state_dict()
                self.valid_comm_round(self.test_set, comm_round)

                for worker in self.workers:
                    worker.model.load_state_dict(federator_weights)
        self.save_model_weights()

    def save_model_weights(self):
        """ Save a copy of the model parameters for future evaluation on cheap
        hardware """

        torch.save(self.model.state_dict(), "{}/{}_{}".format(MODEL_DIR,self.experiment,'Federator')) 
        for worker in self.workers:
            torch.save(worker.model.state_dict(), "{}/{}_{}".format(MODEL_DIR,self.experiment,worker.name)) 

    def valid_comm_round(self, test_loader, comm_round):
        """ Evaluate model performance for a given communication round """

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
                self.writer.add_scalar('validation_loss', loss.data.item(),           
                        iteration_number(comm_round, test_loader, idx))

        test_loss /= len(test_loader.dataset)

        acc = 100. * correct / len(test_loader.dataset)
        self.writer.add_scalar('validation_accuracy', acc, (comm_round * len(test_loader)))
        print('\nFederator Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), acc))

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


        return averaged_weights
