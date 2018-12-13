import matplotlib.pyplot as plt
from torchviz import make_dot, make_dot_from_trace
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import iteration_number
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime

class Worker:
    def __init__(self, name, model_factory, optimizer_factory, train_categories, experiment="Undefined"):
        """ The worker encapsulates model creation and simulates a user in the
        federated context"""

        self.name = name
        self.model = model_factory()
        self.optimizer = optimizer_factory(self.model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.current_model_weights = {}
        time = datetime.now().strftime("%b_%d_%I_%M%S")
        self.writer = SummaryWriter('../training_logs/{}/{}/{}'.format(time,experiment,self.name))
        self.train_categories = train_categories

    def train_communication_round(self, train_loader, comm_round):
        """ Training loop """

        self.store_model_weights()
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('training_loss', loss.data.item(),  iteration_number(comm_round, train_loader, batch_idx))
            if batch_idx % 10 == 0:
                print('{}: Train Comm Round: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.name,
                    comm_round,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item())
                )
        self.update_params_delta()

    def store_model_weights(self):
        """ store the weights for calculating a delta before training begins """

        for name, param in self.model.named_parameters():
            self.current_model_weights[name] = param.clone()


    def update_params_delta(self):
        """ subtract the current model parameters from the updated model parameters"""

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.current_model_weights.keys():
                       self.current_model_weights[name] =  param - self.current_model_weights[name]


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
                self.writer.add_scalar('val_loss', loss.data.item(),
                        iteration_number(comm_round, test_loader, idx))

        test_loss /= len(test_loader.dataset)

        acc = 100. * correct / len(test_loader.dataset)
        self.writer.add_scalar('val_acc', acc, (comm_round * len(test_loader)))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), acc))
