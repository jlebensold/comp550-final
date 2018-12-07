from utils import build_train_loader, build_test_loader
import numpy as np
import torch

ROUNDS = 10
EPOCHS = 2
CATEGORIES = ['EducationalInstitution', 'Artist', 'Company']
TEST_SET_SIZE_PER_CLASS = 1_000
WORKER_SET_SIZE_PER_CLASS = 2_000

class Federator:
    def __init__(self, workers, optimizer_factory, model_factory):
        self.workers = workers
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory

    def train_rounds(self):
        test_set = build_test_loader(test_categories=CATEGORIES, size=TEST_SET_SIZE_PER_CLASS)
        for comm_round in np.arange(0, ROUNDS):
            print("Round {}".format(comm_round))
            for epoch in np.arange(0, EPOCHS):
                for idx, worker in enumerate(self.workers):
                    worker.train_communication_round(build_train_loader(
                        train_categories=CATEGORIES,
                        size=WORKER_SET_SIZE_PER_CLASS),
                    epoch)
                    worker.valid_epoch(test_set, epoch)

            new_model = self.average_worker_models()
            for worker in self.workers:
                worker.model = new_model()
                worker.optimizer = self.optimizer_factory(worker.model)


    def average_worker_models(self):
        acc = {}
        averaged_weights = {}
        for worker in self.workers:
            for name, param in worker.model.named_parameters():
                if param.requires_grad:
                    if name not in acc.keys():
                        acc[name] = []
                    acc[name].append(param.data)
        for key, weights in acc.items():
            # average the weights across the models
            averaged = torch.stack(weights).mean(0)
            averaged_weights[key] = torch.nn.Parameter(averaged)


        def new_model_factory():
            averaged_model = self.model_factory()
            averaged_model.update_weights(averaged_weights)
            return averaged_model
        return new_model_factory
