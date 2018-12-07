from utils import build_train_loader, build_test_loader
import numpy as np
import torch

ROUNDS = 200
CATEGORIES = ['EducationalInstitution', 'Artist']
TEST_SET_SIZE_PER_CLASS = 1_000
WORKER_SET_SIZE_PER_CLASS = 2_000

class Federator:
    def __init__(self, workers, optimizer_factory, model_factory):
        self.workers = workers
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory

        # federator keeps a copy of the model lying around
        # but does no training
        self.model = model_factory()

    def train_rounds(self):
        test_set = build_test_loader(test_categories=CATEGORIES, size=TEST_SET_SIZE_PER_CLASS)


        for comm_round in np.arange(0, ROUNDS):
            for idx, worker in enumerate(self.workers):
                train_loader = build_train_loader(train_categories=CATEGORIES, size=WORKER_SET_SIZE_PER_CLASS)
                worker.train_communication_round(train_loader, comm_round)
                worker.valid_comm_round(test_set, comm_round)

            new_model = self.average_worker_models()

            for worker in self.workers:
                worker.model = new_model()
                worker.optimizer = self.optimizer_factory(worker.model)

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
                self.model.update_weights(averaged_weights)
            return self.model
        return new_model_factory
