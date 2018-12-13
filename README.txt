Fall 2018 COMP 550: 
Group Number: 19

-------------------------------------------------
Team Members
-------------------------------------------------
Amir El Bawab (260645260)
Jonathan Maloney-Lebensold (260825605)


-------------------------------------------------
Introduction
-------------------------------------------------

Before building the federated learning environment for our model, we extracted all 14 subject classifications and replicated the DbPedia results using the same model architecture. We queried DbPedia using the SPARQL endpoint and created our own training and test sets (40,000 training, 5,000 test) in keeping with the cited paper above. 

Given the additional challenge of training multiple models in parallel, we elected to limit our classification task to 5 categories: Educational Institution, Artist, Company, Mean of Transportation and Office Holder. While there are many aspects to a federated learning pipeline (communication, client-drop off, stragglers, etc.), in the context of Natural Language Processing, decided to evaluate the affect of a non-I.I.D. sample set in a federated simulation environment.

One of the criticisms of Federated Learning is the time to convergence and the additional effort required to train a model. Convergence time is a critical metric for evaluating the quality of a machine learning pipeline. Being able to evaluate, train and test a model quickly is necessary when tuning model architectures and hyper-parameters.
Recent work presented at NeuRIPS 2018 on predictive text demonstrated that it took 4 weeks instead of 4 days for the same model to achieve equivalent convergence in a production setting with hundreds of mobile devices.

Our method involved experimenting with how a lack of data from certain DbPedia classes would affect model convergence, training loss and classification accuracy. This project also demonstrated that the same Federated Learning algorithms could be exploited beyond a binary classification setting (e.g. comment prediction with Google+ posts). Many algorithms have been proposed for federated learning, however we chose Federated Averaging since it seemed well suited to the affect of sharing gradients on neural network classification accuracy.

-------------------------------------------------
Dependencies
-------------------------------------------------
Python 3.6

- numpy
- pytorch
- torchvision
- tensorboardX
- matplotlib
- pickle
- fire

-------------------------------------------------
Code Files
-------------------------------------------------
- character_dataset.py
    - PyTorch dataloader for transforming SPARQL data into vectors
- char_cnn.py
    - Defines and initializes the Convolutional Neural Network model
- constants.py
    - Constants and hyperparameters used by the program and the model.
- data_preparation.py
    - Prepare and split the training and test data
- federator.py
    - The server model responsible for updating its weights by averaging out
    the weights from the workers/clients.
- main.py
    - Entry point to the program
- utils.py
    - Utility helper functions used by the application
- worker.py
    - The client model responsible for training on raw data and communicating
    the weights with the federator

-------------------------------------------------
Notes
-------------------------------------------------

The entry point is `main.py`. Different command-line arguments can be used to
run the experiments. Please refer to the README.txt in the `code/` directory for
information about executing the experiments.

The training_logs/ folder includes our raw data in Tensorboard format. You can
review these files by opening them in Tensorboard.
