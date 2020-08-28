"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
import sklearn
import string as s
import re as r


###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################
def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # remove illegal characters
    sample = [r.sub(r'[^\x00-\x7f]', r'', w) for w in sample]
    # remove punctuations
    sample = [w.strip(s.punctuation) for w in sample]
    # remove numbers
    sample = [r.sub(r'[0-9]', r'', w) for w in sample]
    return sample


def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    wordfreq = vocab.freqs
    sum1 = 0
    for v in wordfreq.values():
        sum1 += v
    avgfreq = sum1 / len(wordfreq)
    worditos = vocab.itos
    for (i, x) in enumerate(batch):
        for (j, y) in enumerate(x):
            if 2 >= wordfreq[worditos[y]]:
                x[j] = 0
    return batch


stopWords = {'this', 'item', 'for', 'a', 'i', 'xbox', 'who', 'was', 'game', 'it', 'and', 'to', 'the', 'are',
             'who', 'was', 'game', 'its', 'his', 'to', 'her', 'him', 'being', 'make', 'now', 'actually', 'an',
             'on', 'is', 'son', 'thing', 'games', 'product', '3ds', 'mouse', 'pc', 'psn', 'rpg', 'ps3',
             'controller', 'play', 'graphics', 'nintendo', 'wii', 'xbox360', "i'm", 'graphic', 'played',
             'years', 'playing', 'sound', 'sounds', 'first', 'you', 'of', 'that', "it's", 'there', 'were', 'from',
             'they', 'my', 'your', 'microsoft', 'ssbb', 'lsl', 'edit', 'amazon', 'into', 'gameplay', "that's"
             'been', 'does', "i've", 'then', 'series', 'ever', 'version', 'online', 'put', "you're", 'player'
             'gets', 'got', 'things', 'had', 'be', 'screen', 'found', 'something', 'system', 'by', 'know', 'one'
             'these', 'use', 'about', 'when', 'from', 'bought', 'purchase', 'buy', 'in', 'hours', 'months', 'need',
             'what', 'with', 'am'
             }

cladim = 300
wordVectors = GloVe(name='6B', dim=cladim)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################


def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    return datasetLabel.long() - 1


def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    return (1 + torch.argmax(netOutput, 1)).float()


###########################################################################
################### The following determines the model ####################
###########################################################################


class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """
    # LSTM (bidirectional) -> Linear1 -> Relu -> Linear2 -> output
    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(input_size=cladim, hidden_size=50, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.Linear1 = tnn.Linear(200, 64)
        self.Relu = tnn.ReLU()
        self.Linear2 = tnn.Linear(64, 5)

    def forward(self, input, length):
        out, (h_n, c_n) = self.lstm(input)
        x = torch.cat((out[:, -1, :], out[:, 0, :]), dim=1)
        x = self.Linear1(x)
        x = self.Relu(x)
        x = self.Linear2(x)
        x = x.squeeze()
        return x


class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """
    def __init__(self):
        super(loss, self).__init__()
        self.losscla = tnn.CrossEntropyLoss()

    def forward(self, output, target):
        claresult = self.losscla(output, target)
        return claresult


net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
lossFunc = loss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.005)
