# COMP9444 Assignment 2: Rating Prediction
# Group ID: g022415
# Group members: Junyu Ren (z5195715)
#                Wentao Zou (z5229938)
# Date: 2020 / 08 / 08
# ----------------------------------------------------------------------------------------------------
# Answer to the Question:
'''
For the pre-processing, we remove illegal characters using python standard libraries re, then we
select some high-freq stopWords and convert 1-5 numbers into english words. Lastly, we apply stopWords
and remove punctuations using python standard library string. For postprocessing, we don't think any
processing steps is required. We choose word dimension of 300, we found that word dimension 200 and
300 are both good values as a network input dimension.

We eventually decided to do regression. Before that we also implement a version of classification
and found regression is slightly better than classification with other parameters the same.
For regression, there is no need to do more in convertLabel since label themselves are float values.
However, something should be done in convertNetOutput. Regression's output can be any demicals
 and may exceed the range [1.0, 5.0], so we should clamp them into range [1.0, 5.0] and use torch.round().

For network structure, we use LSTM -> linear1 -> relu -> linear2 -> output. We use bidirectional LSTM
with dropout rate 0.5 in case of over-fitting. 0.5 is a good value after several attempts. In terms
of loss function, we firstly tried MSE and MAE, which are common loss functions for regression questions
but they didn't perform well here. We then implemented our own loss function, we call it 'weighted MAE'.
We give the network different reward (different weights) about correct and one-star away items, 0.2 and
0.98 respectively. Compared with reward weight 0.2 and 0.98, adding the loss by 15 is a penalty which
punishes two, three or four star away prediction items.

We made changes to training hyper-parameters as well. We use trainValSplit ratio 0.99 instead of default
value 0.8 simply because we want more data to be trained. Moreover, we use Adam as optimizer other than
SGD since we found Adam is better at converging the loss with a smaller learning rate 0.005. We also
increase the batch size to 128 to prevent local minima. The epoch is left as default.
'''

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

# python version 3.7.6
# torch version 1.4.0
# torchtext version 0.6.0
# numpy version 1.18.1

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
import string
import re


###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################
def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # remove illegal characters in sample
    sample = [re.sub(r'[^\x00-\x7f]', r'', w) for w in sample]
    # high-freq but no emotional attitude words
    stopWords = {'this', 'the', 'are', 'item', 'for', 'a', 'i', 'xbox', 'who', 'was', 'game', 'it', 'and',
                 'his', 'to', 'her', 'him', 'his', 'on', 'is', 'son', 'thing', 'games', 'product', '3ds',
                 'mouse', 'pc', 'psn', 'rpg', 'ps3', 'controller', 'play', 'graphics', 'nintendo', 'wii',
                 'xbox360', "i'm", 'graphic', 'played', 'years', 'playing', 'sound', 'sounds', 'first',
                 'you', 'of', 'that', "it's", 'they', 'my', 'your'}
    s = []
    # dict to convert number to words
    # only care about 1-5 in case of reviews like 'I want to give x stars because...'
    NUMBER_CONSTANT = {'0': "zero ", '1': "one", '2': "two", '3': "three", '4': "four", '5': "five"}
    for i in sample:
        # filter some stopwords
        if i not in stopWords:
            # convert numbers to words
            if i in NUMBER_CONSTANT.keys():
                s.append(NUMBER_CONSTANT[i])
            else:
                # remove punctuations
                s.append(i.strip(string.punctuation))
    return s


def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    return batch


# more high-freq but no emotional attitude words
stopWords = {'this', 'item', 'for', 'a', 'i', 'xbox', 'who', 'was', 'game', 'it', 'and', 'to'}
wordVectorDimension = 300
wordVectors = GloVe(name='6B', dim=wordVectorDimension)

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
    return datasetLabel


def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    # convert regression output to float label 1.0, 2.0, ... 5.0
    netOutput = netOutput.round()
    netOutput[netOutput > 5] = 5.0
    netOutput[netOutput < 1] = 1.0
    return netOutput

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
    # Network structure: LSTM (bidirectional) -> Linear1 -> Relu -> Linear2 -> output
    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(wordVectorDimension, 50, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.Linear1 = tnn.Linear(200, 64)
        self.Relu = tnn.ReLU()
        self.Linear2 = tnn.Linear(64, 1)

    def forward(self, input, length):
        out, (hide, cell) = self.lstm(input)
        # concatenate the output of normal order and reversed order
        x = torch.cat((out[:, -1, :], out[:, 0, :]), dim=1)
        x = self.Linear1(x)
        x = self.Relu(x)
        x = self.Linear2(x)
        return x.squeeze()


class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """
    def __init__(self):
        super(loss, self).__init__()
        self.lossreg = tnn.L1Loss()

    # self-defined weighted MAE loss
    def forward(self, output, target):
        loss = torch.abs(output - target)
        resultloss = self.lossreg(output, target)
        for (i, j) in enumerate(loss):
            if j < 0.4:
                loss[i] *= 0.2
            elif j < 1:
                loss[i] *= 0.98
            else:
                loss[i] += 15
        loss = torch.mean(loss)
        return loss


net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
lossFunc = loss()

###########################################################################
################ The following determines training options ################
###########################################################################
# more training data than default ratio
trainValSplit = 0.99
# increase batch size to prevent local minima
batchSize = 128
epochs = 10
# faster converge using Adam than SGD
optimiser = toptim.Adam(net.parameters(), lr=0.005)
