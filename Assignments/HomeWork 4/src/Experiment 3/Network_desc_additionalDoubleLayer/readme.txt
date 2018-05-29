In this trial, the number of features in the following architecture are doubled as compared to normal network description file and one more convolutional neural network layer is added to the network,
the network architechture is the text file is shown like below in the Network_desc_additionalDoubleLayer.txt file:

Network Desc:
5 8
6 16
6 32
8 64
128


HyperParameters:

EPSILON = 0.1,
MAX_UPDATES = 50, (i.e. max number of Epoch's)
batch_size = 128
regularization_alpha = 1
Cost Function = cross-l2


More Detailed description on the training and testing file:

Training done for class: Circle
Number of Training Data Set: 5000
Number of Testing Data Set: 1000
Number of Circle Images in Training Data Set: 3020
Number of Circle Images in Testing Data Set: 606


Results:

Total number of Epoch used to reach minimum validation cost: 29
Testing Accuracy Achieved: 98.100001%
Further the results of every epoch and testing accuracy achieved in every epoch can be checked in cross-l2.txt