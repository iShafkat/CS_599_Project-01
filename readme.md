# CS-599 Project-01
Neural Network Based Digit Classifier
# How to Run the Code
Download/clone the repository to the local machine and run the Makefile. The local machine should contain python version 3.6 or above.
# Description
In this classification project, the neural network will classify the ZIP code Dataset into 10 different classes. There would be two phase, one is training phase and the other one would be testing phase. We will consider 5 different neural network architectures and compare their performances. For each of the neural network architecture, the size of epoch in training would be varied and the corresponding testing error would be calculated. Based on this data, the figure would be plotted for 5 different architecture.
Function: The network will learn a function that predicts a value to represent the estimated probability that an image x has digit class k, for k=1, 2, 3, …., 9
# Dataset
The problem is described as the “classification of handwritten digits”. More specifically, classifying a digit image based on the features of the image pixels.
Data Source URL: https://web.stanford.edu/~hastie/ElemStatLearn/data.html
Training Observation: 7291, Test Observation: 2007
Each observation contains a 16x16 (256 pixel) input feature.
Each observation contains an output id (0-9) of a digit.