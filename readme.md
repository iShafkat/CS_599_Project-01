# CS-599 Project-01
Neural Network Based Digit Classifier
# How to Run the Code
Download/clone the repository to the local machine and run the Makefile. The local machine should contain the following software:
python version 3.6 or above.
# Description
This project is to reproduce the following figure(in the below section named Figure).
In this classification project, the neural network will classify the ZIP code Dataset into 10 different digit classes. There would be two phase, one is training phase and the other one would be testing phase. We will consider 5 different neural network architectures and compare their performances. For each of the neural network architecture, the size of epoch in training would be varied and the corresponding testing error would be calculated. Based on this data, the figure would be plotted for 5 different architecture.
Function: Each of the neural networks will learn a function that predicts a value to represent the estimated probability that an image x has digit class k, for k=1, 2, 3, …., 9.
# Dataset
The problem is described as the “classification of handwritten digits”. More specifically, classifying a digit image based on the features of the image pixels.
Data Source URL: https://web.stanford.edu/~hastie/ElemStatLearn/data.html
Training Observation: 7291, Test Observation: 2007
Each observation contains a 16x16 (256 pixel) input feature.
Each observation contains an output id (0-9) of a digit.

# Project Proposal Link:
https://github.com/iShafkat/CS_599_Project-01/blob/master/Project_Week01.pdf


# Figure:
![image](https://github.com/iShafkat/CS_599_Project-01/blob/master/figure1.JPG)
Figure Reference: Hastie, et al. Elements of Statistical Learning, Figure 11.11.
