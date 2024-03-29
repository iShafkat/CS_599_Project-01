# This code creates five different neural networks and analyzes each of their performances in classifying hand written digits.

# The following codes import libraries
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras import losses
from keras.layers import LocallyConnected1D, LocallyConnected2D, Conv2D
import matplotlib.pyplot as plt
import numpy as np

# This is the main Function. It calls each of the neural networks and plots the output curve.It also processes the data set. 
def main():
    dataset = loadtxt('zip.train')         # Load the training Dataset
    X = dataset[:,1:257]                   # This commands stores the input features of each observation
    y = dataset[:,0:1]                     # This command stores the input labels of each observation
    y = to_categorical(y)                  # Categorizes the training labels     
    test_array = loadtxt("zip.test")       # load the Test Dataset
    test_label1 = test_array[:,0:1]        # This commands stores the input labels of each observation
    test_array = test_array[:,1:257]       # This commands stores the input features of each observation
    test_label = to_categorical(test_label1)                                  # Categorizes the training labels
    accuracy_one =network_one(X,y,test_label1,test_array)                     # Calls network one
    accuracy_two =network_two(X,y,test_label1,test_array)                     # Calls network two
    accuracy_three =network_three(X,y,test_label1,test_array)                 # Calls network three
    accuracy_four =network_four(X,y,test_label1,test_array)                   # Calls network four
    accuracy_five =network_five(X,y,test_label1,test_array)                   # Calls network five
# The following codes prodeuces the output curve
    t = np.arange(0., 30., 1)                                                 
    plt.xlabel("Training Epochs")
    plt.ylabel("%Correct On Test Data")
    plt.plot(t, accuracy_one, 'r', label ='Network One')
    plt.plot(t, accuracy_two, 'b', label ='Network Two')
    plt.plot(t, accuracy_three, 'g', label ='Network Three')
    plt.plot(t, accuracy_four, 'y', label ='Network Four')
    plt.plot(t, accuracy_five, 'black', label ='Network Five')
    plt.legend()
    plt.show()
    
# This function implements the network with No hidden layer
def network_one(X,y,test_label1,test_array):
    accuracy_one, epoch =[],1
    model = Sequential()
    model.add(Dense(10, input_dim=256, activation='tanh'))        # Defines a layer
    model.compile(loss=losses.mean_squared_error, optimizer='sgd', metrics=['accuracy'])   # Compiles the network
    while epoch<31:
        model.fit(X, y, epochs=epoch, batch_size=10)         # Trains the network
        _, accuracy = model.evaluate(X, y)                   # Training Accuracy is measured
        predictions = model.predict_classes(test_array)      # The network Predicts the output of the test data set
        correct =0
# Finds the accuracy of the network for test data
        for i in range(len(predictions)):
            if predictions[i]==test_label1[i]:
                correct +=1
            else:
                pass
        accuracy =100 * correct/len(predictions)
        accuracy_one.append(accuracy)
        epoch +=1
    return accuracy_one

# This function implements the network with One hidden layer
def network_two(X,y,test_label1,test_array):
    accuracy_two, epoch =[],1
    model = Sequential()
    model.add(Dense(12, input_dim=256, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['accuracy'])
    while epoch<31:
        model.fit(X, y, epochs=epoch, batch_size=10)
        _, accuracy = model.evaluate(X, y)
        predictions = model.predict_classes(test_array)
        correct =0
        for i in range(len(predictions)):
            if predictions[i]==test_label1[i]:
                correct +=1
            else:
                pass
        accuracy =100 * correct/len(predictions)
        accuracy_two.append(accuracy)
        epoch +=1
    return accuracy_two

# This function implements the network with Two hidden layers locally connected
def network_three(X,y,test_label1,test_array):
    accuracy_three, epoch =[],1
    model = Sequential()
    X= X.reshape(X.shape[0], 16, 16,1).astype('float32')
    test_array= test_array.reshape(test_array.shape[0], 16, 16,1).astype('float32')
    model.add(LocallyConnected2D(64, (3,3), input_shape=(16, 16,1), activation='sigmoid'))
    model.add(LocallyConnected2D(16, (5,5), activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['accuracy'])
    while epoch<31:
        model.fit(X, y, epochs=epoch, batch_size=10)
        _, accuracy = model.evaluate(X, y)
        predictions = model.predict_classes(test_array)
        correct =0
        for i in range(len(predictions)):
            if predictions[i]==test_label1[i]:
                correct +=1
            else:
                pass
        accuracy =100 * correct/len(predictions)
        accuracy_three.append(accuracy)
        epoch +=1
    return accuracy_three

# This function implements the network with Two hidden layers, locally connected with weight sharing
def network_four(X,y,test_label1,test_array):
    accuracy_four, epoch =[],1
    model = Sequential()
    X= X.reshape(X.shape[0], 16, 16,1).astype('float32')
    test_array= test_array.reshape(test_array.shape[0], 16, 16,1).astype('float32')
    model.add(Conv2D(128,(3,3), input_shape=(16,16,1), activation='sigmoid'))
    model.add(LocallyConnected2D(16, (5,5), activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['accuracy'])
    while epoch<31:
        model.fit(X, y, epochs=epoch, batch_size=10)
        _, accuracy = model.evaluate(X, y)
        predictions = model.predict_classes(test_array)
        correct =0
        for i in range(len(predictions)):
            if predictions[i]==test_label1[i]:
                correct +=1
            else:
                pass
        accuracy =100 * correct/len(predictions)
        accuracy_four.append(accuracy)
        epoch +=1
    return accuracy_four

# This function implements the network with Two hidden layers, locally connected, two levels of weight sharing
def network_five(X,y,test_label1,test_array):
    accuracy_five, epoch =[],1
    model = Sequential()
    X= X.reshape(X.shape[0], 16, 16,1).astype('float32')
    test_array= test_array.reshape(test_array.shape[0], 16, 16,1).astype('float32')
    model.add(Conv2D(128,(5,5), input_shape=(16,16,1), activation='relu'))
    model.add(Conv2D(64, (4,4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['accuracy'])
    while epoch<31:
        model.fit(X, y, epochs=epoch, batch_size=10)
        _, accuracy = model.evaluate(X, y)
        predictions = model.predict_classes(test_array)
        correct =0
        for i in range(len(predictions)):
            if predictions[i]==test_label1[i]:
                correct +=1
            else:
                pass
        accuracy =100 * correct/len(predictions)
        accuracy_five.append(accuracy)
        epoch +=1
    return accuracy_five

# This calls the main function
if __name__ == '__main__':
    main()
