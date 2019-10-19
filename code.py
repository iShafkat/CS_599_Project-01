# This code creates five different neural networks and analyzes each of their performances in classifying hand written digits.

#import libraries
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras import losses
from keras.layers import LocallyConnected1D, LocallyConnected2D, Conv2D
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the Training Dataset
    dataset = loadtxt('zip.train')  
    X = dataset[:,1:257]
    y = dataset[:,0:1]
    y = to_categorical(y)
    #load the Test Dataset
    test_array = loadtxt("zip.test")
    test_label1 = test_array[:,0:1]
    test_array = test_array[:,1:257]
    test_label = to_categorical(test_label1)
    accuracy_one =network_one(X,y,test_label1,test_array)
    accuracy_two =network_two(X,y,test_label1,test_array)
    accuracy_three =network_three(X,y,test_label1,test_array)
    accuracy_four =network_four(X,y,test_label1,test_array)
    accuracy_five =network_five(X,y,test_label1,test_array)
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
def network_one(X,y,test_label1,test_array):
    accuracy_one, epoch =[],1
    model = Sequential()
    model.add(Dense(10, input_dim=256, activation='tanh'))
    model.compile(loss=losses.mean_squared_error, optimizer='sgd', metrics=['accuracy'])
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
        accuracy_one.append(accuracy)
        epoch +=1
    return accuracy_one
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
if __name__ == '__main__':
    main()
