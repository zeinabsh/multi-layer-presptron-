import math
from preproccessing1 import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


####################################################################################################

# _____model______
class NeuralNetwork(object):
    def __init__(self, layers, lRate,bias=False):
        self.lRate = lRate
        self.inputsize = 5  # input layer count
        self.outputsize = 3  # output layer count
        self.hiddenlayers = layers
        self.activation_list = []
        self.deltas = []
        self.W = []
        if(bias==False):
            self.w1 = np.random.randn(self.inputsize, self.hiddenlayers[0])  # weights from input to hidden layer
            self.W.append(self.w1)
            for i in np.arange(0, len(self.hiddenlayers) - 1):  # [5][2,5,7][3]
                # randomly initialize a weight matrix connecting the
                # number of nodes in each respective layer together,
                # adding an extra node for the bias
                w = np.random.randn(self.hiddenlayers[i], self.hiddenlayers[i + 1])
                self.W.append(w)
            self.w2 = np.random.randn(self.hiddenlayers[-1], self.outputsize)  # weights from hidden to output
            self.W.append(self.w2)
        else:
            self.w1 = np.random.randn(self.inputsize+1, self.hiddenlayers[0]+1)  # weights from input to hidden layer
            self.W.append(self.w1)
            for i in np.arange(0, len(self.hiddenlayers) - 1):  # [5][2,5,7][3]
                # randomly initialize a weight matrix connecting the
                # number of nodes in each respective layer together,
                # adding an extra node for the bias
                w = np.random.randn(self.hiddenlayers[i]+1, self.hiddenlayers[i + 1]+1)
                self.W.append(w)
            self.w2 = np.random.randn(self.hiddenlayers[-1]+1, self.outputsize)  # weights from hidden to output
            self.W.append(self.w2)

    def FeedForward(self, X, activation_func=False):

        if (activation_func == False):
            self.net_list = []
            self.net = np.dot(X, self.w1)  # net of hidden layer        [5][2,5,7][3]
            self.o = self.sigmoid(self.net)  # Activation Function
            self.activation_list.append(self.o)
            for i in np.arange(0, len(self.hiddenlayers) - 1):
                self.net = np.dot(self.activation_list[i], self.W[i + 1])  # net of hidden layer        [5][2,5,7][3]
                self.o = self.sigmoid(self.net)  # Activation Function
                self.activation_list.append(self.o)
            self.net_output = np.dot(self.activation_list[-1], self.W[-1])  # net of output layer
            self.y = self.sigmoid(self.net_output)  # y output value
            return self.y
        else:
            self.net_list = []
            self.net = np.dot(X, self.w1)  # net of hidden layer        [5][2,5,7][3]
            self.o = self.tanh(self.net)  # Activation Function
            self.activation_list.append(self.o)
            for i in np.arange(0, len(self.hiddenlayers) - 1):
                self.net = np.dot(self.activation_list[i], self.W[i + 1])  # net of hidden layer        [5][2,5,7][3]
                self.o = self.tanh(self.net)  # Activation Function
                self.activation_list.append(self.o)
            self.net_output = np.dot(self.activation_list[-1], self.W[-1])  # net of output layer
            self.y = self.tanh(self.net_output)  # y output value
            return self.y

    def sigmoid(self, n, deriv=False):
        if deriv == True:
            return n * (1 - n)
        return 1 / (1 + np.exp(-n))

    def tanh(self, x, deriv=False):
        ''' It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1.'''
        if (deriv == True):
            k=np.tanh(x)
            return 1 - k*k
        return np.tanh(x)

    def backPropagate(self, X, y, output,act=False):
        if(act==False):
            #backpropagate output
            self.error = y - output  # y is desired output and output is the actual output we got
            self.delta_output = self.error * self.sigmoid(output, deriv=True)
            self.deltas.append(self.delta_output)
            # backpropagate layer
            for i in np.arange(len(self.hiddenlayers) - 1, -1, -1):  # [5][2,5,7][3]
                self.error_Hidden = self.deltas[-1].dot(self.W[i + 1].T)
                self.delta = self.error_Hidden * self.sigmoid(self.activation_list[i], deriv=True)
                self.deltas.append(self.delta)

            # update   wight 1
            self.w1 += X.T.dot(self.delta) * self.lRate  # [5][2,5,7][3]
            self.W[0] = self.w1
            # update wight  of layer
            for i in np.arange(0, len(self.hiddenlayers) - 1):
                self.W[i + 1] += self.activation_list[i].T.dot(self.deltas[len(self.hiddenlayers) - 1 - i]) * self.lRate
           # update wight  of  output layer
            self.w2 += self.activation_list[-1].T.dot(self.delta_output) * self.lRate
            self.W[-1] = self.w2
        else:
            self.error = y - output  # y is desired output and output is the actual output we got
            self.delta_output = self.error * self.tanh(output, deriv=True)
            self.deltas.append(self.delta_output)
            for i in np.arange(len(self.hiddenlayers) - 1, -1, -1):  # [5][2,5,7][3]
                self.error_Hidden = self.deltas[-1].dot(self.W[i + 1].T)
                self.delta = self.error_Hidden * self.tanh(self.activation_list[i], deriv=True)
                self.deltas.append(self.delta)
             #update  wight
            self.w1 += X.T.dot(self.delta) * self.lRate  # [5][2,5,7][3]
            self.W[0] = self.w1

            for i in np.arange(0, len(self.hiddenlayers) - 1):
                self.W[i + 1] += self.activation_list[i].T.dot(self.deltas[len(self.hiddenlayers) - 1 - i]) * self.lRate

            self.w2 += self.activation_list[-1].T.dot(self.delta_output) * self.lRate
            self.W[-1] = self.w2

    def train(self, X, y,act=False,b=False):
        if(b==True):
            X = np.c_[X, np.ones((X.shape[0]))]
        output = self.FeedForward(X, act)
        self.backPropagate(X, y, output,act)
        self.tt=[]
        self.tt.append(self.y)
        for j in range(90):
            self.maxi = self.maximum(self.tt[0][j][0], self.tt[0][j][1], self.tt[0][j][2])      #1,0,0
                                                                                                #0,1,0
            self.tt[0][j][0] = self.tt[0][j][0] - self.maxi
            self.tt[0][j][1] = self.tt[0][j][1] - self.maxi
            self.tt[0][j][2] = self.tt[0][j][2] - self.maxi
            for i in range(3):
                if self.tt[0][j][i] < 0:
                    self.tt[0][j][i] = 0
                else:
                    self.tt[0][j][i] = 1
            self.maxi = 0

    def maximum(self, a, b, c):

        if (a >= b) and (a >= c):
            largest = a

        elif (b >= a) and (b >= c):
            largest = b
        else:
            largest = c

        return largest



    def test(self, Xx, activation_func=False,b=False):
        if (b == True):
            Xx =  np.c_[Xx, np.ones((Xx.shape[0]))]
        self.ll = []
        self.activation_list = []
        self.net_list = []
        if (activation_func == False):
            for hh in range(60):
                self.activation_list = []
                self.net = np.dot(Xx.values[hh], self.w1)  # net of hidden layer        [5][2,5,7][3]
                self.o = self.sigmoid(self.net)  # Activation Function
                self.activation_list.append(self.o)
                for i in np.arange(0, len(self.hiddenlayers) - 1):
                    self.net = np.dot(self.activation_list[i],
                                      self.W[i + 1])  # net of hidden layer        [5][2,5,7][3]
                    self.o = self.sigmoid(self.net)  # Activation Function
                    self.activation_list.append(self.o)
                self.net_output = np.dot(self.activation_list[-1], self.W[-1])  # net of output layer
                self.y = self.sigmoid(self.net_output)  # y output value
                self.maxi = self.maximum(self.y[0], self.y[1], self.y[2])
                self.y[0] = self.y[0] - self.maxi
                self.y[1] = self.y[1] - self.maxi
                self.y[2] = self.y[2] - self.maxi
                for i in range(3):
                    if self.y[i] < 0:
                        self.y[i] = 0
                    else:
                        self.y[i] = 1
                self.maxi = 0
                self.ll.append(self.y)
            return self.ll
        else:
            for hh in range(60):
                self.activation_list = []
                if (b == True):
                    z = Xx[hh]
                else:
                    z=Xx.values[hh]
                self.net = np.dot(z, self.W[0])  # net of hidden layer        [5][2,5,7][3]
                self.o = self.tanh(self.net)  # Activation Function
                self.activation_list.append(self.o)
                self.net_list.append(self.net)
                for i in np.arange(0, len(self.hiddenlayers) - 1):
                    self.net = np.dot(self.activation_list[i],
                                      self.W[i + 1])  # net of hidden layer        [5][2,5,7][3]
                    self.o = self.tanh(self.net)  # Activation Function
                    self.activation_list.append(self.o)

                self.net_output = np.dot(self.activation_list[-1], self.W[-1])  # net of output layer
                self.y = self.tanh(self.net_output)  # y output value__len__ = {int} 1
                self.maxi = self.maximum(self.y[0], self.y[1], self.y[2])
                self.y[0] = self.y[0] - self.maxi
                self.y[1] = self.y[1] - self.maxi
                self.y[2] = self.y[2] - self.maxi
                for i in range(3):
                    if self.y[i] < 0:
                        self.y[i] = 0
                    else:
                        self.y[i] = 1
                self.maxi = 0
                self.ll.append(self.y)
            return self.ll


######################################################################################################

def input_user(nu_hidden, layers, learning_rate, epochs, bias, selected_fun):
    lis = layers
    nn = NeuralNetwork(lis, learning_rate,bias)
    for epochs in range(epochs):
        nn.train(X_train, y_train, selected_fun,b=bias)

    count1 = 0
    count2 = 0
    class1_true = 0
    class2_true = 0
    class3_true = 0
    c1_c2 = 0
    c1_c3 = 0
    c2_c1 = 0
    c2_c3 = 0
    c3_c1 = 0
    c3_c2 = 0

    actual_out = (nn.test(X_test, activation_func=True,b=bias))
    for i in range(60):

        if (actual_out[i][0] == y_test.values[i][0] and actual_out[i][1] == y_test.values[i][1] and actual_out[i][2] ==y_test.values[i][2]):
            count2 += 1
            for j in range(3):
                if (actual_out[i][j] == 1):
                    if (j == 0):
                        class1_true += 1
                    elif (j == 1):
                        class2_true += 1
                    else:
                        class3_true += 1

        else:
            for j in range(3):
                if (actual_out[i][j] == 1):
                    index_act = j
                if (y_test.values[i][j] == 1):
                    index_y = j

            if (index_y == 0):
                if (index_act == 1):
                    c1_c2 += 1
                else:
                    c1_c3 += 1

            elif (index_y == 1):
                if (index_act == 0):
                    c2_c1 += 1
                else:
                    c2_c3 += 1
            else:
                if (index_act == 0):
                    c3_c1 += 1
                else:
                    c3_c2 += 1

    count3 = 0
    count4 = 0
    for i in range(90):
        for j in range(3):
            if nn.tt[0][i][j] - y_train.values[i][j] == 0:
                count3 += 1
        if count3 == 3:
            count4 += 1
        count3 = 0

    print("accuracy training= ", (count4 / 90) * 100)
    print("accuracy = ", (count2 / 60) * 100)
    ###___matrix_____
    print("           ", "c1", "---", "c2", "---", "c3")
    print("class1_test", class1_true, " ---", c2_c1, " ---", c3_c1)
    print("class2_test", c1_c2, "---", class2_true, "---", c3_c2)
    print("class3_test", c1_c3, " ---", c2_c3, " ---", class3_true)
    print(len(X_test.index))

