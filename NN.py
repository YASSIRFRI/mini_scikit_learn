#!/usr/bin/env python
# coding: utf-8

# ## Libraries 

# In[1]:


import math
import time 
import random 


# ## Creating Layer Class
# 

# In[2]:


class Layer:
    def __init__(self, input_size, output_size , activation_function):
        self.input = None
        self.output = []
        self.weights = []
        self.activation_function = activation_function
        ##nested loop to initailze the weigths with random values
        for i in range(input_size):
          col = []
          for j in range(output_size):
              col.append(random.random())
          self.weights.append(col)
        

    def forward_propagation(self, given_input  ):
        self.input = given_input
        self.output = []
        for col in range(len(self.weights[0])):
            sum = 0 
            for i in range(len(self.input) ):
                sum += self.input[i] * self.weights[i][col]
                
            if self.activation_function == "sigmoid" :
                self.output.append( 1/ (  1+(math.exp(-sum)) ))
            
            elif self.activation_function == "tanh" :
                self.output.append( math.tanh(sum))

        
        return self.output
        
        
    def back_propagation(self, previous_errors, learning_rate , targets):
        
        upcoming_error = []
        for i in range( len(self.input)):
            upcoming_error.append(0)

        
       #change the weights at the output layer
        if len(previous_errors) == 0 :
            for cur_neuron in range ( len(self.output) ) :
                delta_j = ( targets[cur_neuron] - self.output[cur_neuron] ) * self.output[cur_neuron]  * (1-self.output[cur_neuron])

                for upcoming_nueron in range( len( self.input) ) :
                    upcoming_error[upcoming_nueron] += delta_j * self.weights[upcoming_nueron][cur_neuron] 
                    self.weights[upcoming_nueron][cur_neuron] += ( delta_j * self.input[upcoming_nueron] * learning_rate)

      
        #change the weights at the hidden layer
        else :
                
            for cur_neuron in range ( len(self.output) ) :
                delta_j = previous_errors[cur_neuron] * self.output[cur_neuron]  * (1-self.output[cur_neuron]) 

                for upcoming_nueron in range( len ( self.input) ) :
                    
                    upcoming_error[upcoming_nueron] += delta_j * self.weights[upcoming_nueron][cur_neuron] 
                    self.weights[upcoming_nueron][cur_neuron] += delta_j * learning_rate * self.input[upcoming_nueron]
                    
                    
        return upcoming_error
            


# ## Creating Network Class

# In[3]:


class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer ):
        self.layers.append(layer )


    def predict(self, input_data):
        
        predicted = []
        for i in range( len(input_data) ):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output )
            predicted.append(output)

        return predicted


    def fit(self, train_data, train_labels, epochs, learning_rate):
        
        samples = len(train_data)

        # training loop
        for epoch in range(epochs):
            
            for sample in range(samples):
        
                cur_deltas = []
                
                output = train_data[sample]
                
                for layer in self.layers:
                    output = layer.forward_propagation(output )


                for layer in reversed(self.layers):
                    cur_deltas = layer.back_propagation(cur_deltas, learning_rate , train_labels[sample])
                
                
class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        predictions = []
        for input_sample in input_data:
            output = input_sample
            for layer in self.layers:
                output = layer.forward_propagation(output)
            predictions.append(output)
        return predictions

    def fit(self, train_data, train_labels, epochs, learning_rate):
        for epoch in range(epochs):
            for sample_index, input_sample in enumerate(train_data):
                output = input_sample
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                cur_deltas = []
                for layer in reversed(self.layers):
                    cur_deltas = layer.back_propagation(cur_deltas, learning_rate, train_labels[sample_index])
"""
    def score(self, test_data, test_labels):
        predictions = self.predict(test_data)
        return calculate_accuracy(predictions, test_labels)

    @staticmethod
    def calculate_accuracy(predictions, test_labels):
        predicted_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(test_labels, axis=1)
        return np.mean(predicted_classes == actual_classes)"""