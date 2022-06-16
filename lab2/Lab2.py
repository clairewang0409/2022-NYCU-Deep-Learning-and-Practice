#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(10)


# In[2]:


# Input


# In[3]:


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
            
    return np.array(inputs), np.array(labels).reshape(n, 1)


# In[4]:


def generate_XOR_easy():
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i ==0.5:
            continue
            
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(21, 1)


# In[5]:


# Show result


# In[6]:


def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground turth', fontsize=18)
    
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.show
            


# In[ ]:





# In[187]:


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)    
    return network
 
# Calculate neuron activation for an input
def weighted_sum(weights, inputs):
    z = weights[-1]
    for i in range(len(weights)-1):
        z += weights[i] * inputs[i]
    return z
 
# Transfer neuron activation
def activation(x, f='sigmoid'):
    if f == 'sigmoid':
        return 1.0/(1.0+np.exp(-x))
    if f == 'None':
        return 1.0 * x
    if f == 'tanh':
        return np.tanh(x)
    if f == "ReLU":
        return np.maximum(0, x)
    if f == "leaky_ReLU":
        if x>=0:
            return x
        else:
            return 0.001*x
        
# Calculate the derivative of an neuron output
def derivative_activation(x, f='sigmoid'):
    if f == 'sigmoid':
        return np.multiply(x,1.0-x)
    if f == 'None':
        return 1.0
    if f == 'tanh':
        return 1.0 - x ** 2
    if f == "ReLU":
        return 1.0 * (x > 0)
    if f == "leaky_ReLU":
        if x>=0:
            return 1
        else:
            return 0.001
        
# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            z = weighted_sum(neuron['weights'], inputs)
            neuron['output'] = activation(z, 'sigmoid')
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * derivative_activation(neuron['output'], 'sigmoid')
            
# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, dataset, l_rate, n_epoch, n_outputs):
    loss_list = []    
    for epoch in range(n_epoch):
        sum_error = 0      
        for row in dataset:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            sum_error = sum_error/len(dataset[0])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
            
        loss_list.append(sum_error)
        if epoch%500==0:
            print('epoch %4d    loss : %.5f' % (epoch, sum_error))
            
    plt.plot(loss_list, marker="s")
    plt.title('Learning curve')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    #print('%.8f' % outputs[1])
    return outputs.index(max(outputs)), outputs[1]


# In[188]:


x, y = generate_linear(n=100)
#x, y = generate_XOR_easy()


# In[189]:


dataset = np.hstack((x,y))

n_inputs = len(x[0])
n_outputs = len(set([i[0] for i in y]))
n_hidden = 2

network = initialize_network(n_inputs, n_hidden, n_outputs)

l_rate = 0.01
n_epoch = 10000

train_network(network, dataset, l_rate, n_epoch, n_outputs)


# In[190]:


network1 = list()
for layer in network:
    #print(layer[0]['weights'], layer[1]['weights'])
    hidden_layer = [{'weights':layer[i]['weights']} for i in range(len(layer))]
    network1.append(hidden_layer)
print(network1)


# In[191]:


count = 0
pre_y = []
for row in dataset:
    prediction, output = predict(network1, row)
    print('%.8f   label = %d  predict = %d' % (output, row[-1], prediction))
    #print(prediction)
    pre_y.append(prediction)
    if row[-1] == prediction:
        count += 1
print('\nAccuracy : ', count/len(dataset))
        


# In[192]:


show_result(x, y, pre_y)


# In[ ]:




