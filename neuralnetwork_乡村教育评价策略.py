#!/usr/bin/env python
# coding: utf-8

# In[267]:


import numpy as np
import scipy.special

class neuralnetwork :
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        self.lr = learningrate
        
        #activation function is the sigmoid function
        self.activation_function = lambda x : scipy.special.expit(x)
        pass
    
    def train(self, inputs_list, targets_list) :
        
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T
        
        #计算输出
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        #计算权重更新
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        #更新hidden-output的权重)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # 更新input-hidden的权重
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass
                                     
    def query(self, inputs_list) :
        
        #把输入的inputs_list弄成一个2纬数组
        inputs = np.array(inputs_list, ndmin = 2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    


# In[268]:


# number of input, hidden and output nodes
input_nodes = 10
hidden_nodes = 20
output_nodes = 5

# learning rate
learning_rate = 0.4

# create instance of neural network
n = neuralnetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

data = []
cycle_produce = 100

for c in range(cycle_produce):
    data_produce = np.random.randint(1,6,size = 10)
    data.append((data_produce))
    #print(data[0])
    pass


# In[269]:


epochs = 1000

for e in range(epochs):
    i = 0
    for d in data:      
        #print("data_train is",data_train)
        inputs = (np.asfarray(data[i])/50.0 * 0.99) + 0.01
        #print(data[i])
        #print("inputs is",inputs)
        targets = np.zeros(output_nodes) + 0.01       
        targets[(data[i][0]-1)] = 0.99
        #print("targets is",targets)
        n.train(inputs, targets)
        #outputs = n.query(inputs)
        #print("outputs_train is",outputs)
        i = i + 1
        pass
    pass

#print(n.wih)
#print(n.who)


# In[270]:


scorecard = []
cycle_test = 100
for c in range(cycle_test):
    data_test = np.random.randint(1,6,size = 10)
    #print("data_test is",data_test)
    inputs = ( data_test/50.0 * 0.99) + 0.01
    #print("inputs is",inputs)
    outputs = n.query(inputs)
    #print("outputs_test is",outputs)
    
    correct_label = int(data_test[0])
    label = np.argmax(outputs)+1
    #print(label)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    
    pass

scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




