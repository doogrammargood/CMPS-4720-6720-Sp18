import csv
import math
import numpy as np
with open('SPECT.train.csv', 'r') as f:
    train_data = list(csv.reader(f))
train_data = map(lambda x: map(lambda y: float(y), x), train_data) #the data comes in as strings.
with open('SPECT.test.csv', 'r') as f:
    test_data = list(csv.reader(f))
test_data = map(lambda x: map(lambda y: float(y), x), test_data)

def sigmoid(x):
    return 1/(1 + math.exp(-x))

#Using a 4-layer archinecture, initialize all matricies to 0.
#The dimensions of the matricies will need to match, with an extra 1
#for the bias.

#Convention: we will be using row vectors to represent the neurons,
#which means we will be multiplying these matricies on the right.

#The model consists of a list of matricies of weights / biases. The biases are the last row of the matrix.
#The address of a neuron will be given by [layer_number, neuron_number].
#These aren't part of the model- the model is just the list of matricies of transitions between them.
#The kth element of the model connects the k-1th and kth levels, the -1th level being the input

#the more complicated architecture below gives similar results to the simpler, two layer architecture.

#layer1 = np.random.rand(23,50)
#layer2 = np.random.rand(51,40)
#layer3 = np.random.rand(41,10)
#layer4 = np.random.rand(11,1)

layer1 = np.random.rand(23,10)
layer4 = np.random.rand(11,1)

model = [layer1, layer4]
#So that what is actually stored in the model consists of the transition rules between the matricies (minus the sigmoid)
def evaluateModel(model, input):
    activation_data = [] #This will record the activation data for entire model, needed for backpropagation
    partial_answer = input[:] #this is for the biases
    partial_answer.append(1)
    for layer in model:
        partial_answer = np.dot(partial_answer, layer).tolist() #apply the next layer, multiply on the right.
        if type(partial_answer[0]) == type([]):
            partial_answer = partial_answer[0] #idk why I have to do this.
        partial_answer = map(lambda x: sigmoid(x), partial_answer) #Apply the sigmoid to each neuron.
        partial_answer.append(1) #For the bias of the next layer.
        activation_data.append(partial_answer[:]) #store the neuron activity in this layer for backpropagation.
    return partial_answer[0], activation_data

    #   d err  = d err * d sig * d net     The calculus chain rule
    #   d wi     d sig   d net   d wi

    #note d net = activations[layer][i]
    #     d wi
    #           d sig = sig(net) * (1 - sig(net))
    #           d net
    #     d err = 2(val - sig)
    #     d sig

def getUpdateVector(address, model, activation_data, input): #The address is which neuron we want to update
    #address comes in the form [layer, node_number]
    #activiation_data records the activation level of each neuron in the model, computed during the forward step.
    #We return d activation
    #          d weights      A vector which has as its components the derivative of the neuron given by address wrt the weights feeding into that neuron.
    #According to the chain rule, this is sig * (1 - sig) * activation
    sig = activation_data[address[0]][address[1]]
    err = (sig - activation_data[len(model)-1][0])**2
    previous_layer_number = address[0] - 1
    number_of_nodes_in_previous_layer = len(model[address[0]])
    if address[0] > 0:
        return sig * (1 - sig) * np.array(map(lambda k: activation_data[previous_layer_number][k], np.arange(number_of_nodes_in_previous_layer))) #+1 for the bias
    else:
        return sig * (1 - sig) * np.array(input + [1]) #+1 for the bias
    #We are returning an array of values to update the weights which come from our target node.
    #These correspond to a column in the weight matrix

def getBlameVector(address, model, activation_data):
    #Returns a vector d node_activation
    #                 d node_activation_of_previous_layer,    A vector containing the derivative of the neuron given by address wrt the neurons feeding into it.
    #Using the chain rule, this is sig * (1 - sig) * weight
    sig = activation_data[address[0]][address[1]]
    #previous_layer_number = address[0] - 1
    blameVector = np.arange(len(model[address[0]])-1) #The number of neurons in the previous layer (-1 for the bias)
    return sig * (1 - sig) * np.array(map(lambda k: model[address[0]].tolist()[k][address[1]], blameVector))

def backPropagate(input, model):
    R = 1.0 #learning rate
    result, activation_data = evaluateModel(model, input[0:-1])
    blameVector = [ 2 * (result - input[-1]) ]

    totalUpdateData = []
    #print result
    #print input[-1]
    for layer in range(len(model)-1, -1, -1): #loops backwards through the layers
        neurons = np.arange(model[layer].shape[1]) # an array of numbers 0 .. #neurons in this layer
        updates_for_this_layer = []
        update_this_layer = map(lambda x: (x[0] * np.array(getUpdateVector([layer, x[1]], model, activation_data, input[0:-1]))).tolist(), zip(blameVector, neurons))
        totalUpdateData.append(np.matrix(update_this_layer).transpose())
        #print totalUpdateData
        blameVector = reduce(lambda x,y: np.array(x) + np.array(y), map(lambda n: n[0]*getBlameVector([layer, n[1]], model, activation_data), zip(blameVector,neurons))) #Gets the blame vector for the next level.
    #--- Now we have to use totalUpdateData:
    totalUpdateData.reverse()
    for layer_number in range(len(model)-1, -1, -1):
        #print totalUpdateData[layer_number]
        model[layer_number] = -R * totalUpdateData[layer_number] + np.matrix(model[layer_number])
    return model

def train_network(training_data, model):
    #model = backPropagate(training_data[0], model)
    np.random.shuffle(training_data)
    for example in training_data:
        model = backPropagate(example, model)
    return model

def test_network(test_data, model):
    totalError = 0
    errorCount = 0
    for example in test_data:
        computerValue, _ = evaluateModel(model, example[0:-1])
        totalError += (computerValue - example[-1])**2
        if(abs(computerValue - example[-1]) > 0.5):
            errorCount +=1
    return totalError, errorCount

model = train_network(train_data, model)
print model
print len(test_data)
print 'total error is'
total_error, error_count = test_network(test_data, model)
print total_error
print 'incorrect answer count'
print error_count
print 'out of'
print len(test_data)

#discussion:
# There is no guarantee that this code works as intended, and there are some reasons to believe it does not.
# The typical error rate is about 84/187 when trained and 104/187 when untrained, so its probably doing something.
# It was a very bad idea to try to use map-filter-reduce with lists that are sometimes numpy matricies.
# It would probably be good to make these matricies sparse, but doing so would be a real hassle.
