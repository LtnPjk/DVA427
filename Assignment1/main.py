import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
from sklearn import datasets
from sklearn import preprocessing

# Global vars
train_features = []
train_labels = []
validation_features = []
validation_labels = []
test_features = []
test_labels = []
feature_set = []

min_features = np.full((19,1), np.inf)
max_features = np.zeros((19,1))

def find_min_max(features):
    global min_features, max_features
    for x in features:
        for i,y in enumerate(x):
            if y > max_features[i]:
                max_features[i] = y
            if y < min_features[i]:
                min_features[i] = y

def normalize():
    global min_features, max_features, feature_set
    for x in feature_set:
        for i,y in enumerate(x):
            y = (y-min_features[i])/(max_features[i]-min_features[i])

def parse_file():
    feature_set, labels = datasets.make_moons(100, noise=0.10)
    '''
    plt.figure(figsize=(10,7))
    plt.scatter(feature_set[:,0], feature_set[:,1], c=labels)
    plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)
    plt.show()
    '''

    labels = labels.reshape(100, 1)

    # Parse training set
    # Split lines into patients
    with open(os.path.join(sys.path[0], "assignment1.txt"), "r") as f:
        patients = f.read().splitlines()

    #remove the first unusable lines
    for i in range(0, 24):
        patients.pop(0)

    patient_att = []

    # Every patient has 19 attributes, split them by ","
    for i, patient in enumerate(patients):
        patient_att.append(patient.split(','))
        #print(patient)
        #if i == 5:
            #print(patient_att)
            #break

    # Take the first 18 attributes as training input
    training_inp = [attribute[0:19] for attribute in patient_att]
    # Take the last attribute as training output (target)
    training_oup = [[attribute[-1] for attribute in patient_att]]


    feature_set = np.array([[float(j) for j in i] for i in training_inp])
    labels = np.array([[float(j) for j in i] for i in training_oup]).T

    find_min_max(feature_set)
    print(min_features)
    print(max_features)
    normalize()

    validation_inp = feature_set[int(len(feature_set)*0.75):int(len(feature_set)*0.85)]
    test_inp = feature_set[int(len(feature_set)*0.85):int(len(feature_set))]
    feature_set = feature_set[0:int(len(feature_set)*0.75)]

    validation_oup = labels[int(len(labels)*0.75):int(len(labels)*0.85)]
    test_oup = labels[int(len(labels)*0.85):int(len(labels))]
    labels = labels[0:int(len(labels)*0.75)]


    print(validation_inp.shape)
    print(test_inp.shape)
    print(feature_set.shape)

    return feature_set, labels, validation_inp, validation_oup, test_inp, test_oup

# Activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of activation function
def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def round_half_up(n, decimals = 0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier

def feed_forward(features, labels):
    
    # "Un-squashed" values on hidden layer
    zh = np.dot(features, wh)
    # Squashed values on hidden layer
    ah = sigmoid(zh)

    # "Un-squashed" values on output layer
    zo = np.dot(ah, wo)
    # Squashed values on output layer
    ao = sigmoid(zo)

    # Calculate error
    error_out = ((1 / 2) * (np.power((ao - labels), 2)))

    return zh, ah, zo, ao, error_out.sum()

def validation():
    global last_error, smallest_error, best_wh, best_wo, wh, wo
    a, b, c, d, validation_error = feed_forward(validation_features, validation_labels)
    #print("Validation Error: ", validation_error, "error diff: ", abs(validation_error-last_error))
    
    last_error = validation_error

    if smallest_error == 1000 or smallest_error > validation_error:
        smallest_error = validation_error
        best_wh = wh
        best_wo = wo
        
    elif validation_error > smallest_error * 1.50:
        print("Validation Interrupt")
        wh = best_wh
        wo = best_wo
        return -1
    return validation_error

def test():
    zh, ah, zo, ao, error_out = feed_forward(test_features, test_labels)

    ao = [int(round_half_up(i, 0)) for i in ao]

    correct_guesses = 0

    for i, j in zip(ao, test_labels):
        if i == int(j):
            correct_guesses = correct_guesses + 1
    
    return (correct_guesses/len(test_labels))

def train(epochs, n):
    global wo, wh, res, lr
    plt.ion()
    epoch_vali = []

    for epoch in range(epochs):
        zh, ah, zo, ao, error = feed_forward(train_features, train_labels)
        '''
        if epoch > 10000:
            lr = 0.0006
        '''
        #lr = np.power(0.0000012 * epoch - 0.07, 2) + 0.0003

        # Validation
        if epoch % n == 0:
            print('Epoch: ', epoch)
            res = validation()
            epoch_vali.append(res)
            if len(epoch_vali) > 100:
                epoch_vali.pop(0)
            if res == -1:
                print("Predictability: ", test())
                exit()

            plt.clf()
            x = np.linspace(0, len(epoch_vali), len(epoch_vali))
            plt.plot(x, epoch_vali, color = 'red')
            plt.title(str(epoch))
            plt.draw()
            plt.pause(0.1)

        # Phase 1 of backpropagation
        dcost_dao = ao - train_labels
        dao_dzo = sigmoid_der(zo)
        dzo_dwo = ah

        dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

        # Phase 2 of backpropagation
        dcost_dzo = dcost_dao * dao_dzo
        dzo_dah = wo
        #print('dcost_dzo: ' , dcost_dzo.shape)
        #print('dzo_dah.T: ' , dzo_dah.T)
        dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
        dah_dzh = sigmoid_der(zh) 
        dzh_dwh = train_features
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

        # Update Weights

        wh -= lr * dcost_wh
        wo -= lr * dcost_wo

    plt.show(block=True)    
    print("Predictability: ", test())

############################################
np.random.seed(0)

# Main program
train_features, train_labels, validation_features, validation_labels, test_features, test_labels = parse_file()

# Generate random weight array for hidden and output layer
wh = np.random.rand(len(train_features[0]), 10)
wo = np.random.rand(10, 1)

# Weights from the best epoch according to the validation function
best_wh = 0
best_wo = 0
smallest_error = 1000
last_error = 0
res = 0

# Initial learning rate
lr = 0.005

train(25000, 100)
