#https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-adding-hidden-layers/

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

smallest_error = 1000

np.random.seed(0)
#feature_set, labels = datasets.make_moons(100, noise=0.10)
#plt.figure(figsize=(10,7))
#plt.scatter(feature_set[:,0], feature_set[:,1], c=labels)
#plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)

#labels = labels.reshape(100, 1)

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

validation_inp = feature_set[int(len(feature_set)*0.75):int(len(feature_set)*0.85)]
test_inp = feature_set[int(len(feature_set)*0.85):int(len(feature_set))]
feature_set = feature_set[0:int(len(feature_set)*0.75)]

validation_oup = labels[int(len(labels)*0.75):int(len(labels)*0.85)]
test_oup = labels[int(len(labels)*0.85):int(len(labels))]
labels = labels[0:int(len(labels)*0.75)]


print(validation_inp.shape)
print(test_inp.shape)
print(feature_set.shape)

#exit()

wh = np.random.rand(len(feature_set[0]), 14) 
wo = np.random.rand(14, 1)
lr = 0.03
old_wh = 0
old_wo = 0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def test():
     
    zh, ah, zo, ao, error_out = feed_forward(test_inp, test_oup)

    ao = [int(round_half_up(i, 0)) for i in ao]

    correct_guesses = 0

    for i, j in zip(ao, test_oup):
        if i == int(j):
            correct_guesses = correct_guesses + 1
    
    print("ANN predictability: ", correct_guesses/len(test_oup))

    #print(correct_guesses)
    #print(ao)
    #print(np.array(test_oup).T)

    exit()

def validation():
    global smallest_error, old_wh, old_wo, wh, wo
    #print("val_diff = ",smallest_error - error)
    ii, kk, xx, yy, validation_error = feed_forward(validation_inp, validation_oup)
    print("validation error = ",validation_error)
    if smallest_error == 1000 or smallest_error*1.50 > validation_error:
        smallest_error = validation_error
        old_wh = wh
        old_wo = wo
    else:
        print("validation interrupt\n")
        wh = old_wh
        wo = old_wo
        test()

def feed_forward(feature_set, labels):
    # feedforward
    zh = np.dot(feature_set, wh)
    ah = sigmoid(zh)

    zo = np.dot(ah, wo)
    ao = sigmoid(zo)

    # Phase1 =======================

    error_out = ((1 / 2) * (np.power((ao - labels), 2)))
    #print(error_out.sum())

    return zh, ah, zo, ao, error_out.sum()

for epoch in range(20000):
    zh, ah, zo, ao, error = feed_forward(feature_set, labels)

    # Validation

    #if epoch % 1 == 0:
        #validation()


    dcost_dao = ao - labels
    dao_dzo = sigmoid_der(zo) 
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

    # Phase 2 =======================

    # dcost_w1 = dcost_dah * dah_dzh * dzh_dw1
    # dcost_dah = dcost_dzo * dzo_dah
    dcost_dzo = dcost_dao * dao_dzo
    dzo_dah = wo
    #print('dcost_dzo: ' , dcost_dzo.shape)
    #print('dzo_dah.T: ' , dzo_dah.T)
    dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
    dah_dzh = sigmoid_der(zh) 
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    # Update Weights ================

    wh -= lr * dcost_wh
    wo -= lr * dcost_wo

    #print(dcost_wh)
    #print(dcost_wo)

test()
