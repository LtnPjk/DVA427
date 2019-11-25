#https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-adding-hidden-layers/

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

np.random.seed(0)
feature_set, labels = datasets.make_moons(100, noise=0.10)
#plt.figure(figsize=(10,7))
#plt.scatter(feature_set[:,0], feature_set[:,1], c=labels)
#plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)

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
    if i == 3:
        #print(patient_att)
        break

# Take the first 18 attributes as training input
training_inp = [attribute[0:18] for attribute in patient_att]
# Take the last attribute as training output (target)
training_oup = [[attribute[-1] for attribute in patient_att]]
# Reverse the list

feature_set = np.array([[float(j) for j in i] for i in training_inp])
labels = np.array([[float(j) for j in i] for i in training_oup]).T

print(feature_set.shape)
print(labels.shape)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

wh = np.random.rand(len(feature_set[0]), 8) 
wo = np.random.rand(8, 1)
lr = 0.5

for epoch in range(100):
    # feedforward
    zh = np.dot(feature_set, wh)
    ah = sigmoid(zh)

    zo = np.dot(ah, wo)
    ao = sigmoid(zo)

    # Phase1 =======================

    error_out = ((1 / 2) * (np.power((ao - labels), 2)))
    print(error_out.sum())

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