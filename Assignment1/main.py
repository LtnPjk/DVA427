import numpy as np
import os
import sys

class NeuralNetwork():

    def __init__(self):
        # Seed the random number generator
        np.random.seed(1)

        # TODO
        # Convert to MLP, 9 nodes in hidden layer.
        

        # Set synaptic weights to a 3x1 matrix, from -1 to 1 and mean 0
        self.synaptic_weights = 2 * np.random.rand(3,1) - 1

    def sigmoid(self, x):
        # INPUT: Weighted sum of synapses
        # OUTPUT: Normalized weighted sum through sigmoid

        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # INPUT: Normalized weighted sum
        # OUTPUT: Derivative necessary to calculate weight adjustments

        return x * (1 - x)
    
    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):
            # Pass training set through the neural network
            output = self.think(training_inputs)

            # Calculate the error rate
            error = training_outputs - output
            
            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            # Adjust synaptic weights
            self.synaptic_weights += adjustments
            
    def think(self, inputs):
        """
        Pass inputs through the neural network to get output
        """
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

if __name__ == "__main__":

    # Initialize the single neuron neural network
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)


    # The training set, with 4 examples consisting of 3
    # input values and 1 output value
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    # Train the neural network
    neural_network.train(training_inputs, training_outputs, 10000)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    
    print("New situation: input data = ", A, B, C)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C])))
