import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): # n_inputs and n_neurons are taking in the batch size and how many neurons we want 
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # Something like this: print(0.10 * np.random.randn(4, 3)) would print a matrix of 4 rows and 3 columns with a weight of 0.10
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5) # With how the function is set up, the neurons can be anything, so I chose 5
layer2 = Layer_Dense(5, 2) # With the second layer, the input HAS to be the same as the number of neurons chosen, so 5

layer1.forward(X)
print(layer1.output) # Gives us a matrix with 3 rows and 5 columns (3 rows because of the X input)
layer2.forward(layer1.output)
print(layer2.output) # Gives us our final 2 neurons at the end of the neural network with 3 columns (from X)


'''
Old Matrix Breakdown:
inputs = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
           [-.05, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_output = np.dot(inputs, np.array(weights).T) + biases # Transposing the weights as without it, np.dot will try to multiply input's top horizontal row (3,4) with weight's left vertical row (4,3), transposing just makes it so that the weight's new left vertical row is (3,4) rather than (4,3)
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2 # layer1_output becomes the new input for the next layer

print(layer2_output) # Outputs a (3,3) Matrix
'''