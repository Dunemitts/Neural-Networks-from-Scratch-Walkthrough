import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases # The first element you pass in dot() will be how the output is "indexed", just trust me put weights first or you'll get an error
print(output)
'''
np.dot breakdown.
np.dot(weights, inputs) = [np.dot(weights[0], inputs), np.dot(weights[1], inputs), np.dot(weights[2], inputs)] = [2.8, -1.79, 1.885]

np.dot(weights, inputs) + biases = [2.8, -1.79, 1.885] + [2.0, 3.0, 0.5] = [4.8, 1.21, 2.385] which is the output
'''

'''
weight vs bias.
some_value= -0.5
weight = 0.7 # Changes magnitude
bias = 0.7 # Offsets it

print(some_value*weight) # Outputs -0.35 (keeps negative as it's changing the magnitude)
print(some_value*bias) # Outputs 0.1999...6 (notice how bias makes it a positive number as it just offsets the output)
'''

'''
Matrix ex.
Array:
lol = 
[[1,5,6,2],
[3,2,1,3]]

Shape:
(how many big lists in the lol, how many small lists in a big list, how many numbers in a single list)
(1, 2, 4) # Usually wouldn't put 1 for the first number if there's only 1

Type: 2D Array, Matrix

3D Shape ex.
Array: 
lolol = 
[[[1,5,6,2],
[3,2,1,3]]
[[5,2,1,2],
[6,4,8,2]]
[[2,8,5,3],
[1,1,9,4]]]

Shape:
(3, 2, 4)

Type: 3D Array
'''