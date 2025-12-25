# library for working with arrays
import numpy as np 

# sigmoid function formula
def sigmoid(x): 
  return 1 / (1 + np.exp(-x)) 

# input dataset
training_inputs = np.array(
              [[0,0,1],
              [1,1,1],
              [1,0,1],
              [0,1,1]]
              )
# expected results (targets)
training_outputs = np.array([[0,1,0,1]]).T 

# set the initial seed for the random number generator
np.random.seed(1) 

# initialize random weights for the synapses
synaptic_weights = 2 * np.random.random((3,1)) - 1 

print("Initial random weights:") 
print(synaptic_weights)  

# training loop (20,000 iterations)
for i in range(20000):
  # backpropagation method
  
  # pass data to a local variable (for safety)
  input_layer = training_inputs
  # compute the output using the sigmoid activation function
  outputs = sigmoid( np.dot(input_layer, synaptic_weights) )

  # subtract the activation function output from the expected results
  err = training_outputs - outputs
  # calculate the weight adjustment using the formula
  adjustments = np.dot( input_layer.T,  err * ( outputs * (1 - outputs) ) )
  # update the weights by adding the calculated adjustment
  synaptic_weights += adjustments 

print("Weights after training:")
print(synaptic_weights)

print("Output after training:")
print(outputs)

#TEST
new_input = np.array([1,0,0])
output = sigmoid( np.dot(new_input, synaptic_weights) )

print("Result:")
print(output)
