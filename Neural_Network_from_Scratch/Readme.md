# Practice Lab Neural Network 1

## Student Information:
- **Name: Yash Anil Mali**
- **Roll Number: 41**
- **Batch: DL2**
- **Date of Submission: 20-01-2025**

## Neural Network Implementation from Scratch

### Objective:
Implement a simple feedforward neural network from scratch in Python without using any in-built deep learning libraries. This implementation will focus on basic components like forward pass, backward propagation (backpropagation), and training using gradient descent.

## Problem Definition

### Dataset:
The dataset used for training and testing the neural network is a simple XOR dataset:
- **Inputs:** `[[0, 0], [0, 1], [1, 0], [1, 1]]`
- **Outputs:** `[[0], [1], [1], [0]]`
- **Task:**
  - The task is a classification problem where the goal is to correctly predict the XOR output for given binary inputs.

## Methodology

### Neural Network Architecture:
- **Input Layer:** 2 neurons (corresponding to the two input features)
- **Hidden Layer:** 4 neurons with sigmoid activation function
- **Output Layer:** 1 neuron with sigmoid activation function for binary classification

### Forward Pass:
- The data moves through the network as follows:
  1. Compute weighted sums at the hidden layer: `hidden_input = dot(X, weights_input_hidden) + bias_hidden`
  2. Apply the sigmoid activation: `hidden_output = sigmoid(hidden_input)`
  3. Compute weighted sums at the output layer: `output_input = dot(hidden_output, weights_hidden_output) + bias_output`
  4. Apply the sigmoid activation: `output = sigmoid(output_input)`

### Backpropagation:
- Compute the output layer error: `output_error = output - y`
- Calculate the gradient for the output layer: `output_gradient = output_error * sigmoid_derivative(output)`
- Compute the hidden layer error: `hidden_error = dot(output_gradient, weights_hidden_output.T)`
- Calculate the gradient for the hidden layer: `hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)`
- Update weights and biases using gradient descent:
  1. Update `weights_hidden_output` and `bias_output`
  2. Update `weights_input_hidden` and `bias_hidden`

### Loss Function:
- **Mean Squared Error (MSE):** `loss = mean((output - y)^2)`

### Optimization:
- **Gradient Descent:** Adjust weights and biases by subtracting the gradients scaled by the learning rate

## Declaration
I, Yash Anil Mali, confirm that the work submitted in this assignment is my own and has been completed following academic integrity guidelines. The code is uploaded on my GitHub repository account, and the repository link is provided below:

**GitHub Repository Link:** (https://github.com/669yash/Deep-Learning/tree/main/Neural_Network_from_Scratch)

**Signature:** Yash Anil Mali
