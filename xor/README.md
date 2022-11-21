# This is Title 

NNxor.c

## This is subtitle

1. We start by defining a couple of helper functions, including the activation function and its corresponding derivative. A third function is used to initialize weights between 0.0 and 1.0.

2. we define the dimensions of the network and allocate the arrays for the layers, biases and weights.

3. back-propagation algorithm

4. The main loop first iterates over a set number of epochs (10,000 in this case), and foreach epoch it picks one pair of inputs and expected outputs to operate on. Because SGD requires that input/output pairs are randomized, I shuffle an array of indexes every epoch in order to select a random pair, while making sure that all the inputs are used per epoch.

5. First portion is to calculate the output of the network given the current weights starting with calculating the hidden layer activation.

6. we compute the output layer activation

7. calculating a small incremental change in the network weights that will move the network towards minimizing the error of the output that the network just computed. 

8. For the hidden layer it is a similar process, with the exception that the error calculation for a given hidden node is the sum of the error across all output nodes (with the appropriate weight applied to it.

9. the final step is applying them to their respective weight matrices and bias units .

-------------------------------------
# Neural Network in C

This is assignment 1

## Compile & Run

1. 用 Visual Studio Code按執行跑出結果、而非使用ubuntu下指令去執行
2. 有使用wsl去開啟Visual Studio Code
