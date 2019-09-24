# Keras Deep Learning Model
### Building a Concrete Compressive Strength Model using Deep Learning Keras Library

1. [Building and optimizing a regression model using Keras deep learning library](https://msyazwan.github.io/Keras-Deep-Learning-Model/Concrete-Strength-Keras)
+ Build a base model
+ Normalize features value
+ Increase epochs
+ Increase hidden layer
...MODEL:

|Step |Hidden Layers|Nodes|Activation Function|Optimizer|Loss Function     |Epochs|
|-----|-------------|-----|-------------------|---------|------------------|------|
|A    |3            |25   |ReLU               |Adam     |Mean Squared Error|25    |
|B    |3            |25   |ReLU               |Adam     |Mean Squared Error|25    |
|C    |3            |25   |ReLU               |Adam     |Mean Squared Error|25    |
|D    |3            |25   |ReLU               |Adam     |Mean Squared Error|25    |

3 hidden layers with 10 nodes, ReLu activation function, adam optimizer, mse loss function and 100 epochs
Model Evaluation: Mean Squared Error

2. Building a regression model using Keras deep learning library Version 2.0
FINAL MODEL: 7 hidden layers with 25 nodes, ReLu activation function, adam optimizer, mse loss function and 250 epochs
MODEL EVALUATION: Mean Squared Error = 31.35 R^2 = 0.89

3. Predict a new value using pre-train model
