# Keras Deep Learning Model
### Building a Concrete Compressive Strength Model using Deep Learning Keras Library

**1. Building and optimizing a regression model using Keras deep learning library[Link]**(https://msyazwan.github.io/Keras-Deep-Learning-Model/Concrete-Strength-Keras)
+ Build a base model
+ Normalize features value
+ Increase epochs
+ Increase hidden layer

 **FINAL MODEL:**

|Step |Input         |Hidden Layers|Nodes|Activation Function|Optimizer|Loss Function     |Epochs |
|-----|--------------|-------------|-----|-------------------|---------|------------------|-------|
|A    |Original Value|1            |10   |ReLU               |Adam     |Mean Squared Error|50     |
|B    |Normalized    |1            |10   |ReLU               |Adam     |Mean Squared Error|50     |
|C    |Normalized    |1            |10   |ReLU               |Adam     |Mean Squared Error|100    |
|D    |Normalized    |3            |10   |ReLU               |Adam     |Mean Squared Error|100    |

**2. Building a regression model using Keras deep learning library Version 2.0[Link]**(https://msyazwan.github.io/Keras-Deep-Learning-Model/Concrete-Strength-Keras-v2)

 **FINAL MODEL:**

|Input     |Hidden Layers|Nodes|Activation Function|Optimizer|Loss Function     |Epochs |
|----------|-------------|-----|-------------------|---------|------------------|-------|
|Normalized|7            |25   |ReLU               |Adam     |Mean Squared Error|250    |

**3. Predict a new value using pre-train model[Link]**(https://msyazwan.github.io/Keras-Deep-Learning-Model/Concrete-Strength-Keras-v2)
