# Predict New Concrete Strength Value with Pretrained Model

## Import Module


```python
import numpy as np
from keras.models import load_model
```

    Using TensorFlow backend.
    

## Load Pretrained Model


```python
keras_reg = load_model('keras_reg.h5')
```

    WARNING:tensorflow:From C:\Users\syazwan\Anaconda3\envs\pyvirtual\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From C:\Users\syazwan\Anaconda3\envs\pyvirtual\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    

## Insert New Parameters


```python
#Cement
x1 = 1540.0

#Blast Furnace Slag
x2 = 250.0

#Fly Ash
x3 = 0.0

#Water
x4 = 162.0

#Superplasticizer
x5 = 2.5

#Coarse Aggregate
x6 = 1055.0

#Fine Aggregate
x7 = 676.0

#Age
x8 = 450
```


```python
X = np.array([[x1, x2, x3, x4, x5, x6, x7, x8]])
X_norm = (X - X.mean()) / X.std()
strength_pred = keras_reg.predict(X_norm)
```

## New Prediction


```python
print('The predicted concrete strength is {:.2f}'.format(strength_pred[0][0]))
```

    The predicted concrete strength is 79.01
    
