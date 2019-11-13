In this part 2 notebook, we will increase all the network properties to achieve a higher model aacuracy. Later, the model will be saved, load and predict new concrete strength with new user defined parameters.

<h2><center> Building a Concrete Compressive Strength Model using Deep Learning Keras Library </center></h2>

<img src = "tf-keras.png" width = 500>


```python
import pandas as pd
import numpy as np
```


```python
url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
```


```python
df = pd.read_csv(url)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
      <th>Strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>79.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>61.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
      <td>40.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
      <td>41.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
      <td>44.30</td>
    </tr>
  </tbody>
</table>
</div>



### MODEL - Construct Model to Predict and Forecast.

#### Split Data to Predictors and Target


```python
X = df.iloc[:,:-1]
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = df.iloc[:,-1]
y.head()
```




    0    79.99
    1    61.89
    2    40.27
    3    41.05
    4    44.30
    Name: Strength, dtype: float64



Normalizing Data


```python
X_norm = (X - X.mean()) / X.std()
X_norm.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.476712</td>
      <td>-0.856472</td>
      <td>-0.846733</td>
      <td>-0.916319</td>
      <td>-0.620147</td>
      <td>0.862735</td>
      <td>-1.217079</td>
      <td>-0.279597</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.476712</td>
      <td>-0.856472</td>
      <td>-0.846733</td>
      <td>-0.916319</td>
      <td>-0.620147</td>
      <td>1.055651</td>
      <td>-1.217079</td>
      <td>-0.279597</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.491187</td>
      <td>0.795140</td>
      <td>-0.846733</td>
      <td>2.174405</td>
      <td>-1.038638</td>
      <td>-0.526262</td>
      <td>-2.239829</td>
      <td>3.551340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.491187</td>
      <td>0.795140</td>
      <td>-0.846733</td>
      <td>2.174405</td>
      <td>-1.038638</td>
      <td>-0.526262</td>
      <td>-2.239829</td>
      <td>5.055221</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.790075</td>
      <td>0.678079</td>
      <td>-0.846733</td>
      <td>0.488555</td>
      <td>-1.038638</td>
      <td>0.070492</td>
      <td>0.647569</td>
      <td>4.976069</td>
    </tr>
  </tbody>
</table>
</div>



Save number of feature columns, <strong><i>n_cols</i></strong> to use later in model development.


```python
n_cols = X.shape[1]
n_cols
```




    8



#### Importing Libraries


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import keras

from keras.models import Sequential
from keras.layers import Dense
```

    Using TensorFlow backend.
    

### Building the Model

<strong>Network Properties:</strong>
<ul>
  <li>Hidden Layers: 7</li>
  <li>Nodes: 25</li>
  <li>Activation Function: ReLU</li>
  <li>Optimizer: Adam</li>
  <li>Loss Function: Mean Squared Error</li>
  <li>Epochs: 250</li>
</ul>


```python
mse = []
r2 = []

for i in range(50):
    
    #Split Data to Train and Test Set
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.3)

    #Create model
    model = Sequential()
    model.add(Dense(25, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))

    #Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #fit the model
    model.fit(X_train, y_train, epochs=250, verbose=0)

    #predict output on test set
    y_pred = model.predict(X_test)
    
    mse.append(mean_squared_error(y_test, y_pred))
    r2.append(r2_score(y_test, y_pred))
```


```python
print('mse_Mean: {:.2f}'.format(np.mean(mse)))
print('mse_StdDev: {:.2f}'.format(np.std(mse)))
```

    mse_Mean: 32.41
    mse_StdDev: 5.47
    


```python
print('R^2_Mean: {:.2f}'.format(np.mean(r2)))
print('R^2_StdDev: {:.2f}'.format(np.std(r2)))
```

    R^2_Mean: 0.88
    R^2_StdDev: 0.02
    


```python
from IPython.display import HTML, display
import tabulate

tabletest = [['PART','MSE: Mean','MSE: StdDev','R^2: Mean','R^2: StdDev'],
         ['2', round(np.mean(mse),2), round(np.std(mse),2), round(np.mean(r2),2), round(np.std(r2),2)]]

display(HTML(tabulate.tabulate(tabletest, tablefmt='html')))
```


<table>
<tbody>
<tr><td>PART</td><td>MSE: Mean</td><td>MSE: StdDev</td><td>R^2: Mean</td><td>R^2: StdDev</td></tr>
<tr><td>2   </td><td>32.41    </td><td>5.47       </td><td>0.88     </td><td>0.02       </td></tr>
</tbody>
</table>


#### Comparing the results from Part 1:
- mean squared error has gone down from 95.28 to 32.41
- R^2 has gone up from 0.65 to 0.88

<b>which means the overall accuracy has gone up compared from the previous run.</b>

### SAVING MODEL


```python
model.save('keras_reg.h5')
```

# Predict New Concrete Strength Value with Pretrained Model

## Import Module


```python
import numpy as np
from keras.models import load_model
```

## Load Pretrained Model


```python
keras_reg = load_model('keras_reg.h5')
```

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

    The predicted concrete strength is 71.87
    
