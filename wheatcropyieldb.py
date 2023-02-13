# Upload a correct file from your local machine
from io import BytesIO
from google.colab import files
uploaded_file = files.upload()
for name in uploaded_file.keys():
    filename = name
data = BytesIO(uploaded_file[filename])

uploaded_file = files.upload()
for name in uploaded_file.keys():
    filename = name
data2 = BytesIO(uploaded_file[filename])

from google.colab import drive
drive.mount('/content/drive')

#Function launch_fe is acquired code from AWS


def launch_fe(data):
    import os
    import pandas as pd
    from io import StringIO
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction import text
    import pickle
    from scipy import sparse
    MAX_TEXT_FEATURES = 200
    columns_list = ["apparentTemperatureMax", "apparentTemperatureMin", "dewPoint", "humidity", "precipIntensityMax", "temperatureMax", "temperatureMin", "Yield"]

    dataset = pd.read_csv(data, skipinitialspace=True)
    num_samples = len(dataset)

    # Move the label column
    cols = list(dataset.columns)
    colIdx = dataset.columns.get_loc("Yield")
    # Do nothing if the label is in the 0th position
    # Otherwise, change the order of columns to move label to 0th position
    if colIdx != 0:
        cols = cols[colIdx:colIdx+1] + cols[0:colIdx] + cols[colIdx+1:]
        dataset = dataset[cols]


    # Write train and test csv
    dataset.to_csv('train.csv', index=False, header=False)
    
def get_model_id():
    return "None"

def launch_fe2(data):
    import os
    import pandas as pd
    from io import StringIO
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction import text
    import pickle
    from scipy import sparse
    MAX_TEXT_FEATURES = 200
    columns_list = ["apparentTemperatureMax", "apparentTemperatureMin", "dewPoint", "humidity", "precipIntensityMax", "temperatureMax", "temperatureMin", "Yield"]

    dataset = pd.read_csv(data, skipinitialspace=True)
    num_samples = len(dataset)

    # Move the label column
    cols = list(dataset.columns)
    colIdx = dataset.columns.get_loc("Yield")
    # Do nothing if the label is in the 0th position
    # Otherwise, change the order of columns to move label to 0th position
    if colIdx != 0:
        cols = cols[colIdx:colIdx+1] + cols[0:colIdx] + cols[colIdx+1:]
        dataset = dataset[cols]


    # Write train and test csv
    dataset.to_csv('test.csv', index=False, header=False)

#End of acquired code
    
def get_model_id():
    return "None"

# Launch FE
launch_fe(data)
launch_fe2(data2)

import pandas as pd
import numpy as np

# Load the test and train datasets
train = pd.read_csv('train.csv', skipinitialspace=True, header=None)
test = pd.read_csv('test.csv', skipinitialspace=True, header=None)

# LINEAR REGRESSION

print ("LINEAR REGRESSION!")



# import the library of the algorithm
from sklearn.linear_model import LinearRegression

# Initialize hyperparams
fit_intercept = True
normalize = False

# Initialize the algorithm
model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)

model.fit(train.iloc[:,1:], train.iloc[:,0])


# Predict the target values
y_pred = model.predict(test.iloc[:, 1:])
# calculate rmse
rmse = np.sqrt(np.mean((y_pred - test.iloc[:, 0])**2))
print('RMSE of the model is: ', rmse)
# import the library to calculate mae
from sklearn.metrics import mean_absolute_error
# calculate mae
mae = mean_absolute_error(np.array(test.iloc[:, 0]), y_pred)
print('MAE of the model is: ', mae)

mape = np.mean(np.abs((np.array(test.iloc[:, 0]) - y_pred)/np.array(test.iloc[:, 0])))*100
print (mape)

#POLYNOMIAL REGRESSOR
 
print ("POLYNOMIAL REGRESSION")

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train.iloc[:,1:], train.iloc[:,0])

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(train.iloc[:,1:])
X_poly_test = poly_reg.fit_transform(test.iloc[:,1:])

pol_reg = LinearRegression()
pol_reg.fit(X_poly, train.iloc[:,1])


import numpy as np
# Predict the target values
y_pred = pol_reg.predict(X_poly_test)
# calculate rmse
rmse = np.sqrt(np.mean((y_pred - test.iloc[:, 0])**2))
print('RMSE of the model is: ', rmse)
# import the library to calculate mae
from sklearn.metrics import mean_absolute_error
# calculate mae
mae = mean_absolute_error(np.array(test.iloc[:, 0]), y_pred)
print('MAE of the model is: ', mae)

mape = np.mean(np.abs((np.array(test.iloc[:, 0]) - y_pred)/np.array(test.iloc[:, 0])))*100
print (mape)

#LINEAR SVR
print ("LINEAR SVR!")


# import the library of the algorithm
from sklearn.svm import LinearSVR

# Initialize hyperparams
tol = 0.0001
fit_intercept = True
max_iter = 1000

# Initialize the algorithm
model = LinearSVR(random_state=0, tol=tol, fit_intercept=fit_intercept, max_iter=max_iter)

model.fit(train.iloc[:,1:], train.iloc[:,0])


import numpy as np
# Predict the target values
y_pred = model.predict(test.iloc[:, 1:])
# calculate rmse
rmse = np.sqrt(np.mean((y_pred - test.iloc[:, 0])**2))
print('RMSE of the model is: ', rmse)
# import the library to calculate mae
from sklearn.metrics import mean_absolute_error
# calculate mae
mae = mean_absolute_error(np.array(test.iloc[:, 0]), y_pred)
print('MAE of the model is: ', mae)

mape = np.mean(np.abs((np.array(test.iloc[:, 0]) - y_pred)/np.array(test.iloc[:, 0])))*100
print (mape)

#RANDOM FOREST

print ("RANDOM FOREST!!")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Initialize hyperparams
max_depth = [None, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
n_estimators = 10

min_rmse = 10000000
mdfinal = 10000000
nefinal = 10000000
min_mae = 10000000
min_mape = 100000

for n_e in range (1,100, +2):
  for m_d in max_depth:
    # Initialize the algorithm
    model = RandomForestRegressor(max_depth=m_d, random_state=0, n_estimators=n_e)
    model.fit(train.iloc[:,1:], train.iloc[:,0])

    # Predict the target values
    y_pred = model.predict(test.iloc[:, 1:])
    # calculate rmse
    rmse = np.sqrt(np.mean((y_pred - test.iloc[:, 0])**2))


    if rmse <min_rmse:
      min_rmse = rmse
      mdfinal = m_d
      nefinal = n_e
      min_mae = mean_absolute_error(np.array(test.iloc[:, 0]), y_pred)
      min_mape = np.mean(np.abs((np.array(test.iloc[:, 0]) - y_pred)/np.array(test.iloc[:, 0])))*100

print (min_rmse, min_mae, mdfinal,nefinal)
print (min_mape)

#K NEAREST NEIGHBOURS

print ("K NEAREST NEIGHBOURS")


# import the library of the algorithm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np


# Initialize hyperparams

min_n = 1000000
min_rmse = 1000000
min_mae = 1000000

for n_neighbors in range (2,100,+1):
  model = KNeighborsRegressor(n_neighbors=n_neighbors)
  model.fit(train.iloc[:,1:], train.iloc[:,0])


  y_pred = model.predict(test.iloc[:, 1:])
  rmse = np.sqrt(np.mean((y_pred - test.iloc[:, 0])**2))

  if rmse < min_rmse:
    min_rmse=rmse
    min_mae = mean_absolute_error(np.array(test.iloc[:, 0]), y_pred)
    min_n = n_neighbors
    min_mape = np.mean(np.abs((np.array(test.iloc[:, 0]) - y_pred)/np.array(test.iloc[:, 0])))*100

print (min_rmse,min_mae,min_n)
print (min_mape)

#GRADIENT BOOSTING REGRESSOR

file1 = open("/content/drive/Shareddrives/1:1 Meenakshi Nair/CropYieldB_GBR.txt", "a")  # append mode

print ("GRADIENT BOOSTING REGRESSION!!")

from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np

learningratelist = [0.001, 0.005, 0.007, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.045, 0.05, 0.055]

min_rmse= 1000000
min_md= 10000000
min_lr = 1000000
min_ne = 1000000
min_mae = 1000000


for m_d in range (5,50):
  for l_r in learningratelist:
    for n_e in range (300,800,+20):
      gbr = GradientBoostingRegressor(n_estimators=n_e, max_depth=m_d, learning_rate=l_r, min_samples_split=3)

      gbr.fit(train.iloc[:,1:], train.iloc[:,0])

      y_pred = gbr.predict(test.iloc[:,1:])
      # calculate rmse
      rmse = np.sqrt(np.mean((y_pred - test.iloc[:, 0])**2))
      # import the library to calculate mae

      file1.write(str(rmse))
      file1.write("\n")
      file1.flush()

      if rmse < min_rmse:
        min_rmse=rmse
        min_md=m_d
        min_lr = l_r
        min_ne = n_e
        min_mae = mean_absolute_error(np.array(test.iloc[:, 0]), y_pred)
        min_mape = np.mean(np.abs((np.array(test.iloc[:, 0]) - y_pred)/np.array(test.iloc[:, 0])))*100

print (min_rmse)
print (min_md, min_lr, min_ne)
print (min_mae)
print (min_mape)

file1.write(min_rmse+" "+min_md+" "+min_lr+" "+min_ne)


file1.close()

#RNN

print ("RNN!!")

file1 = open("/content/drive/Shareddrives/1:1 Meenakshi Nair/CropYieldB_RNN.txt", "a")  # append mode


import pandas as pd
import numpy as np

# Load the test and train datasets
train = pd.read_csv('train.csv', skipinitialspace=True, header=None)
test = pd.read_csv('test.csv', skipinitialspace=True, header=None)


x_training_data = np.array(train.iloc[:,1:])

y_training_data = np.array(train.iloc[:,0])


x_training_data = np.reshape(x_training_data, (x_training_data.shape[0],  x_training_data.shape[1],  1))

from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dropout

rnn = Sequential()


for i in [True, True, False]:

    rnn.add(LSTM(units = 45, return_sequences = i))

    rnn.add(Dropout(0.2))

rnn.add(Dense(units = 1))

rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

min_epochs = 10000000
min_batch_size = 10000000
min_rmse = 10000000
min_mae = 10000000

for epochs in range (40,200,+20):
  for batch_size in range (8,50,+4):
    rnn.fit(x_training_data, y_training_data, epochs = epochs, batch_size = batch_size)

    x_testing_data = np.array(test.iloc[:,1:])


    # Predict the target values
    y_pred = rnn.predict(x_testing_data)
    y_testing_data = np.array(test.iloc[:,0])
    

    # calculate rmse
    rmse = np.sqrt(np.mean((y_pred - y_testing_data)**2))

    file1.write(str(rmse))
    file1.write("\n")
    file1.flush()


    if rmse < min_rmse:
      min_rmse = rmse
      min_epochs = epochs
      min_batch_size = batch_size
      min_mae = mean_absolute_error(np.array(test.iloc[:, 0]), y_pred)
      min_mape = np.mean(np.abs((np.array(test.iloc[:, 0]) - y_pred)/np.array(test.iloc[:, 0])))*100
    
    print (min_rmse, min_epochs, min_batch_size, min_mae, min_mape)

print (min_rmse,min_epochs,min_batch_size)
print (min_mae)
print (min_mape)

file1.write(min_rmse+" "+min_md+" "+min_lr+" "+min_ne)


file1.close()
