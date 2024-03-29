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
    
def get_model_id():
    return "None"

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

# Launch FE
launch_fe(data)
launch_fe2(data2)

import pandas as pd
import numpy as np

# Load the test and train datasets
train = pd.read_csv('train.csv', skipinitialspace=True, header=None)
test = pd.read_csv('test.csv', skipinitialspace=True, header=None)

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=26)
model.fit(train.iloc[:,1:], train.iloc[:,0])

# Predict the target values
y_pred = model.predict(test.iloc[:, 1:])

plt.scatter(x=test.iloc[:, 0], y=y_pred)

x = np.linspace(10, 80, 80)
y = x 
 
plt.plot(x, y)

plt.yticks([10,15,20,25,30,35,40,45,50,55,60,65,70,75,80])
plt.xticks([10,15,20,25,30,35,40,45,50,55,60,65,70,75,80])


plt.xlabel("Actual Crop Yield")
plt.ylabel("Predicted Crop Yield")

plt.show()

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=26)
model.fit(train.iloc[:,1:], train.iloc[:,0])


# Predict the target values
y_pred = model.predict(test.iloc[:, 1:])

plt.scatter(x=test.iloc[:, 0], y=y_pred-test.iloc[:, 0])

plt.yticks([-60,-50,-40,-30,-20,-10,0,10,20,30,40, 50,  60])
plt.xticks([-60,-50,-40,-30,-20,-10,0,10,20,30,40, 50,  60])

plt.axhline(y = 0, color = 'k', linestyle = '-')

plt.xlabel("Actual Crop Yield")
plt.ylabel("Crop Yield Error")

plt.show()
