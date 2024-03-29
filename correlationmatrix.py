from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from io import BytesIO
from google.colab import files
uploaded_file = files.upload()
for name in uploaded_file.keys():
    filename = name
datasetframe = pd.read_csv(BytesIO(uploaded_file[filename]))

pearsoncorr = datasetframe.corr(method='pearson')

# plotting correlation heatmap
plt.figure(figsize = (60,60))
dataplot = sb.heatmap(pearsoncorr, cmap="YlGnBu", annot=True)

# displaying heatmap
plt.show()

from scipy.stats.stats import kendalltau
kendallcorr = datasetframe.corr(method='kendall')

# plotting correlation heatmap
plt.figure(figsize = (60,60))
dataplot = sb.heatmap(kendallcorr, cmap="YlGnBu", annot=True)

# displaying heatmap
plt.show()

spearcorr = datasetframe.corr(method='spearman')

# plotting correlation heatmap
plt.figure(figsize = (60,60))
dataplot = sb.heatmap(spearcorr, cmap="YlGnBu", annot=True)

# displaying heatmap
plt.show()

datasetframe['Yield'].hist()
