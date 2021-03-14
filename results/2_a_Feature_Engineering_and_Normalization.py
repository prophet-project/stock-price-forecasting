#!/usr/bin/env python
# coding: utf-8

# In[1]:


# plotly standard imports
import plotly.graph_objs as go
import chart_studio.plotly as py

# Cufflinks wrapper on plotly
import cufflinks

# Data science imports
import pandas as pd
import numpy as np

# Options for pandas
pd.options.display.max_columns = 30

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from plotly.offline import iplot, init_notebook_mode
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# Set global theme
cufflinks.set_config_file(world_readable=True, theme='pearl')


# # Load dataset
# And get interesting features

# In[16]:


from src.load_datasets import load_datasets
from src.prepare_datasets import feature_list

train, test = load_datasets()

train_features = train[feature_list]
test_features = test[feature_list]

train_features.index = train['Date']
test_features.index = test['Date']

train_features


# # Data featuring
# 
# In theory we are going to use 4 features: The price itself and three extra technical indicators.
# 
# MACD (Trend)
# Stochastics (Momentum)
# Average True Range (Volume)
# 
# ## Functions
# 
# **Exponential Moving Average**: Is a type of infinite impulse response filter that applies weighting factors which decrease exponentially. The weighting for each older datum decreases exponentially, never reaching zero.
# 
# **MACD**: The Moving Average Convergence/Divergence oscillator (MACD) is one of the simplest and most effective momentum indicators available. The MACD turns two trend-following indicators, moving averages, into a momentum oscillator by subtracting the longer moving average from the shorter moving average.
# 
# **Stochastics oscillator**: The Stochastic Oscillator is a momentum indicator that shows the location of the close relative to the high-low range over a set number of periods.
# 
# **Average True Range**: Is an indicator to measure the volalitility (NOT price direction). The largest of:
# 
# - Method A: Current High less the current Low
# - Method B: Current High less the previous Close (absolute value)
# - Method C: Current Low less the previous Close (absolute value)

# In[3]:


from src.indicators import MACD, stochastics_oscillator, ATR


# In[4]:


days_to_show = 120


# ## MACD

# In[6]:


macd = MACD(train_features['Close'][-days_to_show:], 12, 26, 9)

pd.DataFrame({'MACD': macd}).iplot()


# ## Stochastics Oscillator

# In[7]:


stochastics = stochastics_oscillator(train_features['Close'][-days_to_show:], 14)

pd.DataFrame({'Stochastics Oscillator': stochastics}).iplot()


# ## Average True Range

# In[8]:


atr = ATR(train_features.iloc[-days_to_show:], 14)

atr.head()

atr.iplot()


# ## Check for normal distribution

# In[10]:


import scipy.stats as stats
import pylab

close_change = train_features['Close'].pct_change()[1:]
close_change.head()

stats.probplot(close_change, dist='norm', plot=pylab)


# ### Check time relation
# 
# Check depenence of trading and price from date in year and time of day
# 
# #### Firstly define function for display frequiency

# In[11]:


import tensorflow as tf
import matplotlib.pyplot as plt

def plot_log_freaquency(series):
    fft = tf.signal.rfft(series)    
    f_per_dataset = np.arange(0, len(fft))

    n_samples_d = len(series)
    days_per_year = 365
    years_per_dataset = n_samples_d/(days_per_year)

    f_per_year = f_per_dataset/years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.xticks([1, 365], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')


# #### Frequency of price

# In[13]:


plot_log_freaquency(train_features['Close'])


# #### Frequency of transaction volume

# In[15]:


plot_log_freaquency(train_features['Volume'])


# ## Compare train and test datasets

# In[17]:


import sweetviz as sv

compare_report = sv.compare([train_features, 'Train data'], [test_features, 'Test data'], "Close")
compare_report.show_notebook()


# ### Training data exploration

# In[18]:


train_features.iplot(subplots=True)


# ### Testing data exploration

# In[20]:


test_features


# In[21]:


test_features.iplot(subplots=True)


# ## Normalise data
# 
# Will use only training mean and deviation for not give NN access to test dataset
#  
# Subtract the mean and divide by the standard deviation of each feature will give required normalisation
# 

# In[22]:


train_mean = train_features.mean()
train_std = train_features.std()

train_normalised = (train_features - train_mean) / train_std
test_normalised = (test_features - train_mean) / train_std

train_normalised.head()

train_normalised.index = train_features.index
train_normalised.iplot(subplots=True, title="Train")

test_normalised.index = test_features.index
test_normalised.iplot(subplots=True, title="Test")


# ### Normalisation based on max-min

# In[23]:


normalised_min_max = (train_features - train_features.mean()) / (train_features.max() - train_features.min())

normalised_min_max.head()

normalised_min_max.iplot(subplots=True)


# Normalisation for testing must be based on train mean

# In[24]:


normalised_min_max_test = (test_features - train_features.mean()) / (train_features.max() - train_features.min())

normalised_min_max_test.head()

normalised_min_max_test.iplot(subplots=True)


# In[25]:


feature2normaliesd = pd.DataFrame({ 'Real': train_features['Close'], 'Normalised': train_normalised['Close']})
feature2normaliesd.index = train_features.index

feature2normaliesd.iplot(subplots=True)


# In[ ]:




