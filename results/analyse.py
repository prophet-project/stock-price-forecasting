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


# In[2]:



from plotly.offline import iplot, init_notebook_mode
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# Set global theme
cufflinks.set_config_file(world_readable=True, theme='pearl')


# ## Inspiration sources
# 
# https://github.com/BenjiKCF/Neural-Net-with-Financial-Time-Series-Data
# https://github.com/alberduris/SirajsCodingChallenges/tree/master/Stock%20Market%20Prediction

# ## Let's explore datasets

# ### Explore input dataset
# 
# Will use target dataset [Bitcoin in Cryptocurrency Historical Prices](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory?select=coin_Bitcoin.csv)
# 
# Bitcoin data at 1-day intervals from April 28, 2013

# In[3]:


from src.load_datasets import load_input_dataset

input_dataset = load_input_dataset()

input_dataset.head()


# Will explore full input dataset

# In[4]:


import sweetviz as sv

target_features = input_dataset[['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']]

analyse_report = sv.analyze([target_features, 'Bitcoin'], target_feat="Close")
analyse_report.show_notebook()


# In[5]:


target_features.head()


# Feature evalution over time

# In[6]:


datetime = pd.to_datetime(input_dataset['Date'])
target_features.index = datetime

target_features.iplot(
    subplots=True,
)


# In[7]:


target_features.describe().transpose()


# Will take only last 4 years, because they mostly interesting

# In[8]:


year = 365

years_count = 4
items_count = round(years_count * year)

last_years_dataset = input_dataset[-1 * items_count:]
last_years_datetime = pd.to_datetime(last_years_dataset['Date'])

last_years_dataset.head()
len(last_years_dataset)


# In[9]:


last_years_features = last_years_dataset[['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']]
last_years_features.index = last_years_datetime

last_years_features.iplot(
    subplots=True,
)


# ## Data featuring
# 
# In theory we are going to use 4 features: The price itself and three extra technical indicators.
# 
# MACD (Trend)
# Stochastics (Momentum)
# Average True Range (Volume)
# 
# ### Functions
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

# In[16]:


from src.indicators import MACD, stochastics_oscillator, ATR


# In[29]:


days_to_show = 120


# ## MACD

# In[31]:


macd = MACD(last_years_features['Close'][-days_to_show:], 12, 26, 9)

pd.DataFrame({'MACD': macd}).iplot()


# ## Stochastics Oscillator

# In[30]:


stochastics = stochastics_oscillator(last_years_features['Close'][-days_to_show:], 14)

pd.DataFrame({'Stochastics Oscillator': stochastics}).iplot()


# ## Average True Range

# In[56]:


atr = ATR(last_years_features.iloc[-days_to_show:], 14)

atr.head()

atr.iplot()


# ## Check for normal distribution

# In[15]:


import scipy.stats as stats
import pylab

close_change = last_years_features['Close'].pct_change()[1:]
close_change.head()

stats.probplot(close_change, dist='norm', plot=pylab)


# ### Check depenence of trading and price from date in year and time of day

# Firstly define function for display frequiency

# In[10]:


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


# Frequency of price

# In[11]:


plot_log_freaquency(last_years_dataset['Close'])


# Frequency of transaction volume

# In[12]:


plot_log_freaquency(last_years_dataset['Volume'])


# ## Compare train and test datasets

# In[40]:


from src.load_datasets import load_datasets

train_df, test_df = load_datasets()

train_df


# In[41]:


import sweetviz as sv

feature_list = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']

train_features = train_df[feature_list]
test_features = test_df[feature_list]

compare_report = sv.compare([train_features, 'Train data'], [test_features, 'Test data'], "Close")
compare_report.show_notebook()


# In[42]:


train_datetime = pd.to_datetime(train_df['Date'])
test_datetime = pd.to_datetime(test_df['Date'])

train_features.index = train_datetime
test_features.index = test_datetime


# ### Training data exploration

# In[43]:


train_features.iplot(subplots=True)


# ### Testing data exploration

# In[44]:


test_df


# In[45]:


test_features.iplot(subplots=True)


# ## Normalise data
# 
# Will use only training mean and deviation for not give NN access to test dataset
#  
# Subtract the mean and divide by the standard deviation of each feature will give required normalisation
# 

# In[46]:


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

# In[50]:


normalised_min_max = (train_features - train_features.mean()) / (train_features.max() - train_features.min())

normalised_min_max.head()

normalised_min_max.iplot(subplots=True)


# Normalisation for testing must be based on train mean

# In[51]:


normalised_min_max_test = (test_features - train_features.mean()) / (train_features.max() - train_features.min())

normalised_min_max_test.head()

normalised_min_max_test.iplot(subplots=True)


# In[20]:


feature2normaliesd = pd.DataFrame({ 'Real': train_features['Close'], 'Normalised': train_normalised['Close']})
feature2normaliesd.index = train_features.index

feature2normaliesd.iplot(subplots=True)


# ## Check window generator

# In[3]:


from src.prepare_datasets import get_prepared_datasets

train_df, test_df = get_prepared_datasets()

train_df.head()

train_df.iplot(subplots=True)


# ### Calculate batch size

# In[5]:


COUNT_BATCHES = 35 # divide full dataset on equal batches
LABEL_SHIFT = 1

len(train_df)

full_window_width = len(train_df) / (COUNT_BATCHES)
full_window_width

input_width = round(full_window_width - LABEL_SHIFT)
input_width

test_delimetor = round(len(test_df) / COUNT_BATCHES)
test_df = test_df[:test_delimetor*COUNT_BATCHES]

len(test_df)


# In[6]:


from src.window_generator import WindowGenerator

w1 = WindowGenerator(
    input_width=24, label_width=1, shift=24, 
    train_df=train_df, test_df=test_df, 
    label_columns=['Close']
)

w1


# In[22]:


w1.plot(plot_col='Close')


# In[23]:


w1.train.element_spec


# ## Try baseline model

# In[24]:


single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    train_df=train_df, test_df=test_df, 
    label_columns=['Close'])

single_step_window


# In[25]:


import tensorflow as tf
from src.BaselineModel import Baseline

column_indices = {name: i for i, name in enumerate(train_df.columns)}

baseline = Baseline(label_index=column_indices['Close'])
    
baseline.compile(
    loss=tf.losses.MeanSquaredError(),
    metrics=[tf.metrics.MeanAbsoluteError()]
)


performance = {}
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=1)


# In[9]:


wide_window = WindowGenerator(
    input_width=30, label_width=30, shift=1,
    train_df=train_df, test_df=test_df,
    label_columns=['Close'])

wide_window


# In[27]:


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)


# In[28]:


wide_window.plot(baseline)


# In[7]:


from src.libs import load

model = load()


# In[10]:


model.evaluate(wide_window.test, verbose=2)


# Try plot model

# In[11]:


model.reset_states()

wide_window.plot(model)


# In[12]:


import tensorflow as tf

test_window, label_window = next(iter(wide_window.test))
model.reset_states()
predictions = model(test_window)

predictions = tf.reshape(predictions, [-1])
label_window = tf.reshape(label_window, [-1])

pred2labels = pd.DataFrame({ 'Predicted': predictions, 'Labels': label_window})

pred2labels.iplot()


# In[10]:


import plotly.express as px

fig = px.scatter(x=pred2labels['Predicted'], y=pred2labels['Labels'])
fig.show()


# In[ ]:


OUT_STEPS=30
multi_window = WindowGenerator(
    input_width=30, label_width=OUT_STEPS, shift=OUT_STEPS,
    train_df=train_df, test_df=test_df, 
    label_columns=['Close'])

multi_window


# In[ ]:


import tensorflow as tf
from src.RepeatBaselineModel import RepeatBaseline

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

repeat_baseline.evaluate(multi_window.test, verbose=1)
multi_window.plot(repeat_baseline)


# ## Explore training metrics

# In[13]:


df = pd.read_csv('./metrics/training.csv')
df.head()


# In[14]:


df[['epoch', 'loss', 'val_loss']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='loss', 
    title='Training loss',
    linecolor='black',
)


# In[15]:


df[['epoch', 'mean_absolute_error', 'val_mean_absolute_error']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='mean_absolute_error', 
    title='mean_absolute_error'
)


# In[ ]:




