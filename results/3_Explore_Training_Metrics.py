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


# # Explore training metrics
# 
# ## Get baseline metrics

# In[2]:


import tensorflow as tf
from src.BaselineModel import Baseline
from src.prepare_datasets import get_prepared_datasets, make_window_generator

train_df, test_df = get_prepared_datasets()

column_indices = {name: i for i, name in enumerate(train_df.columns)}

baseline = Baseline(label_index=column_indices['close'])
    
baseline.compile(
    loss=tf.losses.MeanSquaredError(),
    metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredLogarithmicError()]
)


# In[3]:


window = make_window_generator()


# In[4]:


baseline_test_metrics = baseline.evaluate(window.test, verbose=1)

baseline_test_metrics = pd.DataFrame(data=[baseline_test_metrics], columns=baseline.metrics_names)
baseline_test_metrics


# In[5]:


baseline_train_metrics = baseline.evaluate(window.train, verbose=1)

baseline_train_metrics = pd.DataFrame(data=[baseline_train_metrics], columns=baseline.metrics_names)
baseline_train_metrics


# In[6]:


df = pd.read_csv('./metrics/training.csv')

df['baseline_test_loss'] = baseline_test_metrics['loss'][0]
df['baseline_test_mean_absolute_error'] = baseline_test_metrics['mean_absolute_error'][0]
df['baseline_test_mean_squared_logarithmic_error'] = baseline_test_metrics['mean_squared_logarithmic_error'][0]

df['baseline_train_loss'] = baseline_train_metrics['loss'][0]
df['baseline_train_mean_absolute_error'] = baseline_train_metrics['mean_absolute_error'][0]
df['baseline_train_mean_squared_logarithmic_error'] = baseline_train_metrics['mean_squared_logarithmic_error'][0]

df.head()


# In[7]:


df[['epoch', 'loss', 'val_loss']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='loss', 
    title='Training loss',
    linecolor='black',
)


# In[8]:


df[['epoch', 'mean_absolute_error', 'val_mean_absolute_error']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='mean_absolute_error', 
    title='mean_absolute_error'
)


# In[9]:


df[['epoch', 'mean_squared_logarithmic_error', 'val_mean_squared_logarithmic_error']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='mean_absolute_error', 
    title='mean_absolute_error'
)


# In[10]:


df[['epoch', 'loss', 'baseline_train_loss']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='loss', 
    title='Training loss with baseline',
    linecolor='black',
)


# In[11]:


df[['epoch', 'val_loss', 'baseline_test_loss']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='loss', 
    title='Training validation loss with baseline',
    linecolor='black',
)


# In[ ]:




