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


# # Window Generator
# 
# For split input dataset on qual batches need split them on windows.
# Will use WindowGenerator class for spliting data into batches
# 
# ## Explore datasets

# In[2]:


from src.prepare_datasets import get_prepared_datasets

train_df, test_df = get_prepared_datasets()

train_df.head()

train_df.iplot(subplots=True)


# ## Calculate batch size

# In[3]:


from src.window_generator import WindowGenerator

w1 = WindowGenerator(
    input_width=24, label_width=1, shift=24, 
    train_df=train_df, test_df=test_df, 
    label_columns=['Close']
)

w1


# In[4]:


w1.plot(plot_col='Close')


# In[5]:


w1.train.element_spec


# ## Try baseline model

# In[6]:


single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    train_df=train_df, test_df=test_df, 
    label_columns=['Close'])

single_step_window


# In[7]:


import tensorflow as tf
from src.BaselineModel import Baseline

column_indices = {name: i for i, name in enumerate(train_df.columns)}

baseline = Baseline(label_index=column_indices['Close'])
    
baseline.compile(
    loss=tf.losses.MeanSquaredError(),
    metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredLogarithmicError()]
)


# In[8]:


baseline.evaluate(single_step_window.test, verbose=1)


# In[9]:


wide_window = WindowGenerator(
    input_width=32, label_width=32, shift=1,
    train_df=train_df, test_df=test_df,
    label_columns=['Close'])

wide_window


# In[10]:


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)


# In[11]:


wide_window.plot(baseline)


# In[ ]:




