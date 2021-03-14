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


# In[3]:


train_df[59::60].iplot(subplots=True)


# In[4]:


train_df.info()


# In[5]:


target_column = 'close'


# ## Calculate batch size

# In[6]:


from src.window_generator import WindowGenerator

w1 = WindowGenerator(
    input_width=24, label_width=1, shift=24, 
    train_df=train_df, test_df=test_df, 
    label_columns=[target_column]
)

w1


# In[7]:


w1.plot(plot_col=target_column)


# In[8]:


w1.train.element_spec


# ## Try baseline model

# In[9]:


single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    train_df=train_df, test_df=test_df, 
    label_columns=[target_column])

single_step_window


# In[10]:


import tensorflow as tf
from src.BaselineModel import Baseline

column_indices = {name: i for i, name in enumerate(train_df.columns)}

baseline = Baseline(label_index=column_indices[target_column])
    
baseline.compile(
    loss=tf.losses.MeanSquaredError(),
    metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredLogarithmicError()]
)


# In[11]:


baseline.evaluate(single_step_window.test, verbose=1)


# In[12]:


wide_window = WindowGenerator(
    input_width=32, label_width=32, shift=1,
    train_df=train_df, test_df=test_df,
    label_columns=[target_column])

wide_window


# In[13]:


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)


# In[14]:


wide_window.plot(baseline)


# ## Calculate train/test window size

# In[31]:


len(train_df)

batch_size = 8
full_window_width = 33
train_delimetor = len(train_df) // (full_window_width * batch_size)
train_delimetor


# In[35]:


len(test_df)

test_delimetor = len(test_df) // (full_window_width * batch_size)
test_delimetor


# In[ ]:




