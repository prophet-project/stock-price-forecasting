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


import tensorflow as tf

def make_generator(data, targets, shuffle, batch_size=8, sequence_length=33, sequence_stride=1):
    return tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data[:-sequence_length],
      targets=targets[sequence_length:],
      sequence_length=sequence_length,
      sequence_stride=sequence_stride,
      shuffle=shuffle,
      batch_size=batch_size,
  )

example_dataset = list(range(100))
print('dataset', example_dataset, '\n')

example_iterator = make_generator(example_dataset, example_dataset, shuffle=False)

input, target = next(iter(example_iterator))

print('Input', input, '\n')
print('Target', target, '\n')


# ## How baseline work

# In[7]:


result = input[:,:]
result

result[:,-1,tf.newaxis]


# ## Prepare real datasets

# In[8]:


train_iterator = make_generator(train_df, train_df[[target_column]], shuffle=True)
test_iterator = make_generator(test_df, test_df[[target_column]], shuffle=False)

input, target = next(iter(test_iterator))


# In[19]:


import matplotlib.pyplot as plt

def plot_window(batches, target, predictions=None):
    plt.figure(figsize=(15,len(batches) * 10))
    
    batches = batches.numpy()
    target = target.numpy()
    
    for i in range(0, len(batches)):
        
        batch = batches[i]
        feature = [x[train_df.columns.get_loc(target_column)] for x in batch]
        plt.subplot(len(feature), 1, i+1)
        plt.plot(feature, 
                 label='Inputs', marker='.', zorder=-10
                )
        
        label = target[i][0]
        plt.scatter(len(feature), label,
                 label='Labels', edgecolors='k', c='#2ca02c', s=64
                )
        
        if predictions is not None:
            prediction = predictions[i][0]
            plt.scatter(len(feature), prediction,
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)
        
        plt.legend()
        
plot_window(input, target)


# ## Try baseline model

# In[10]:


import tensorflow as tf
from src.BaselineModel import Baseline

column_indices = {name: i for i, name in enumerate(train_df.columns)}

baseline = Baseline(label_index=column_indices[target_column])
    
baseline.compile(
    loss=tf.losses.MeanSquaredError(),
    metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredLogarithmicError()]
)


# In[12]:


predictions = baseline.predict(test_iterator, verbose=1, use_multiprocessing=True)


# In[13]:


predictions.shape


# In[20]:


plot_window(input, target, predictions)


# In[ ]:




