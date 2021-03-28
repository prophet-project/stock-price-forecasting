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


# In[2]:


from src.prepare_datasets import make_window_generator, get_prepared_datasets

train_df, test_df = get_prepared_datasets()

train, test = make_window_generator()


# # Test model predictions

# In[3]:


from src.libs import checkpoints
from src.model import build_model

model = build_model()
model = checkpoints.load_weights(model)


# # Compare predictions and labels

# In[4]:


import tensorflow as tf

input_window, label_window = next(iter(test))

predictions = model.predict(test, verbose=1, use_multiprocessing=True)


# In[6]:



input_window.shape
predictions.shape


# In[12]:


one_window = input_window[:1]
one_window.shape

one_window

predictions = model.predict_on_batch(one_window)
predictions.shape


# In[ ]:


test2predictions = pd.DataFrame({ 
    'Test': test_df['close'][:len(predictions)], 
    'Predicted': [ x[0] for x in predictions]
})
test2predictions.index = test_df[:len(predictions)].index

test2predictions.iplot()


# In[ ]:


import matplotlib.pyplot as plt

target_column='close'

def plot_window(batches, target, predictions=None):
    plt.figure(figsize=(15,len(batches) * 40))
    
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


plot_window(input_window[:8], label_window[:8], predictions[:8])


# In[ ]:


import plotly.express as px

fig = px.scatter(x=test2predictions['Predicted'], y=test2predictions['Test'])
fig.show()


# In[ ]:




