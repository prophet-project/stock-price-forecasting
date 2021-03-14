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


# # Check baseline metrics

# In[4]:


from src.prepare_datasets import make_window_generator, get_prepared_datasets

train_df, test_df = get_prepared_datasets()

window = make_window_generator()


# In[5]:


import tensorflow as tf
from src.BaselineModel import Baseline

column_indices = {name: i for i, name in enumerate(train_df.columns)}

baseline = Baseline(label_index=column_indices['Close'])
    
baseline.compile(
    loss=tf.losses.MeanSquaredError(),
    metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredLogarithmicError()]
)


# In[7]:


baseline.evaluate(window.test, verbose=1)


# # Test model predictions

# In[8]:


from src.libs import load

model = load()


# In[10]:


model.evaluate(window.test, verbose=2)
model.reset_states()


# Plot model

# In[11]:


model.reset_states()

window.plot(model)


# # Compare predictions and labels

# In[12]:


import tensorflow as tf

test_window, label_window = next(iter(window.test))
model.reset_states()
predictions = model(test_window)

predictions = tf.reshape(predictions, [-1])
label_window = tf.reshape(label_window, [-1])

pred2labels = pd.DataFrame({ 'Predicted': predictions, 'Labels': label_window})

pred2labels.iplot()


# In[13]:


import plotly.express as px

fig = px.scatter(x=pred2labels['Predicted'], y=pred2labels['Labels'])
fig.show()


# In[ ]:




