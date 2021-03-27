#!/usr/bin/env python
# coding: utf-8

# In[21]:


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


# # Prediction
# 
# Will try predict and denormalise model prediction

# In[22]:


from src.load_datasets import load_datasets
from src.prepare_datasets import add_indicators

train, test = load_datasets()

test.index = pd.to_datetime(test.pop('timestamp'), unit='ms')

test = add_indicators(test)
test = test.dropna()

test
test[::60].iplot(subplots=True)


# In[23]:


from tqdm import tqdm
from src.prepare_datasets import normalize_row

tqdm.pandas(desc="test dataset")
test_norm = test.progress_apply(normalize_row, axis=1)

test_norm[::60].iplot(subplots=True)


# In[46]:


test_norm = test_norm[test.columns.tolist()]

test_norm[::60].iplot(subplots=True)


# Check denormalisation working correctly

# In[10]:


from src.prepare_datasets import denormalise_row

tqdm.pandas(desc="test norm")
test_denorm = test_norm.progress_apply(denormalise_row, axis=1)

test_denorm[::60].iplot(subplots=True)


# In[47]:


test.head()
test_norm.head()
test_denorm.head()


# In[54]:


from src.libs import load
import tensorflow as tf

model = load()

ds = tf.keras.preprocessing.timeseries_dataset_from_array(
    test_norm, 
    targets=None, 
    sequence_length=32,
    sequence_stride=32,
    shuffle=False,
    batch_size=8
)
input = next(iter(ds))

len(input)

predictions = model.predict(input)

len(predictions)


# In[49]:


predictions.shape
output = pd.Series(tf.reshape(predictions, [-1]).numpy())
output.index = test_norm[:256].index

output.iplot(subplots=True)


# In[50]:


predicted2norm = pd.DataFrame({
    'predicted': output,
    'real': test_norm[:256]['close']
})

predicted2norm.index = test_norm[:256].index

predicted2norm.iplot(subplots=True)


# In[51]:


from src.prepare_datasets import norm_d

norm_d

predicted_denorm = output.apply(lambda x: x * norm_d['close'] )

predicted_denorm.iplot()


# In[52]:


predicted2real = pd.DataFrame({
    'predicted': predicted_denorm,
    'real': test[:256]['close']
})

predicted2real.index = test[:256].index

predicted2real.iplot(subplots=True)


# In[ ]:




