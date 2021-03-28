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


# # Prediction
# 
# Will try predict and denormalise model prediction

# In[2]:


from src.load_datasets import load_datasets
from src.prepare_datasets import add_indicators

train, test = load_datasets()

test.index = pd.to_datetime(test.pop('timestamp'), unit='ms')

test
test[::60].iplot(subplots=True)


# In[5]:


test = add_indicators(test)
test = test.dropna()

test.head()
test.iplot(subplots=True)


# In[3]:


from tqdm import tqdm
from src.prepare_datasets import get_prepared_datasets

train_norm, test_norm = get_prepared_datasets()

train_norm[::60].iplot(subplots=True)


# In[4]:


test_norm = test_norm[test.columns.tolist()]

test_norm[::60].iplot(subplots=True)


# Check denormalisation working correctly

# In[4]:


from src.prepare_datasets import get_scaler

scaler = get_scaler()


# In[ ]:


test_denorm = pd.DataFrame(scaler.inverse_transform(test_norm))
test_denorm.columns = test_norm.columns
test_denorm.index = test.index

test_denorm[::60].iplot(subplots=True)


# In[6]:


test.head()
test_norm.head()
test_denorm.head()


# In[5]:


from src.libs import params, prepare, save_metrics, load, checkpoints
from src.model import build_model
from src.prepare_datasets import make_generator

model = build_model()
model = checkpoints.load_weights(model)

len(test_norm)

for_pred = test_norm[:5000]

ds = make_generator(test_norm, test_norm[['close']], shuffle=False)

predictions = model.predict(ds)

len(predictions)
predictions.shape


# In[ ]:


input_w, label_w = next(iter(ds))

len(input_w)
input_w.shape


# In[11]:


# load if memory not enough
predictions = pd.read_csv('./predictions.csv')
predictions.index = test[-len(predictions):].index

predictions = predictions[['0']]
predictions


# In[13]:


predictions[::60].iplot()


# In[26]:


len(predictions)

predictions.head()


# In[28]:


predictions['0']


# In[29]:


for_inversion = test_norm[-len(predictions):]
for_inversion.index = predictions.index
for_inversion['close'] = predictions['0']

for_inversion.head()

for_inversion[::60].iplot(subplots=True)


# In[32]:


denorm_pred = pd.DataFrame(scaler.inverse_transform(for_inversion))

denorm_pred.columns = for_inversion.columns
denorm_pred.index = for_inversion.index

len(denorm_pred)

denorm_pred.head()


# In[35]:


predicted2norm = pd.DataFrame({
    'predicted': denorm_pred['close'],
    'real': test['close'][-len(denorm_pred):]
})

predicted2norm.head()

predicted2norm.index = test[-len(denorm_pred):].index

predicted2norm[::60].iplot()


# In[ ]:




