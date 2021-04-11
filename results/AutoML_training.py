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


import os
import mlflow

trackng_url = os.getenv('MLFLOW_TRACKING_URI')
print(trackng_url)
mlflow.set_tracking_uri(trackng_url)


# # Prepare dataset

# In[3]:


from src.prepare_datasets import get_prepared_datasets

# five minute dataset
train, test = get_prepared_datasets()

train


# In[4]:


train.index = pd.to_datetime(train.pop('timestamp'))
test.index = pd.to_datetime(test.pop('timestamp'))

train[::15].iplot(subplots=True)


# In[5]:


test[::15].iplot(subplots=True)


# In[6]:


len(test)
prediction_size = 3*31*24*15 # last three monthes

# for later prediction test
unseen_data = test[-prediction_size:]
test_only = test[0:-prediction_size]

len(test_only)

unseen_data


# # Start training

# In[7]:


from pycaret.regression import *

exp_reg_btc = setup(
    data = train, test_data = test_only, target = 'close', 
    session_id = 123, 
    log_experiment = True, log_plots=True, log_profile=True, experiment_name = 'bitcoin_close_5m_norm',
    use_gpu = True,
    data_split_shuffle = False,
    feature_selection = True, feature_selection_threshold = 0.6
)


# ### TODO
# 
# * try use all features from add_ta_features and set feature_selection=True
# * try add featues by tsfreash
# * try extract features by tsfreash
# * try remove trend by sktime

# In[8]:


best = compare_models(n_select=3)


# In[9]:


best


# In[10]:


models()


# In[11]:


plot_model(best[0], plot = 'learning')


# In[12]:


plot_model(best[1], plot = 'learning')


# In[14]:


plot_model(best[0], plot = 'error')


# In[15]:


plot_model(best[0])


# In[16]:


lgbmr = tune_model(best[0], choose_better = True)


# In[17]:


lgbmr


# In[18]:


plot_model(lgbmr, plot='feature')


# In[19]:


evaluate_model(lgbmr)


# In[20]:


predict_model(lgbmr)


# In[21]:


final_lgbmr = finalize_model(lgbmr)


# In[22]:


save_model(final_lgbmr, './saved_models/Final_LGBMR_5m_bitcoin_norm')


# In[23]:


unseen_predictions = predict_model(final_lgbmr, data=unseen_data)
unseen_predictions.head()


# In[24]:


from pycaret.utils import check_metric
check_metric(unseen_predictions['close'], unseen_predictions['Label'], 'R2')


# In[25]:


unseen_predictions[['close', 'Label']].iplot()


# In[ ]:




