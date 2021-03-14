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


# # Let's explore datasets
# 
# ## input dataset
# 
# Will use target dataset [Bitcoin in Cryptocurrency Historical Prices](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory?select=coin_Bitcoin.csv)
# 
# Bitcoin data at 1-day intervals from April 28, 2013

# In[2]:


from src.load_datasets import load_input_dataset

input_dataset = load_input_dataset()

input_dataset.head()


# ### Target Features
# 
# Will analyze only interesting features

# In[3]:


feature_columns = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']
target_features = input_dataset[feature_columns]

target_features.head()


# In[4]:


import sweetviz as sv

analyse_report = sv.analyze([target_features, 'Bitcoin'], target_feat="Close")
analyse_report.show_notebook()


# ### Feature evalution over time

# In[5]:


datetime = pd.to_datetime(input_dataset['Date'])
target_features.index = datetime

target_features.iplot(
    subplots=True,
)


# In[6]:


target_features.describe().transpose()


# ### Slice dataset
# 
# Only last 4 years have active trading, will use them for explaration and training

# In[7]:


year = 365

years_count = 4
items_count = round(years_count * year)

last_years_dataset = input_dataset[-1 * items_count:]
last_years_datetime = pd.to_datetime(last_years_dataset['Date'])

last_years_dataset.head()
len(last_years_dataset)


# In[8]:


last_years_features = last_years_dataset[feature_columns]
last_years_features.index = last_years_datetime

last_years_features.iplot(
    subplots=True,
)


# In[ ]:




