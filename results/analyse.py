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


# In[2]:



from plotly.offline import iplot, init_notebook_mode
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# Set global theme
cufflinks.set_config_file(world_readable=True, theme='pearl')


# ## Let's explore datasets

# ### Explore input dataset
# 
# Will use target dataset [Bitcoin Historical Data](https://www.kaggle.com/mczielinski/bitcoin-historical-data)
# 
# Bitcoin data at 1-min intervals from select exchanges, Jan 2012 to Dec 2020

# In[3]:


from src.load_datasets import load_input_dataset

input_dataset = load_input_dataset()

input_dataset.head()


# Will explore full input dataset, some values contain NaN, which not ineraptebale by sweetviz, so will use timestamp as target feature for now

# In[4]:


import sweetviz as sv

analyse_report = sv.analyze([input_dataset, 'Input'], target_feat="Timestamp")
analyse_report.show_notebook()


# Will take one timestamp per hour for faster interpretation

# In[5]:


hours_dataset = input_dataset[59::60]


# timestamp need interprate as date for charts processing

# In[6]:


raw_timestamps = hours_dataset.pop('Timestamp')
hours_datetime = pd.to_datetime(raw_timestamps, unit='s')


# In[7]:


hours_dataset.head()


# Feature evalution over time

# In[8]:


hours_features = hours_dataset[['Open', 'Close', 'Weighted_Price', 'Volume_(BTC)', 'Volume_(Currency)']]
hours_features.index = hours_datetime

hours_features.iplot(
    subplots=True,
)


# In[9]:


hours_dataset.describe().transpose()


# Will take only last three yers, because they have data without missing values

# In[10]:


day = 24
year = (365)*day

years_count = 3.5
items_count = round(years_count * year)

last_years_dataset = hours_dataset[-1 * items_count:]
last_years_datetime = hours_datetime[-1 * items_count:]

last_years_dataset.head()
len(last_years_dataset)


# In[11]:


last_years_features = last_years_dataset[['Open', 'Close', 'Weighted_Price', 'Volume_(BTC)', 'Volume_(Currency)']]
last_years_features.index = last_years_datetime

last_years_features.iplot(
    subplots=True,
)


# ## Remove NaN values

# In[31]:


# Is have NaN
last_years_dataset.isnull().values.any()


# In[32]:


last_years_dataset.isnull().sum().sum()


# In[50]:


last_years_dataset_with_time = input_dataset[59::60][-1 * items_count:]
len(last_years_dataset_with_time)
len(last_years_dataset)

nan_datafreame = last_years_dataset_with_time[last_years_dataset_with_time.isna().any(axis=1)]
nan_datafreame


# In[51]:


nan_hours_datetime = pd.to_datetime(nan_datafreame.pop('Timestamp'), unit='s')


# In[56]:


nan_features =  nan_datafreame['Weighted_Price']
nan_features.index = nan_hours_datetime
nan_features.iplot(
    kind='scatter',
    mode='markers'
)


# ### Check depenence of trading and price from date in year and time of day

# Firstly define function for display frequiency

# In[28]:


import tensorflow as tf
import matplotlib.pyplot as plt

def plot_log_freaquency(series):
    fft = tf.signal.rfft(series)    
    f_per_dataset = np.arange(0, len(fft))

    n_samples_h = len(series)
    hours_per_year = 24*365.2524
    years_per_dataset = n_samples_h/(hours_per_year)

    f_per_year = f_per_dataset/years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')


# Frequency of price

# In[29]:


without_nan = last_years_dataset.dropna()

plot_log_freaquency(without_nan['Weighted_Price'])


# Frequency of transaction volume

# In[30]:


plot_log_freaquency(without_nan['Volume_(Currency)'])


# ### Training data distribution

# In[ ]:


train_df = pd.DataFrame(tfds.as_numpy(train_data), columns=['text', 'type'])

train_df['type'] = train_df['type'].apply(humanize_label)

train_df.head()


# In[ ]:


print('Training dataset records', len(train_df.index))

train_df['type'].iplot(
    kind='hist',
    yTitle='count',
    xTitle='Type',
    title='Training data distribution'
)


# ### Testing data distribution

# In[ ]:


test_df = pd.DataFrame(tfds.as_numpy(test_data), columns=['text', 'type'])

test_df['type'] = test_df['type'].apply(humanize_label)

test_df[30:40]


# In[ ]:


print('Testing dataset records', len(test_df.index))

neutralSeries = test_df.apply(lambda x: True if x['type'] == 'neutral' else False, axis=1)
print('Count of neutral rows', len(neutralSeries[neutralSeries == True].index))

test_df['type'].iplot(
    kind='hist',
    yTitle='count',
    xTitle='Type',
    title='Testing data distribution'
)


# ### Check preprocessed training datasets distribution

# In[ ]:


train_prep_df = pd.DataFrame(tfds.as_numpy(train_prep_dataset), columns=['text', 'type'])

train_prep_df['type'] = train_prep_df['type'].apply(humanize_label)

train_prep_df.head()


# In[ ]:


print('Training dataset records', len(train_prep_df.index))

train_prep_df['type'].iplot(
    kind='hist',
    yTitle='count',
    xTitle='Type',
    title='Preprocessed training data distribution'
)


# ### Check testing dataset
# 

# In[ ]:


test_prep_df = pd.DataFrame(tfds.as_numpy(test_prep_dataset), columns=['text', 'type'])

test_prep_df['type'] = test_prep_df['type'].apply(humanize_label)

test_prep_df.head()


# In[ ]:


print('Training dataset records', len(test_prep_df.index))

test_prep_df['type'].iplot(
    kind='hist',
    yTitle='count',
    xTitle='Type',
    title='Preprocessed testing data distribution'
)


# ## Explore training metrics

# In[ ]:


df = pd.read_csv('./metrics/training.csv')
df.head()


# In[ ]:


df[['epoch', 'accuracy', 'val_accuracy']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='accuracy', 
    title='Training accuracy',
    linecolor='black',
)


# In[ ]:


df[['epoch', 'loss', 'val_loss']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='accuracy', 
    title='Losses'
)


# ## Predictions
# 
# ### Load probability model
# 
# which can give predictions on model classes
# 
# 0 - bad review, 1 - good revie

# In[ ]:


from src.predict import get_probability_model

model = get_probability_model()


# **Firstly will try predict on some data from training dataset**

# In[ ]:


from src.predict import get_text_and_label_from_dataset, predict
REVIEW_INDEX = 110

text, real_label = get_text_and_label_from_dataset(REVIEW_INDEX)

print('text for prediction\n\n', text, '\n')

predicted_label, predictions = predict(text, model)

print(label_categories[predicted_label], 'review')

print('\n\nPredicted label:', predicted_label, 'real label: ', real_label, 'predictions:', predictions)
if (predicted_label == real_label):
    print('Successfully predicted')
else:
    print('Failed to predict')


# **Then will try predict hadnwritten text**

# In[ ]:


# Can change text and check model
hadwriten = 'This is good film'

print('Hendwriten text:\n', hadwriten, '\n')

handwriten_label, predictions = predict(hadwriten, model)

print(label_categories[predicted_label], 'review')

print('Probabilities', predictions)


# In[ ]:




