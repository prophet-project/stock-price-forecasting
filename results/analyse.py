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
# Will use target dataset [Bitcoin in Cryptocurrency Historical Prices](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory?select=coin_Bitcoin.csv)
# 
# Bitcoin data at 1-day intervals from April 28, 2013

# In[3]:


from src.load_datasets import load_input_dataset

input_dataset = load_input_dataset()

input_dataset.head()


# Will explore full input dataset

# In[4]:


import sweetviz as sv

target_features = input_dataset[['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']]

analyse_report = sv.analyze([target_features, 'Bitcoin'], target_feat="Close")
analyse_report.show_notebook()


# In[5]:


target_features.head()


# Feature evalution over time

# In[6]:


datetime = pd.to_datetime(input_dataset['Date'])
target_features.index = datetime

target_features.iplot(
    subplots=True,
)


# In[7]:


target_features.describe().transpose()


# Will take only last 4 years, because they mostly interesting

# In[8]:


year = 365

years_count = 4
items_count = round(years_count * year)

last_years_dataset = input_dataset[-1 * items_count:]
last_years_datetime = pd.to_datetime(last_years_dataset['Date'])

last_years_dataset.head()
len(last_years_dataset)


# In[9]:


last_years_features = last_years_dataset[['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']]
last_years_features.index = last_years_datetime

last_years_features.iplot(
    subplots=True,
)


# ### Check depenence of trading and price from date in year and time of day

# Firstly define function for display frequiency

# In[10]:


import tensorflow as tf
import matplotlib.pyplot as plt

def plot_log_freaquency(series):
    fft = tf.signal.rfft(series)    
    f_per_dataset = np.arange(0, len(fft))

    n_samples_d = len(series)
    days_per_year = 365
    years_per_dataset = n_samples_d/(days_per_year)

    f_per_year = f_per_dataset/years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.xticks([1, 365], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')


# Frequency of price

# In[18]:


plot_log_freaquency(last_years_dataset['Close'])


# Frequency of transaction volume

# In[19]:


plot_log_freaquency(last_years_dataset['Volume'])


# ## Compare train and test datasets

# In[11]:


from src.load_datasets import load_datasets

train_df, test_df = load_datasets()

train_df


# In[12]:


feature_list = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']

train_features = train_df[feature_list]
test_features = test_df[feature_list]

compare_report = sv.compare([train_features, 'Train data'], [test_features, 'Test data'], "Close")
compare_report.show_notebook()


# In[13]:


train_datetime = pd.to_datetime(train_df['Date'])
test_datetime = pd.to_datetime(test_df['Date'])

train_features.index = train_datetime
test_features.index = test_datetime


# ### Training data exploration

# In[14]:


train_features.iplot(subplots=True)


# ### Testing data exploration

# In[15]:


test_df


# In[16]:


test_features.iplot(subplots=True)


# ## Normalise data
# 
# Will use only training mean and deviation for not give NN access to test dataset
# 
# Subtract the mean and divide by the standard deviation of each feature will give required normalisation

# In[17]:


train_mean = train_features.mean()
train_std = train_features.std()

train_features = (train_features - train_mean) / train_std
test_features = (test_features - train_mean) / train_std


# In[18]:


train_features


# In[19]:


train_features.iplot(subplots=True)


# In[20]:


test_features.iplot(subplots=True)


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

def show_normalised(df):
    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    # plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)


# In[21]:


show_normalised(train_features)


# In[24]:


show_normalised(test_features)


# ## Check window generator

# In[6]:


from src.prepare_datasets import get_prepared_datasets
from src.window_generator import WindowGenerator

train_df, test_df = get_prepared_datasets()
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


# In[13]:


import tensorflow as tf
from src.BaselineModel import Baseline

column_indices = {name: i for i, name in enumerate(train_df.columns)}

baseline = Baseline(label_index=column_indices['Close'])
    
baseline.compile(
    loss=tf.losses.MeanSquaredError(),
    metrics=[tf.metrics.MeanAbsoluteError()]
)


performance = {}
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=1)


# In[7]:


wide_window = WindowGenerator(
    input_width=30, label_width=30, shift=1,
    train_df=train_df, test_df=test_df,
    label_columns=['Close'])

wide_window


# In[17]:


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)


# In[18]:


wide_window.plot(baseline)


# In[3]:


from src.libs import load

model = load()


# Try plot model

# In[8]:




wide_window.plot(model)


# In[7]:


OUT_STEPS=30
multi_window = WindowGenerator(
    input_width=30, label_width=OUT_STEPS, shift=OUT_STEPS,
    train_df=train_df, test_df=test_df, 
    label_columns=['Close'])

multi_window


# In[16]:


import tensorflow as tf
from src.RepeatBaselineModel import RepeatBaseline

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

repeat_baseline.evaluate(multi_window.test, verbose=1)
multi_window.plot(repeat_baseline)


# In[8]:


from src.libs import load_custom
from src.model import build_model

model = build_model()

load_custom(model)

multi_window.plot(model)


# ## Explore training metrics

# In[3]:


df = pd.read_csv('./metrics/training.csv')
df.head()


# In[4]:


df[['epoch', 'loss', 'val_loss']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='loss', 
    title='Training loss',
    linecolor='black',
)


# In[5]:


df[['epoch', 'mean_absolute_error', 'val_mean_absolute_error']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='mean_absolute_error', 
    title='mean_absolute_error'
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




