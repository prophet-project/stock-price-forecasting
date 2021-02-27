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


# In[3]:


import tensorflow_datasets as tfds


# ## Let's explore datasets

# In[4]:


from src.datasets import download

train_data, test_data = download(display_train_progress=True)


# ### Humanize labels
# 
# Labels can be 0, 0.5, 1. From bad to good sentimen. 
# 
# Will map them to correct words for easier exploring

# In[5]:


label_categories = ['bad', 'neutral', 'good']

def humanize_label(x):
    return label_categories[int(x * 2)] 


# ### Training data distribution

# In[6]:


train_df = pd.DataFrame(tfds.as_numpy(train_data), columns=['text', 'type'])

train_df['type'] = train_df['type'].apply(humanize_label)

train_df.head()


# In[7]:


print('Training dataset records', len(train_df.index))

train_df['type'].iplot(
    kind='hist',
    yTitle='count',
    xTitle='Type',
    title='Training data distribution'
)


# ### Testing data distribution

# In[8]:


test_df = pd.DataFrame(tfds.as_numpy(test_data), columns=['text', 'type'])

test_df['type'] = test_df['type'].apply(humanize_label)

test_df[30:40]


# In[9]:


print('Testing dataset records', len(test_df.index))

neutralSeries = test_df.apply(lambda x: True if x['type'] == 'neutral' else False, axis=1)
print('Count of neutral rows', len(neutralSeries[neutralSeries == True].index))

test_df['type'].iplot(
    kind='hist',
    yTitle='count',
    xTitle='Type',
    title='Testing data distribution'
)


# ## Normalize text

# ### Preprocess text
# 
# - Need remove special symbols. 
# - Replace usernames and links with readable words.
# - Split hashtags for stay they meaning 

# In[4]:


from src.normalize.normalize_text import preprocess_text

preprocess_text('Best awards øøøfor http://something.io/slpha ~and_ @futurer #thebest?')


# ### Replace misspels
# 
# Need replace misspells for decrease vocabularity size 
# and improve network results.

# In[2]:


from src.normalize.replace_misspells import replace_misspells

replace_misspells('Berst awwwards for link and username the best')


# ### Replace contractions
# 
# Contractions are words that we write with an apostrophe.
# Examples of contractions are words like “ain’t” or “aren’t”.
# 
# For standartize text better replace them

# In[1]:


from src.normalize.replace_contractions import replace_contractions

replace_contractions([
    "I'm a text with contraction, which can't be' easilly 'parsed' by NN",
    "This's unexpected for pycontractions, possible can be fixed by changenging word corpus"
])


# ### Lemmatize words
# 
# Will replace words with root form for decrease vocabulirity size

# In[3]:


from src.normalize.lemmatization import lematize

[lematize(word) for word in ['changing', 'connected', 'us', 'back']]


# ### Remove stopwords
# 
# Stopwords not give actual meaning but create noize in processing

# In[9]:


from src.normalize.remove_stopwords import is_stopword

[word + ' - ' + str(is_stopword(word)) for word in ['is', 'word', 'be', 'a', 'super', 'still', 'up', 'this','too', 'much', 'nothing', 'where', 'everyone', 'very', 'down', 'last', 'ok', 'good', 'it', 'back', 'empty', 'anyone', 'so', 'why', 'my', 'already', 'us']]


# ### Replace numbers 
# 
# Will replace numbers with `#`, it allow remove all possible numbers from text, but have they meaning

# In[1]:


from src.normalize.clean_text import replace_numbers

replace_numbers('I have $1 billion, but they only in my imagination. 1 billiion > 500 thouthands')


# ### Remove continiuse dublications
# 
# In case when author add dublicated words or punctuation for increase expression

# In[2]:


from src.normalize.clean_text import remove_continiues_dublications

remove_continiues_dublications("very very cool ! ! !".split())


# ### Result normalization

# In[2]:


from src.normalize.normalize_text import normalize_text

normalize_text("Barrichello to win the #f1 today???. I really want Kubica to place, he's a fantastic driver. Damn, why can't I watch it. In the US is back")


# ## Explore prepared dataset

# In[4]:


from src.prepare_datasets import load_preprocessed_datasets

train_prep_dataset, test_prep_dataset = load_preprocessed_datasets(display_train_progress=True)


# ### Check preprocessed training datasets distribution

# In[7]:


train_prep_df = pd.DataFrame(tfds.as_numpy(train_prep_dataset), columns=['text', 'type'])

train_prep_df['type'] = train_prep_df['type'].apply(humanize_label)

train_prep_df.head()


# In[8]:


print('Training dataset records', len(train_prep_df.index))

train_prep_df['type'].iplot(
    kind='hist',
    yTitle='count',
    xTitle='Type',
    title='Preprocessed training data distribution'
)


# ### Check testing dataset
# 

# In[9]:


test_prep_df = pd.DataFrame(tfds.as_numpy(test_prep_dataset), columns=['text', 'type'])

test_prep_df['type'] = test_prep_df['type'].apply(humanize_label)

test_prep_df.head()


# In[10]:


print('Training dataset records', len(test_prep_df.index))

test_prep_df['type'].iplot(
    kind='hist',
    yTitle='count',
    xTitle='Type',
    title='Preprocessed testing data distribution'
)


# ### Explore vocabulary

# In[5]:


from src.normalize import load_encoder

encoder, vocab_size = load_encoder()
sorted_vocab = sorted(encoder.tokens)

def show_vocab_tokens(vocab):
    for i in range(len(vocab)):
        word = vocab[i]
        character_numbers = ','.join([str(ord(character)) for character in word])
        print(word, '|', character_numbers, '|', len(word), '\n')

# print(' | '.join(sorted_vocab[90:100]))
show_vocab_tokens(sorted_vocab[400:600])


# ## Explore training metrics

# In[2]:


from validators.url import url as validate_url

validate_url('glamourkills.com')


# In[5]:


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


# In[7]:


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

# In[8]:


from src.predict import get_probability_model

model = get_probability_model()


# **Firstly will try predict on some data from training dataset**

# In[9]:


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

# In[10]:


# Can change text and check model
hadwriten = 'This is good film'

print('Hendwriten text:\n', hadwriten, '\n')

handwriten_label, predictions = predict(hadwriten, model)

print(label_categories[predicted_label], 'review')

print('Probabilities', predictions)


# In[ ]:




