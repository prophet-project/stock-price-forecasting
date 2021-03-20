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


# # Backtesting
# 
# For understand quality of predictions need make backtesting of bot on testing dataset
# 
# **TODO: Create strategy which can trade best way based on real future data.
#     Then add model which can predict more accurate future values**

# In[2]:


from src.load_datasets import load_datasets

train, test = load_datasets()

test


# In[3]:


test.index = pd.to_datetime(test.pop('timestamp'), unit='ms')

test


# In[4]:


test_hours = test[17::60]

test_hours


# In[5]:


test_hours.iplot(subplots=True)


# In[6]:


initial_cash = 10000 # dollars


# ## Calculate potentialy profit
# 
# Basically we have increasing market, so even with bad strategy trader can generate profit
# 
# To understand is strategy good or bad, need calculate basic profit from just holding value

# In[7]:


open_price = test['open'][0]
close_price = test['close'][-1]

print('Open %.2f and %.2f close price' % (open_price, close_price))

potential_profit = round((initial_cash / open_price) * close_price) - initial_cash

print('Potential profit after holding value:', potential_profit)


# In[8]:


import backtrader as bt
import backtrader.feeds as btfeeds

# Pass it to the backtrader datafeed and add it to the cerebro
data = bt.feeds.PandasData(dataname=test)

data


# ## Setup strategy

# In[9]:


class TestStrategy(bt.Strategy):
    
    params = (
        ('period_me1', 12),
        ('period_me2', 26),
        ('period_signal', 9),
        ('stohastic_period', 14),
        ('atr_period', 14)
    )

    
    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
        
    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        self.pbar = tqdm(total=len(test))
        
        # indicators
        self.macdHisto = bt.indicators.MACDHistogram(
            self.datas[0], 
            period_me1=self.params.period_me1,
            period_me2=self.params.period_me2,
            period_signal=self.params.period_signal
        )
        # get macd histogram and pass to model as MACD
        
        self.stohastic = bt.indicators.StochasticFast(
            self.datas[0], 
            period=self.params.stohastic_period
        )
        # get stohastic percK and pass as Stochastics Oscillator
        
        self.atr = bt.indicators.AverageTrueRange(
            self.datas[0],
            period=self.params.atr_period
        )
        # get atr and pass as ATR
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm
                    ))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm
                         ))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Canceled:
                self.log('Order is Canceled (by the user)')
            elif order.status == order.Margin:
                self.log('Order is Margin (not enough cash to execute the order)')
            else:
                self.log('Order is Rejected (by the broker)')

        self.order = None
        
    def notify_store(self, msg, *args, **kwargs):
        self.log('Notification from store: %s' % msg)
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
    
    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close %.2f, MACD %.2f, SO %.2f, ATR %.2f'  % (
            self.dataclose[0], self.macdHisto[0], self.stohastic[0], self.atr[0]
        ))
        self.pbar.update()

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] < self.dataclose[-1]:
                    # current close less than previous close

                    if self.dataclose[-1] < self.dataclose[-2]:
                        # previous close less than the previous close

                        # BUY, BUY, BUY!!! (with default parameters)
                        self.log('BUY CREATE, %.2f' % self.dataclose[0])

                        # Keep track of the created order to avoid a 2nd order
                        self.order = self.buy()
                        self.log('Created order %.2f size, and expected value %.2f' % (
                            self.order.created.size, self.order.created.size * self.dataclose[0]
                        ))

        else:

            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + 5):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
                self.log('Created order %.2f size, and expected value %.2f' % (
                    self.order.created.size, self.order.created.size * self.dataclose[0]
                ))


# ## Setup testing enviroment

# In[10]:


# Create a cerebro entity
cerebro = bt.Cerebro(stdstats=True)

# Add a strategy
cerebro.addstrategy(TestStrategy)

cerebro.adddata(data)

cerebro.broker.setcash(initial_cash)


# ### Size of each order
# 
# Each order will use percent of available cash
# in case if price will increase before order executed need use part of available cache
# and also take in mind possible commission

# In[11]:



cerebro.addsizer(bt.sizers.PercentSizer, percents=20) 


# ### Set commission

# In[12]:


# 0.1% ... divide by 100 to remove the %
cerebro.broker.setcommission(commission=0.001)


# ## Run backtesting

# In[13]:


from tqdm.auto import tqdm

initial_value = cerebro.broker.getvalue()

print('Starting strategy initiation...')
cerebro_results = cerebro.run()


# In[14]:


print('Starting Portfolio Value: %.2f' % initial_value)

result_value = cerebro.broker.getvalue()
print('Final Portfolio Value: %.2f' % result_value)

result_profit = result_value - initial_value
percent_of_potential = (result_profit / potential_profit) * 100

print('Result profit %.2f from potential profit %.2f' % (result_profit, potential_profit))
print('Realised of', percent_of_potential, '% of profit')


# In[15]:


cerebro.plot()

