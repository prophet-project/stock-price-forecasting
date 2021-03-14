import pandas as pd

def MACD(df,period1,period2,periodSignal):
    EMA1 = pd.DataFrame.ewm(df,span=period1).mean()
    EMA2 = pd.DataFrame.ewm(df,span=period2).mean()
    MACDframe = EMA1-EMA2
    
    Signal = pd.DataFrame.ewm(MACDframe,periodSignal).mean()
    
    Histogram = MACDframe-Signal
    
    return Histogram

def stochastics_oscillator(df,period):
    l, h = pd.DataFrame.rolling(df, period).min(), pd.DataFrame.rolling(df, period).max()
    k = 100 * (df - l) / (h - l)
    return k

'''
Method A: Current High less the current Low
'''
def ATR(df,period):
    df['H-L'] = abs(df['high']-df['low'])
    df['H-PC'] = abs(df['high']-df['close'].shift(1))
    df['L-PC'] = abs(df['low']-df['close'].shift(1))
    
    TR = df[['H-L','H-PC','L-PC']].max(axis=1)
    
    return TR.to_frame()