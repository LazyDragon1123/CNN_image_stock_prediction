import pandas as pd
import datetime
import matplotlib.pyplot as plt
import mplfinance as fplt
import os
import numpy as np 

class IMAGEN:

    def __init__(self,df,ticker,last_date,frames):
        '''
        last_date: 2021-05-21
        '''
        self.df = df
        self.days_interval = 5
        self.last_date = datetime.datetime.strptime(last_date,'%Y-%m-%d')
        self.frames = frames
        self.ticker = ticker
        os.makedirs('./images/' + self.ticker, exist_ok=True)

    def generator_w_target(self):
        df = self.df
        name_index = df.index.to_list()
        ind = [datetime.datetime.strptime(i,'%Y-%m-%d') for i in df.index.to_list()]
        df.index = ind
        df['Open'] = df['Open'] * (df['Adj Close'] / df['Close']) 
        df['High'] = df['High'] * (df['Adj Close'] / df['Close']) 
        df['Low'] = df['Low'] * (df['Adj Close'] / df['Close']) 
        df['Close'] = df['Adj Close'] 
        targets = np.ones(self.frames)
        target_index = []
        for index in range(self.frames): # descending order
            fig_name = name_index[-index - 1]
            target_index.append(fig_name)
            ans_ind = ind[-index]
            if index == 0:
                fplt.plot(
                    df.iloc[-index-self.days_interval:,:],
                    type='candle',
                    style='charles',
                    volume=True,
                    axisoff=True,
                    savefig=dict(fname='./images/' + self.ticker + '/' + fig_name + '.png',dpi=100,pad_inches=0.1,bbox_inches='tight')
                    )
            else:
                fplt.plot(
                    df.iloc[-index-self.days_interval:-index,:],
                    type='candle',
                    style='charles',
                    volume=True,
                    axisoff=True,
                    savefig=dict(fname='./images/' + self.ticker + '/' + fig_name + '.png',dpi=100,pad_inches=0.1,bbox_inches='tight')
                    )
            if df.loc[ans_ind,'Open'] <= df.loc[ans_ind,'Close']:
                targets[index] = 1
            else:
                targets[index] = 0
        target_df = pd.DataFrame()
        target_df['target'] = targets
        target_df.index = target_index
        target_df.to_csv('./images/' + self.ticker + '/' + 'target.csv')
        

