'''This file fetches the data in a clean way for Indian Stocks'''

import yfinance as yf
import pandas as pd

class DataFetcher:
    '''This class handles stock data collection from Yahoo Finance'''

    def __init__(self, period='2y'):
        self.period=period

    def get_stock_data(self, symbol):
        '''Fetch and clean stock data'''
        try:
            stock=yf.Ticker(symbol)
            df=stock.history(period=self.period)
            return self._clean_data(df)
        except Exception as e:
            print(f"Error Fetching {symbol}:{e}")
            return None
    
    def _clean_data(self, df):
        '''Function to remove invalid data points'''
        if df.empty:
            return df
        
        # Forward fill and remove NaN
        df=df.fillna(method='ffill').dropna()

        # Remove invalid price data
        mask=(
            (df['High'] >= df['Low']) &
            (df['Close'] > 0) &
            (df['Volume'] >=0)
        )
        return df[mask]
    
    def get_multiple_stocks(self, symbols):
        '''Fetching Data for multiple stocks'''
        data={}
        for symbol in symbols:
            df=self.get_stock_data(symbol)
            if df is not None and not df.empty:
                data[symbol]=df
                company=symbol.replace('.NS', '')
                print(f"{company}: {len(df)} records")
        return data