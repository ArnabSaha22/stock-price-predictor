"""Technical indicators"""
import pandas as pd
import numpy as np

class FeatureEngineer:
    """Creates technical indicators"""
    
    def __init__(self, df):
        self.df = df.copy()
    
    def add_technical_indicators(self):
        """Add technical indicators"""
        # Simple moving averages
        self.df['SMA_10'] = self.df['Close'].rolling(10).mean()
        self.df['SMA_30'] = self.df['Close'].rolling(30).mean()
        self.df['EMA_12'] = self.df['Close'].ewm(span=12).mean()
        
        # RSI
        self.df['RSI'] = self._rsi(self.df['Close'])
        
        # MACD
        self.df['MACD'], self.df['MACD_Signal'] = self._macd(self.df['Close'])
        
        # Stochastic
        self.df['Stoch_K'] = self._stochastic(self.df)
        
        # Bollinger Bands
        sma20 = self.df['Close'].rolling(20).mean()
        std20 = self.df['Close'].rolling(20).std()
        self.df['BB_Upper'] = sma20 + (2 * std20)
        self.df['BB_Middle'] = sma20
        self.df['BB_Lower'] = sma20 - (2 * std20)
        
        # ATR
        self.df['ATR'] = self._atr(self.df)
        
        # OBV
        self.df['OBV'] = self._obv(self.df)
        self.df['Volume_MA'] = self.df['Volume'].rolling(20).mean()
        
        return self
    
    def _rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _macd(self, prices):
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _stochastic(self, df, window=14):
        low_min = df['Low'].rolling(window=window).min()
        high_max = df['High'].rolling(window=window).max()
        return ((df['Close'] - low_min) / (high_max - low_min)) * 100
    
    def _atr(self, df, window=14):
        hl = df['High'] - df['Low']
        hc = abs(df['High'] - df['Close'].shift())
        lc = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    def _obv(self, df):
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)
    
    def add_custom_features(self):
        """Add engineered features"""
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Volatility'] = self.df['Returns'].rolling(20).std()
        self.df['Price_Range'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        
        self.df['MA_Signal'] = (self.df['SMA_10'] > self.df['SMA_30']).astype(int)
        self.df['RSI_Signal'] = ((self.df['RSI'] > 30) & (self.df['RSI'] < 70)).astype(int)
        self.df['Volume_Spike'] = (self.df['Volume'] > self.df['Volume_MA'] * 1.5).astype(int)
        
        return self
    
    def create_target(self, horizon=1):
        future_price = self.df['Close'].shift(-horizon)
        self.df['Target'] = (future_price > self.df['Close']).astype(int)
        return self
    
    def get_features(self):
        return self.df.dropna()
    
    def get_feature_columns(self):
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']
        return [col for col in self.df.columns if col not in exclude]
