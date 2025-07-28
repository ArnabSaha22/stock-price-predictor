'''
This file centralizes all project settings in one place for easy modification.
'''

# List of some Indian stock symbols for analysis and .NS suffix indicates these are NSE (National Stock Exchange) listed stocks

STOCKS=[
    'RELIANCE.NS',
    'TCS.NS',
    'INFY.NS',
    'HDFCBANK.NS',
    'ICICIBANK.NS'
]

PERIOD = '2y'  #Time period for historical data collection
PRED_DAYS = 1  #Time period for historical data collection

import os
os.makedirs('data', exist_ok=True)      # This will store downloaded stock data and intermediate files
os.makedirs('models', exist_ok=True)    # This will store trained machine learning models
