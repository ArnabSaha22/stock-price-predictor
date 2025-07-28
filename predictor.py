'''Prediction engine for Indian stocks'''
from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer

class StockPredictor:
    '''Handles stock price prediction'''

    def __init__(self, trained_models):
        self.models=trained_models
        self.data_fetcher= DataFetcher(period='3mo') # 3 months for fresh prediction

    def predict_stock(self, symbol):
        '''Predict single stock movement'''
        company=symbol.replace('.NS', '')

        #checking if the model exists
        if company not in self.models:
            return None, None, "Model not found"
        
        try:
            df=self.data_fetcher.get_stock_data(symbol)
            if df is None or df.empty:
                return None, None, "No data is available"
            
            # Engineer features
            engineer=FeatureEngineer(df)
            processed_df=engineer.add_technical_indicators().add_custom_features().get_features()

            # Get model and features
            model_info=self.models[company]
            model=model_info['model']

            # Get feature columns (same as training)
            feature_cols=engineer.get_feature_columns()
            latest_features=processed_df[feature_cols].iloc[-1:].fillna(0)

            # Make prediction
            prediction=model.predict(latest_features)[0]
            probabilities=model.predict_proba(latest_features)[0]
            confidence=max(probabilities)*100

            return prediction, confidence, "Success"
        
        except Exception as e:
            return None, None, f"Error {str(e)}"
        
    def predict_portfolio(self, symbols):
        '''Predict multiple stocks'''
        results={}

        for symbol in symbols:
            company=symbol.replace('.NS', '')
            pred, conf, status=self.predict_stock(symbol)

            results[company]={
                'prediction': pred,
                'confidence': conf,
                'status': status,
                'symbol': symbol
            }
        return results
    
    def format_prediction(self, prediction, confidence):
        '''Format prediction for display'''
        if prediction is None:
            return "Error", "N/A"
        
        direction="UP" if prediction==1 else "DOWN"

        if confidence > 75:
            conf_icon="Great"
        elif confidence > 65:
            conf_icon="Okayish"
        else:
            conf_icon="Bad"

        return direction, f"{conf_icon} {confidence:.1f}%"