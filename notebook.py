from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from predictor import StockPredictor
from utils import *
from config import STOCKS
import matplotlib.pyplot as plt
from utils import print_performance_summary


print("================ Indian Stock Analysis ================")

fetcher = DataFetcher()
trainer = ModelTrainer()
stock_data = fetcher.get_multiple_stocks(STOCKS)
results = {}

for symbol, df in stock_data.items():
    company = symbol.replace('.NS', '')
    engineer = FeatureEngineer(df)
    processed_df = engineer.add_technical_indicators().add_custom_features().create_target().get_features()
    
    feature_cols = engineer.get_feature_columns()
    X, y = processed_df[feature_cols].fillna(0), processed_df['Target']
    
    score = trainer.train_single_stock(X, y, symbol)
    results[company] = score


companies = list(results.keys())
scores = list(results.values())

plt.figure(figsize=(12, 6))
colors = ['green' if s > 0.55 else 'orange' if s > 0.52 else 'red' for s in scores]
plt.bar(companies, scores, color=colors, alpha=0.7)
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
plt.title('🇮🇳 Indian Stock Prediction Accuracy')
plt.ylabel('Cross-Validation Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print_performance_summary(results)


predictor = StockPredictor(trainer.models)
predictions = predictor.predict_portfolio(STOCKS)

for company, result in predictions.items():
    if result['prediction'] is not None:
        direction, conf = predictor.format_prediction(result['prediction'], result['confidence'])
        print(f"{company}: {direction} {conf}")

save_models(trainer.models)
print("\n✅ Complete!")