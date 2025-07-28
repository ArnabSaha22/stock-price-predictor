"""Main training script - simplified and modular"""
from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from utils import save_models, print_performance_summary
from config import STOCKS

def main():
    print("🇮🇳 Indian Stock Market Predictor")
    print("=" * 40)
    
    # Initialize components
    fetcher = DataFetcher()
    trainer = ModelTrainer()
    
    # Fetch all stock data
    print("📊 Fetching stock data...")
    stock_data = fetcher.get_multiple_stocks(STOCKS)
    
    if not stock_:
        print("No data fetched")
        return
    
    # Train models
    print("\n Training models...")
    results = {}
    
    for symbol, df in stock_data.items():
        company = symbol.replace('.NS', '')
        
        # Engineer features
        engineer = FeatureEngineer(df)
        processed_df = engineer.add_technical_indicators().add_custom_features().create_target().get_features()
        
        # Prepare training data
        feature_cols = engineer.get_feature_columns()
        X, y = processed_df[feature_cols].fillna(0), processed_df['Target']
        
        # Train model
        score = trainer.train_single_stock(X, y, symbol)
        results[company] = score
    
    # Display results
    print_performance_summary(results)
    
    # Save models
    save_models(trainer.models)
    
    # Summary stats
    summary = trainer.get_summary()
    print(f"\n Training complete!")
    print(f"Average accuracy: {summary['avg_score']:.4f}")
    print(f"Best performer: {summary['best_score']:.4f}")

if __name__ == "__main__":
    main()
