"""Live prediction script - clean and simple"""
from predictor import StockPredictor
from utils import load_models, calculate_portfolio_metrics
from config import STOCKS

def main():
    print("Live Indian Stock Predictions")
    print("=" * 35)
    
    # Load trained models
    models = load_models()
    if not models:
        print("Please train models first: python main.py")
        return
    
    # Create predictor
    predictor = StockPredictor(models)
    
    # Get predictions
    predictions = predictor.predict_portfolio(STOCKS)
    
    # Display predictions
    for company, result in predictions.items():
        if result['status'] == "Success":
            direction, confidence = predictor.format_prediction(
                result['prediction'], result['confidence']
            )
            print(f"{company:12}: {direction} {confidence}")
        else:
            print(f"{company:12}: --- {result['status']}")
    
    # Portfolio summary
    metrics = calculate_portfolio_metrics(predictions)
    if metrics:
        print(f"\n Portfolio Summary:")
        print(f"Bullish: {metrics['bullish_count']}/{metrics['total_predictions']}" f"({metrics['bullish_percentage']:.0f}%)")
        print(f"Avg confidence: {metrics['avg_confidence']:.1f}%")

if __name__ == "__main__":
    main()
