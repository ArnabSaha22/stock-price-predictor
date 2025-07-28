'''Utility functions for the stock predictor'''

import joblib
import os

def save_models(models, filename='models/indian_models.joblib'):
    '''saving the trained models to local storage'''
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(models, filename)
    print(f"File saved {len(models)} models to {filename}")

def load_models(filename='models/indian_models.joblib'):
    '''Load models from the local storage'''
    try:
        models=joblib.load(filename)
        print(f"Loaded {len(models)} models")
        return models
    except FileNotFoundError:
        print("No saved mode is found")
        return {}
    
def print_performance_summary(results):
    '''Print formatted performance summary'''
    if not results:
        print("No results to display")
        return
    print("performance summary:")
    print("="*40)

    for company, score in results.items():
        if score > 0.60:
            status="Excellent"
        elif score > 0.55:
            status ="Good"
        elif score > 0.52:
            status = "Fair"
        else:
            status="Poor"

        print(f"{company:12}: {score:.4f} {status}")

def calculate_portfolio_metrics(predictions):
    '''Calculate portfolio-level metrics'''
    successful=[p for p in predictions.values() if p['prediction'] is not None]

    if not successful:
        return {}
    
    return {
        'total_predictions': total,
        'bullish_count': bullish,
        'bullish_percentage': (bullish / total) * 100,
        'high_confidence_count': high_conf,
        'avg_confidence': sum(p['confidence'] for p in successful if p['confidence']) / total
    }