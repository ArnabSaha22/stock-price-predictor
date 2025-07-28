"""Model training for Indian stocks"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class ModelTrainer:
    """Handles ML model training and validation"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def create_pipeline(self, model_type='rf'):
        """Create ML pipeline with scaling"""
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'lr':
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
    
    def train_single_stock(self, X, y, symbol, model_types=['rf', 'lr']):
        """Train models for one stock"""
        company = symbol.replace('.NS', '')
        tscv = TimeSeriesSplit(n_splits=5)  # FIXED - was n_split, now n_splits
        
        best_model = None
        best_score = 0
        best_type = None
        
        for model_type in model_types:
            pipeline = self.create_pipeline(model_type)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
            avg_score = cv_scores.mean()
            
            # Train on full data
            pipeline.fit(X, y)
            
            # Store if best
            if avg_score > best_score:
                best_score = avg_score
                best_model = pipeline
                best_type = model_type
        
        # Store results
        self.models[company] = {
            'model': best_model,
            'score': best_score,
            'type': best_type,
            'symbol': symbol
        }
        
        return best_score
    
    def get_model_info(self, company):
        """Get trained model info"""
        return self.models.get(company, None)
    
    def get_summary(self):
        """Get training summary"""
        if not self.models:
            return {}
        
        scores = [info['score'] for info in self.models.values()]
        return {
            'count': len(self.models),
            'avg_score': np.mean(scores),
            'best_score': max(scores),
            'worst_score': min(scores)
        }
