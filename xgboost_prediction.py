import numpy as np
import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

class BigMartSalesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.columns_to_scale = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years']
        
    def load_and_preprocess_data(self, data_path='Final.csv'):
        """Load and preprocess the data"""
        print("Loading data...")
        df = pd.read_csv(data_path)
        print(f"Data shape: {df.shape}")
        
        # Remove columns as done in the notebook
        remove_cols = [
            'Item_Identifier',
            'Item_Type',
            'Outlet_Identifier',
            'Outlet_Establishment_Year'
        ]
        df = df.drop(remove_cols, axis=1)
        
        # Separate features and target
        y = df.Item_Outlet_Sales.values
        X = df.drop('Item_Outlet_Sales', axis=1)
        
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Store feature columns for future predictions
        self.feature_columns = X.columns.tolist()
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X[self.columns_to_scale] = self.scaler.fit_transform(X[self.columns_to_scale])
        
        return X, y
    
    def train_model(self, X, y, use_hyperparameter_tuning=False):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        if use_hyperparameter_tuning:
            # Use hyperparameter tuning as shown in notebook
            from sklearn.model_selection import RandomizedSearchCV
            
            params = {
                "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
                "min_child_weight": [1, 3, 5, 7],
                "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
                "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
            }
            
            base_model = xgboost.XGBRegressor()
            random_search = RandomizedSearchCV(
                base_model, 
                param_distributions=params, 
                n_iter=5, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1, 
                cv=5, 
                verbose=1
            )
            
            random_search.fit(X, y)
            print(f"Best parameters: {random_search.best_params_}")
            self.model = random_search.best_estimator_
        else:
            # Use the final model configuration from notebook
            self.model = xgboost.XGBRegressor(
                base_score=0.5, 
                booster='gbtree', 
                colsample_bylevel=1,
                colsample_bynode=1, 
                colsample_bytree=1, 
                gamma=0.6, 
                importance_type='gain', 
                interaction_constraints='',
                learning_rate=0.4, 
                max_delta_step=0, 
                max_depth=15,
                min_child_weight=1, 
                monotone_constraints='()',
                n_estimators=100, 
                n_jobs=0, 
                num_parallel_tree=1, 
                random_state=0,
                reg_alpha=0, 
                reg_lambda=1, 
                scale_pos_weight=1, 
                subsample=1,
                tree_method='exact', 
                validate_parameters=1, 
                verbosity=0
            )
            self.model.fit(X, y)
        
        print("Model training completed!")
        return self.model
    
    def evaluate_model(self, X, y):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mae = np.sqrt(mean_absolute_error(y, y_pred))
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"\nModel Performance:")
        print(f"R² Score: {r2:.6f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        return r2, mae, rmse
    
    def train_test_split_evaluation(self, X, y, test_size=0.2, random_state=2):
        """Evaluate model with train-test split"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        print(f"\nTrain-Test Split:")
        print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
        
        # Train predictions
        y_train_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        
        # Test predictions
        y_test_pred = self.model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print(f"\nTraining Performance:")
        print(f"R² Score: {train_r2:.6f}")
        print(f"RMSE: {train_rmse:.4f}")
        
        print(f"\nTesting Performance:")
        print(f"R² Score: {test_r2:.6f}")
        print(f"RMSE: {test_rmse:.4f}")
        
        return {
            'train': {'r2': train_r2, 'rmse': train_rmse},
            'test': {'r2': test_r2, 'rmse': test_rmse}
        }
    
    def predict_new_data(self, new_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        if isinstance(new_data, dict):
            # Convert dict to DataFrame
            new_data = pd.DataFrame([new_data])
        
        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(new_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Reorder columns to match training data
        new_data = new_data[self.feature_columns]
        
        # Scale numerical features
        new_data_scaled = new_data.copy()
        new_data_scaled[self.columns_to_scale] = self.scaler.transform(new_data_scaled[self.columns_to_scale])
        
        # Make predictions
        predictions = self.model.predict(new_data_scaled)
        return predictions
    
    def save_model(self, model_path='xgboost_model.pkl', scaler_path='scaler.pkl'):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature columns
        with open('feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print("Feature columns saved to feature_columns.pkl")
    
    def load_model(self, model_path='xgboost_model.pkl', scaler_path='scaler.pkl'):
        """Load a pre-trained model and scaler"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open('feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
            print("Feature columns loaded from feature_columns.pkl")
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Train a new model first.")

def main():
    """Main function to demonstrate the XGBoost prediction workflow"""
    
    # Initialize predictor
    predictor = BigMartSalesPredictor()
    
    # Load and preprocess data
    X, y = predictor.load_and_preprocess_data('Final.csv')
    
    # Train model
    predictor.train_model(X, y, use_hyperparameter_tuning=False)
    
    # Evaluate model
    predictor.evaluate_model(X, y)
    
    # Train-test split evaluation
    predictor.train_test_split_evaluation(X, y)
    
    # Save the model
    predictor.save_model()
    
    # Example prediction on new data
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION ON NEW DATA")
    print("="*50)
    
    # Create sample data for prediction (using values from the first row of the dataset)
    sample_data = {
        'Item_Weight': 9.30,
        'Item_Visibility': 0.922960,
        'Item_MRP': 249.8092,
        'Outlet_Years': 14,
        'Item_Fat_Content_1': False,
        'Item_Fat_Content_2': False,
        'Outlet_Location_Type_1': False,
        'Outlet_Location_Type_2': False,
        'Outlet_Size_1': True,
        'Outlet_Size_2': False,
        'Outlet_Size_3': False,
        'Outlet_Type_1': True,
        'Outlet_Type_2': False,
        'Outlet_Type_3': False,
        'Item_Type_Combined_1': True,
        'Item_Type_Combined_2': False,
        'Outlet_1': False,
        'Outlet_2': False,
        'Outlet_3': False,
        'Outlet_4': False,
        'Outlet_5': False,
        'Outlet_6': False,
        'Outlet_7': False,
        'Outlet_8': False,
        'Outlet_9': True
    }
    
    try:
        prediction = predictor.predict_new_data(sample_data)
        print(f"\nSample Input Data:")
        for key, value in sample_data.items():
            print(f"  {key}: {value}")
        
        print(f"\nPredicted Item_Outlet_Sales: {prediction[0]:.2f}")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
    
    print("\n" + "="*50)
    print("PREDICTION WORKFLOW COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()
