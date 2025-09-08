import numpy as np
import pandas as pd
from xgboost_prediction import BigMartSalesPredictor
import warnings
warnings.filterwarnings('ignore')

def predict_single_item():
    """Function to make prediction for a single item"""
    
    # Initialize predictor and load trained model
    predictor = BigMartSalesPredictor()
    
    try:
        # Try to load existing model
        predictor.load_model()
    except:
        print("No pre-trained model found. Training new model...")
        # Load and train model
        X, y = predictor.load_and_preprocess_data('Final.csv')
        predictor.train_model(X, y)
        predictor.save_model()
    
    print("\n" + "="*60)
    print("BIGMART SALES PREDICTION SYSTEM")
    print("="*60)
    
    # Sample predictions with different scenarios
    scenarios = [
        {
            'name': 'High-end Dairy Product in Large Urban Store',
            'data': {
                'Item_Weight': 12.5,
                'Item_Visibility': 0.05,  # Low visibility (premium placement)
                'Item_MRP': 200.0,  # High price
                'Outlet_Years': 20,  # Established store
                'Item_Fat_Content_1': False,  # Low Fat
                'Item_Fat_Content_2': False,
                'Outlet_Location_Type_1': False,  # Tier 1 city
                'Outlet_Location_Type_2': False,
                'Outlet_Size_1': True,  # Large store
                'Outlet_Size_2': False,
                'Outlet_Size_3': False,
                'Outlet_Type_1': False,  # Supermarket Type1
                'Outlet_Type_2': True,
                'Outlet_Type_3': False,
                'Item_Type_Combined_1': True,  # Food category
                'Item_Type_Combined_2': False,
                'Outlet_1': False,
                'Outlet_2': False,
                'Outlet_3': True,  # Urban outlet
                'Outlet_4': False,
                'Outlet_5': False,
                'Outlet_6': False,
                'Outlet_7': False,
                'Outlet_8': False,
                'Outlet_9': False
            }
        },
        {
            'name': 'Regular Snack in Small Rural Store',
            'data': {
                'Item_Weight': 8.0,
                'Item_Visibility': 0.15,  # High visibility
                'Item_MRP': 45.0,  # Low price
                'Outlet_Years': 5,  # New store
                'Item_Fat_Content_1': True,  # Regular Fat
                'Item_Fat_Content_2': False,
                'Outlet_Location_Type_1': True,  # Tier 3 city
                'Outlet_Location_Type_2': False,
                'Outlet_Size_1': False,  # Small store
                'Outlet_Size_2': True,
                'Outlet_Size_3': False,
                'Outlet_Type_1': True,  # Grocery Store
                'Outlet_Type_2': False,
                'Outlet_Type_3': False,
                'Item_Type_Combined_1': False,  # Non-food category
                'Item_Type_Combined_2': True,
                'Outlet_1': True,  # Rural outlet
                'Outlet_2': False,
                'Outlet_3': False,
                'Outlet_4': False,
                'Outlet_5': False,
                'Outlet_6': False,
                'Outlet_7': False,
                'Outlet_8': False,
                'Outlet_9': False
            }
        },
        {
            'name': 'Medium Beverage in Average Store',
            'data': {
                'Item_Weight': 10.0,
                'Item_Visibility': 0.08,  # Medium visibility
                'Item_MRP': 120.0,  # Medium price
                'Outlet_Years': 12,  # Established store
                'Item_Fat_Content_1': False,  # Low Fat (beverages)
                'Item_Fat_Content_2': True,
                'Outlet_Location_Type_1': False,  # Tier 2 city
                'Outlet_Location_Type_2': True,
                'Outlet_Size_1': True,  # Medium store
                'Outlet_Size_2': False,
                'Outlet_Size_3': False,
                'Outlet_Type_1': False,  # Supermarket Type2
                'Outlet_Type_2': False,
                'Outlet_Type_3': True,
                'Item_Type_Combined_1': False,  # Drinks category
                'Item_Type_Combined_2': True,
                'Outlet_1': False,
                'Outlet_2': False,
                'Outlet_3': False,
                'Outlet_4': False,
                'Outlet_5': True,  # Medium outlet
                'Outlet_6': False,
                'Outlet_7': False,
                'Outlet_8': False,
                'Outlet_9': False
            }
        }
    ]
    
    # Make predictions for all scenarios
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        print("-" * len(f"{i}. {scenario['name']}:"))
        
        try:
            prediction = predictor.predict_new_data(scenario['data'])
            print(f"   Predicted Sales: ${prediction[0]:.2f}")
            
            # Show key characteristics
            print(f"   Key Features:")
            print(f"     - Item MRP: ${scenario['data']['Item_MRP']:.2f}")
            print(f"     - Item Weight: {scenario['data']['Item_Weight']:.1f}")
            print(f"     - Item Visibility: {scenario['data']['Item_Visibility']:.3f}")
            print(f"     - Outlet Age: {scenario['data']['Outlet_Years']} years")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "="*60)
    print("CUSTOM PREDICTION")
    print("="*60)
    
    # Example of how to create custom prediction
    print("\nTo make a custom prediction, modify the sample data below:")
    print("```python")
    print("custom_data = {")
    print("    'Item_Weight': 15.0,")
    print("    'Item_Visibility': 0.07,")
    print("    'Item_MRP': 180.0,")
    print("    'Outlet_Years': 15,")
    print("    'Item_Fat_Content_1': False,")
    print("    'Item_Fat_Content_2': True,")
    print("    'Outlet_Location_Type_1': False,")
    print("    'Outlet_Location_Type_2': True,")
    print("    'Outlet_Size_1': True,")
    print("    'Outlet_Size_2': False,")
    print("    'Outlet_Size_3': False,")
    print("    'Outlet_Type_1': False,")
    print("    'Outlet_Type_2': True,")
    print("    'Outlet_Type_3': False,")
    print("    'Item_Type_Combined_1': True,")
    print("    'Item_Type_Combined_2': False,")
    print("    'Outlet_1': False,")
    print("    'Outlet_2': False,")
    print("    'Outlet_3': False,")
    print("    'Outlet_4': True,")
    print("    'Outlet_5': False,")
    print("    'Outlet_6': False,")
    print("    'Outlet_7': False,")
    print("    'Outlet_8': False,")
    print("    'Outlet_9': False")
    print("}")
    print("```")

def predict_from_csv():
    """Function to make predictions from a CSV file"""
    
    print("\n" + "="*60)
    print("BATCH PREDICTION FROM CSV")
    print("="*60)
    
    # Initialize predictor and load trained model
    predictor = BigMartSalesPredictor()
    
    try:
        predictor.load_model()
    except:
        print("No pre-trained model found. Training new model...")
        X, y = predictor.load_and_preprocess_data('Final.csv')
        predictor.train_model(X, y)
        predictor.save_model()
    
    # Check if test data exists
    try:
        # Try to load test data
        test_data = pd.read_csv('Test.csv')
        print(f"Loaded test data with {len(test_data)} rows")
        
        # Since Test.csv might not have the same preprocessing, 
        # we'll show how to handle it
        print("\nNote: Test.csv would need the same preprocessing as training data.")
        print("This includes feature engineering, encoding, and scaling.")
        print("The prediction script handles this automatically for the Final.csv format.")
        
    except FileNotFoundError:
        print("Test.csv not found. To use batch prediction:")
        print("1. Prepare a CSV file with the same columns as the training data")
        print("2. Ensure all preprocessing steps are applied")
        print("3. Use predictor.predict_new_data(dataframe) for predictions")

if __name__ == "__main__":
    # Run single item predictions
    predict_single_item()
    
    # Show batch prediction example
    predict_from_csv()
    
    print("\n" + "="*60)
    print("PREDICTION EXAMPLES COMPLETED")
    print("="*60)
