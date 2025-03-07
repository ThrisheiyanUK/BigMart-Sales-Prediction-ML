# Big Mart Sales Prediction  

## Project Overview  
This project aims to predict sales for Big Mart stores using machine learning models. The workflow includes **hypothesis generation, data exploration, handling missing values, feature engineering, and model evaluation** to determine the most accurate predictor.  

## Project Structure  
The project follows a structured approach, with each stage covered in separate Jupyter Notebooks:  

1. **[0 Hypotheses and Data Exploration.ipynb](0%20Hypotheses%20and%20Data%20Exploration.ipynb)**  
   - Understanding the dataset and formulating key hypotheses  
   - Performing Exploratory Data Analysis (EDA)  

2. **[1 Handling Missing Values.ipynb](1%20Handling%20Missing%20Values.ipynb)**  
   - Identifying and handling missing data  
   - Generating `Mid.csv` after cleaning  

3. **[2 Uni and Bivariate Analysis.ipynb](2%20Uni%20and%20Bivariate%20Analysis.ipynb)**  
   - Analyzing individual variables (univariate)  
   - Studying relationships between features (bivariate)  

4. **[3 Feature Engineering.ipynb](3%20Feature%20Engineering.ipynb)**  
   - Creating new features to enhance model performance  

5. **[4 Fitting Linear Regression.ipynb](4%20Fitting%20Linear%20Regression.ipynb)**  
   - Implementing and evaluating **Linear Regression**  

6. **[5 Decision Tree Regression model.ipynb](5%20Decision%20Tree%20Regression%20model.ipynb)**  
   - Implementing and evaluating **Decision Tree Regression**  

7. **[6 Random Forest.ipynb](6%20Random%20Forest.ipynb)**  
   - Implementing and evaluating **Random Forest Regression**  

8. **[7 XGBoost.ipynb](7%20XGBoost.ipynb)**  
   - Implementing and evaluating **XGBoost**  

9. **Final Comparison**  
   - Evaluating models using RMSE, R² Score  
   - Selecting the best model based on performance  

## Dataset Workflow  
- **`Test.csv`** – Used initially for testing  
- **`Train.csv`** – Used for training models  
- **`Mid.csv`** – Generated after handling missing values  
- **`Final.csv`** – Processed dataset for final modeling  

## 🏗Model Evaluation  
- **Metrics used:** RMSE, R² Score  
- **Goal:** Identify the best model for accurate sales prediction  

## Installation & Usage  
### Prerequisites  
- Python  
- Jupyter Notebook  
- Required libraries:  
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn xgboost
## Co-authored-by
Sai Srinivasan - [@Sai-Srinivasan05](https://github.com/Sai-Srinivasan05)

