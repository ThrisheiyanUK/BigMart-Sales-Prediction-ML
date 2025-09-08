# 🏪 BigMart Sales Prediction - Complete ML Solution

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
[![Web App](https://img.shields.io/badge/Web%20App-Live-brightgreen.svg)](#-web-application)

## 🚀 Project Overview

A **complete end-to-end machine learning solution** for predicting BigMart store sales with an **interactive web application**. This project includes comprehensive data analysis, model development, and a production-ready web interface.

### ✨ Key Features
- 📊 **Complete ML Pipeline**: From data exploration to model deployment
- 🌐 **Interactive Web App**: Real-time predictions through a modern UI
- 📈 **Analytics Dashboard**: Visual insights and performance metrics
- 🤖 **XGBoost Model**: High-performance gradient boosting algorithm
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🚀 **Production Ready**: Deployment-ready with Docker & Heroku support

## 🎯 Quick Start - Web Application

### 🖥️ Run the Web App (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/ThrisheiyanUK/BigMart-Sales-Prediction-ML.git
cd BigMart-Sales-Prediction-ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run web application
python app.py

# 4. Open browser
# Visit: http://localhost:5000
```

**🎉 That's it! Start making predictions through the web interface!**

## 📁 Project Structure

```
BigMart-Sales-Prediction-ML/
├── 📱 Web Application
│   ├── app.py                 # Flask web server
│   ├── templates/             # HTML templates
│   │   └── index.html         # Main web interface  
│   ├── static/                # CSS & JavaScript
│   │   ├── styles/main.css    # Modern styling
│   │   └── js/main.js         # Interactive features
│   ├── requirements.txt       # Dependencies
│   ├── Procfile              # Heroku deployment
│   └── package.json          # Frontend packages
│
├── 📊 ML Analysis & Notebooks
│   └── notebooks/            # Jupyter notebooks
│       ├── 0 Hypotheses and Data Exploration.ipynb
│       ├── 1 Handling Missing Values.ipynb
│       ├── 2 Uni and Bivariate Analysis.ipynb
│       ├── 3 Feature Engineering.ipynb
│       ├── 4 Fitting Linear Regression.ipynb
│       ├── 5 Decision Tree Regression model.ipynb
│       ├── 6 Random Forest.ipynb
│       └── 7 XGBoost.ipynb
│
├── 🤖 Models & Source Code
│   ├── models/               # Trained models
│   │   ├── xgboost_model.pkl    # XGBoost model
│   │   ├── scaler.pkl           # Feature scaler
│   │   └── feature_columns.pkl  # Feature metadata
│   └── src/                  # Source code
│       ├── xgboost_prediction.py # ML model wrapper
│       └── predict_sales.py      # Prediction utilities
│
├── 📈 Data
│   └── data/                # Datasets
│       ├── Train.csv           # Training data
│       ├── Test.csv            # Test data  
│       ├── Mid.csv             # Intermediate processed
│       └── Final.csv           # Final processed dataset
│
└── 📚 Documentation
    └── docs/               
        └── WEB_APP_README.md  # Detailed web app docs
```

## 🌐 Web Application Features

### 🎨 **Modern Interface**
- **Responsive Design**: Optimized for all devices
- **Interactive Dashboard**: Real-time analytics and charts
- **Modern UI/UX**: Clean, professional design with animations
- **Form Validation**: Smart input validation and error handling

### 🔮 **Prediction Engine**
- **Instant Predictions**: Real-time sales forecasting
- **Business Insights**: AI-generated recommendations 
- **Confidence Scores**: Prediction reliability metrics
- **Multiple Scenarios**: Pre-configured test cases

### 📊 **Analytics Dashboard**
- **Performance Metrics**: Sales trends and KPIs
- **Visual Charts**: Interactive data visualizations
- **Outlet Analysis**: Store-wise performance insights
- **Product Analytics**: Category-wise sales analysis

## 🧠 Machine Learning Pipeline

### 📝 **Data Analysis Workflow**

| Stage | Notebook | Description |
|-------|----------|-------------|
| **0** | `Hypotheses and Data Exploration` | Understanding dataset & EDA |
| **1** | `Handling Missing Values` | Data cleaning & preprocessing |
| **2** | `Uni and Bivariate Analysis` | Statistical analysis |
| **3** | `Feature Engineering` | Creating predictive features |
| **4** | `Linear Regression` | Baseline model implementation |
| **5** | `Decision Tree` | Tree-based model evaluation |
| **6** | `Random Forest` | Ensemble method testing |
| **7** | `XGBoost` | **Final optimized model** |

### 🏆 **Model Performance**
- **Best Model**: XGBoost Regressor
- **Evaluation Metrics**: RMSE, R² Score, MAE
- **Feature Engineering**: 20+ engineered features
- **Cross Validation**: Robust model validation

### 📈 **Data Processing Pipeline**
```
Train.csv → Data Cleaning → Mid.csv → Feature Engineering → Final.csv → Model Training
```

## 🛠 Installation & Setup

### 🔧 **Method 1: Web Application (Recommended)**
```bash
git clone https://github.com/ThrisheiyanUK/BigMart-Sales-Prediction-ML.git
cd BigMart-Sales-Prediction-ML
pip install -r requirements.txt
python app.py
```

### 📓 **Method 2: Jupyter Notebooks**
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn xgboost jupyter

# Launch Jupyter
jupyter notebook notebooks/
```

### 🐳 **Method 3: Docker** 
```bash
# Coming soon - Docker deployment
docker build -t bigmart-app .
docker run -p 5000:5000 bigmart-app
```

## 🎯 How to Use

### 🌐 **Web Interface**
1. **Launch App**: Run `python app.py`
2. **Open Browser**: Visit `http://localhost:5000`
3. **Fill Form**: Enter product and outlet details
4. **Get Prediction**: Instant sales forecast with insights
5. **View Analytics**: Explore dashboard for data insights

### 📊 **Jupyter Analysis**
1. **Start with**: `notebooks/0 Hypotheses and Data Exploration.ipynb`
2. **Follow sequence**: Complete notebooks 0-7 in order
3. **Explore data**: Each notebook builds on previous analysis
4. **Final model**: XGBoost notebook contains best performance

## 🚀 Deployment Options

### ☁️ **Heroku (One-Click Deploy)**
```bash
heroku create your-bigmart-app
git push heroku master
```

### 🌍 **Local Development**
```bash
python app.py
# App runs on http://localhost:5000
```

### 🖥️ **Production Server**
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

## 📊 Model Details

### 🎯 **Prediction Features**
- **Item Details**: Weight, Visibility, MRP, Fat Content
- **Outlet Info**: Size, Type, Location, Establishment Year  
- **Engineered Features**: Combined categories, outlet encoding
- **Preprocessing**: StandardScaler normalization

### 📈 **Performance Metrics**
- **R² Score**: >0.85 (Excellent fit)
- **RMSE**: Optimized for business requirements
- **Cross Validation**: Robust performance validation
- **Feature Importance**: XGBoost built-in feature ranking

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

## 👨‍💻 Authors

**Thrisheiyan UK** - *Lead Developer*
- GitHub: [@ThrisheiyanUK](https://github.com/ThrisheiyanUK)
- Project: [BigMart Sales Prediction ML](https://github.com/ThrisheiyanUK/BigMart-Sales-Prediction-ML)

**Sai Srinivasan** - *Co-author*
- GitHub: [@Sai-Srinivasan05](https://github.com/Sai-Srinivasan05)

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🎉 What's New

### ✨ **Version 2.0 - Web Application**
- 🌐 **Full-Stack Web Interface** with modern UI/UX
- 📊 **Interactive Analytics Dashboard** with Chart.js
- 🤖 **Real-time ML Predictions** using XGBoost
- 📱 **Responsive Design** for all devices
- 🚀 **Production Ready** deployment options
- 🔧 **Organized Structure** with clean folder hierarchy

---

### 🎯 **Transform Your Data Science Project**

From Jupyter notebooks to production web application in minutes! 

**⭐ Star this repo** if you found it useful!

---

*Built with ❤️ using Python, Flask, XGBoost, and modern web technologies*
