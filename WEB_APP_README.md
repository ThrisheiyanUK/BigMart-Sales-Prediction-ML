# 🏪 BigMart Sales Prediction Web Application

## 🚀 Live Interactive Web Interface

This repository now includes a **fully functional web application** that brings the BigMart Sales Prediction model to life! Users can interact with the machine learning model through a beautiful, responsive web interface.

## ✨ Features

### 🎨 **Modern Web Interface**
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Interactive Dashboard**: Real-time analytics and visualizations  
- **Modern UI/UX**: Clean, professional design with smooth animations
- **Live Predictions**: Instant sales predictions with confidence scores

### 🤖 **Machine Learning Integration**
- **XGBoost Model**: High-performance gradient boosting algorithm
- **Real-time Predictions**: Instant sales forecasts based on user input
- **Feature Engineering**: Advanced preprocessing with encoded categorical variables
- **Model Persistence**: Pre-trained model loaded automatically

### 📊 **Analytics Dashboard**
- **Sales Metrics**: Total sales, product count, outlet analysis
- **Interactive Charts**: Visual data insights and trends
- **Performance Analytics**: Outlet-wise and product-wise performance
- **Sample Predictions**: Pre-configured scenarios for demonstration

## 🛠 **Installation & Setup**

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### 1. Clone Repository
```bash
git clone https://github.com/ThrisheiyanUK/BigMart-Sales-Prediction-ML.git
cd BigMart-Sales-Prediction-ML
```

### 2. Install Dependencies
```bash
pip install flask flask-cors pandas numpy scikit-learn xgboost joblib
```

### 3. Run the Application
```bash
python app.py
```

### 4. Access Web Interface
Open your browser and visit: **http://localhost:5000**

## 🎯 **How to Use**

### **Making Predictions**
1. **Fill out the prediction form** with product details:
   - Item Weight, Visibility, MRP
   - Fat Content (Low Fat/Regular)
   - Outlet Information (Size, Type, Location)
   - Product Category

2. **Click "Predict Sales"** to get instant results

3. **View Results** with:
   - Predicted sales amount
   - Confidence score
   - Business insights and recommendations

### **Dashboard Analytics**
- **Overview Metrics**: Total sales, products, outlets
- **Performance Charts**: Top products, outlet analysis
- **Sales Trends**: Historical patterns and forecasts

## 📁 **Project Structure**

```
BigMart-Sales-Prediction-ML/
├── 📊 Jupyter Notebooks/          # Original ML analysis
│   ├── 0 Hypotheses and Data Exploration.ipynb
│   ├── 1 Handling Missing Values.ipynb
│   ├── 2 Uni and Bivariate Analysis.ipynb
│   ├── 3 Feature Engineering.ipynb
│   ├── 4 Fitting Linear Regression.ipynb
│   ├── 5 Decision Tree Regression model.ipynb
│   ├── 6 Random Forest.ipynb
│   └── 7 XGBoost.ipynb
│
├── 🌐 Web Application/             # NEW: Interactive Web Interface
│   ├── app.py                     # Flask web server
│   ├── xgboost_prediction.py      # ML model wrapper
│   ├── templates/
│   │   └── index.html             # Main web interface
│   └── static/
│       ├── styles/main.css        # Modern styling
│       └── js/main.js             # Interactive features
│
├── 🤖 Model Files/                # Pre-trained models
│   ├── xgboost_model.pkl          # Trained XGBoost model
│   ├── scaler.pkl                 # Feature scaler
│   └── feature_columns.pkl        # Feature metadata
│
└── 📈 Data Files/                 # Datasets
    ├── Train.csv                  # Training data
    ├── Test.csv                   # Test data
    ├── Mid.csv                    # Intermediate processed data
    └── Final.csv                  # Final processed dataset
```

## 🎪 **Web Application Screenshots**

### 🏠 **Dashboard Overview**
- **Real-time Metrics**: Live sales data and KPIs
- **Interactive Charts**: Visualizations powered by Chart.js
- **Responsive Layout**: Optimized for all screen sizes

### 🔮 **Prediction Interface**
- **User-friendly Form**: Intuitive input fields with validation
- **Instant Results**: Real-time predictions with loading animations
- **Business Insights**: Actionable recommendations based on predictions

## 🧠 **Machine Learning Details**

### **Model Performance**
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Features**: 20+ engineered features including categorical encoding
- **Accuracy**: High-performance model with comprehensive validation
- **Preprocessing**: StandardScaler normalization and feature engineering

### **Prediction Features**
- Item Weight, Visibility, MRP
- Fat Content (One-hot encoded)
- Outlet Size, Type, Location (One-hot encoded)  
- Product Categories (Combined feature engineering)
- Outlet-specific encoding (10 outlet identifiers)

## 🚀 **Deployment Ready**

The web application is ready for deployment on:
- **Heroku** (with Procfile)
- **AWS EC2** (with requirements.txt)
- **Docker** (containerization ready)
- **Local Development** (instant setup)

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ **Author**

**Thrisheiyan UK**
- GitHub: [@ThrisheiyanUK](https://github.com/ThrisheiyanUK)
- Project: [BigMart Sales Prediction ML](https://github.com/ThrisheiyanUK/BigMart-Sales-Prediction-ML)

---

## 🎉 **What's New in Web Version**

✅ **Interactive Web Interface** - Beautiful, responsive design  
✅ **Real-time Predictions** - Instant ML-powered forecasts  
✅ **Analytics Dashboard** - Comprehensive data visualizations  
✅ **Business Insights** - Actionable recommendations  
✅ **Model Integration** - Seamless XGBoost model deployment  
✅ **Production Ready** - Scalable Flask architecture  

Transform your data science project into a professional web application! 🚀

---

*Built with ❤️ using Flask, XGBoost, and modern web technologies*
