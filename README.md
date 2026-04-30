# SCT_ML_1
# House Price Prediction using Machine Learning

This project uses **Machine Learning Regression Techniques** to predict house prices based on important housing features such as living area, number of bedrooms, bathrooms, and other property details.

The model is trained on historical housing data to estimate property prices accurately and provide valuable insights for the real estate market.

---

## Project Overview

House price prediction is one of the most common real-world applications of Machine Learning. In this project:

- Housing dataset is loaded and analyzed
- Missing values are handled
- Important features are selected
- Data is prepared for training
- Linear Regression model is trained
- Predictions are generated on test data
- Model performance is evaluated using regression metrics

---

## Objective

To build an efficient predictive model that estimates house prices using supervised machine learning techniques.

---

## Key Features

- Data Cleaning & Preprocessing
- Feature Selection
- Exploratory Data Analysis
- Model Training using Linear Regression
- Performance Evaluation
- House Price Prediction
- Result Visualization

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Dataset Information

The dataset contains housing attributes such as:

- Living Area (`GrLivArea`)
- Bedrooms (`BedroomAbvGr`)
- Full Bathrooms (`FullBath`)
- Half Bathrooms (`HalfBath`)
- Sale Price (`SalePrice`)

---

## Project Structure

```bash
HousePricePrediction/
│── train.csv
│── test.csv
│── house_price_model.py
│── README.md
│── model.pkl
│── prediction_chart.png
│── feature_analysis.png
```

---

## Machine Learning Workflow

1. Load training and testing datasets  
2. Analyze dataset structure  
3. Handle missing values  
4. Select useful features  
5. Split training and validation data  
6. Train Linear Regression model  
7. Predict house prices  
8. Evaluate performance  
9. Visualize results  

---

## Model Used

### Linear Regression

Linear Regression is a supervised learning algorithm used to predict continuous values like house prices based on input features.

---

## Validation Metrics

```text
MAE  : $36,018.56
RMSE : $53,018.33
R²   : 0.6335
```

### Metric Explanation

- **MAE (Mean Absolute Error):** Average prediction error  
- **RMSE (Root Mean Squared Error):** Penalizes large errors  
- **R² Score:** Measures how well the model explains variance  

---

## How to Run

### Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run Project

```bash
python house_price_model.py
```

---

## Sample Output

```text
House Price Prediction Model

MAE  : $36,018.56
RMSE : $53,018.33
R²   : 0.6335

Prediction Completed Successfully
```

---

## Why Linear Regression?

Linear Regression is suitable for house price prediction because:

- Easy to understand and implement
- Fast training time
- Good baseline model for regression problems
- Shows relationship between features and price

---
