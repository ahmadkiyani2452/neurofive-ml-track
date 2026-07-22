# NeuroFive Solutions – Machine Learning Fundamentals Internship

## Internee Information

- **Name:** Ahmad Munir
- **Internee ID:** NFS-2607-0391
- **Track:** Machine Learning Fundamentals
- **Organization:** NeuroFive Solutions

---

# Week 1

## 1. Set Up Your Data Science Toolkit & Explore Your First Dataset

- Set up the Python data science environment.
- Loaded and explored the Titanic dataset.
- Performed initial Exploratory Data Analysis (EDA).
- Checked dataset shape, columns, data types, and missing values.
- Used Pandas and NumPy for data analysis.
- Identified important features and prepared the dataset for further analysis.

**Tools:** Python, Pandas, NumPy, Google Colab

---

## 2. Clean & Visualize Real-World Data

- Cleaned the Titanic dataset.
- Filled missing `Age` values using the median.
- Filled missing `Embarked` values using the mode.
- Dropped the `Cabin` column due to many missing values.
- Detected potential outliers using boxplots.
- Created histogram, boxplot, bar chart, and correlation heatmap.
- Found that `Sex` and `Pclass` were important factors related to survival.

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn

---

# Week 2

## 3. Predict Titanic Survival — Your First Classification Model

- Built a Logistic Regression classification model.
- Used the Titanic dataset to predict passenger survival.
- Split data into 80% training and 20% testing sets.
- Encoded categorical features using `pd.get_dummies()`.
- Achieved approximately **80.45% accuracy**.
- Generated and analyzed a confusion matrix.

**Tools:** Python, Pandas, Scikit-learn

---

## 4. House Price Prediction with Linear Regression

- Built a Linear Regression model for house price prediction.
- Selected important numerical features for predicting house prices.
- Split the dataset into training and testing sets.
- Evaluated the model using RMSE and R² score.
- Created a predicted vs actual price scatter plot.
- Explained the meaning of the R² score in simple business terms.

**Tools:** Python, Pandas, NumPy, Matplotlib, Scikit-learn

---

# Week 3

## 5. Model Evaluation & Tuning: Beyond Accuracy

- Evaluated the Titanic classification model using Precision, Recall, and F1-score.
- Original Model Accuracy: **80.45%**
- Applied `GridSearchCV` for hyperparameter tuning.
- Best Parameters: `C=0.1`, `solver='lbfgs'`.
- Tuned Model Accuracy: **79.89%**.
- Compared the original and tuned model performance.
- Observed that tuning does not always improve model performance.

**Tools:** Python, Scikit-learn, GridSearchCV

---

## 6. Customer Churn Prediction — Working with a Business Problem

- Used the Telco Customer Churn dataset.
- Performed Exploratory Data Analysis to identify churn patterns.
- Trained Decision Tree and Logistic Regression models.
- Encoded categorical variables.
- Considered class imbalance in the churn dataset.
- Identified the top 3 Decision Tree churn drivers:
  1. `tenure`
  2. `InternetService_Fiber optic`
  3. `TotalCharges`
- Created a business summary with customer retention recommendations.

**Tools:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn

---

# Week 4

## 7. Build a Proper ML Pipeline with Feature Engineering

- Built a machine learning pipeline using Scikit-learn.
- Used `ColumnTransformer` for preprocessing.
- Applied `StandardScaler` to numerical features.
- Applied `OneHotEncoder` to categorical features.
- Combined preprocessing and model training into a single pipeline.
- Created engineered features to improve model performance.
- Saved the final machine learning pipeline using `joblib`.

**Tools:** Python, Pandas, Scikit-learn, Joblib

---

## 8. Ensemble Learning: Random Forest vs XGBoost

- Compared Logistic Regression, Random Forest, and XGBoost models using the Titanic dataset.
- Logistic Regression Accuracy: **80.45%**
- Random Forest Accuracy: **82.12%**
- XGBoost Accuracy: **79.89%**
- Random Forest achieved the best overall performance.
- Compared feature importance from Random Forest and XGBoost.
- Random Forest highly ranked `Fare`, `Sex_male`, and `Age`.
- XGBoost highly ranked `Sex_male`, `Pclass`, and `Age`.
- Explained the difference between Random Forest and XGBoost ensemble methods.

**Tools:** Python, Pandas, Scikit-learn, XGBoost

---

# Week 5

## 9. Handling Imbalanced & Messy Real-World Data

- Studied class imbalance using customer churn data.
- Compared an original model with a balanced model.
- Original Model:
  - Accuracy: **79.56%**
  - Precision: **63.35%**
  - Recall: **54.55%**
  - F1-score: **58.62%**
- Balanced Model:
  - Accuracy: **73.74%**
  - Precision: **50.34%**
  - Recall: **79.68%**
  - F1-score: **61.70%**
- Recall improved by **25.13 percentage points**.
- Demonstrated why accuracy alone can be misleading for imbalanced datasets.

**Tools:** Python, Pandas, Scikit-learn

---

## 10. Deploy Your Model as a Live Web App

- Selected a Titanic Survival Prediction model.
- Saved the trained model using `joblib`.
- Built a Streamlit web application.
- Added user input fields for passenger information.
- Added a prediction button.
- Displayed the predicted survival result.
- Displayed estimated survival probability.
- Deployed the application using Streamlit Community Cloud.

**Tools:** Python, Scikit-learn, Joblib, Streamlit, GitHub

### Live App

Add your Streamlit Community Cloud URL here:

`https://human-activity-app.streamlit.app/`

---

# Week 6

## 11. Capstone: End-to-End Machine Learning Project

### Human Activity Recognition System

- Built an end-to-end Machine Learning project using the UCI Human Activity Recognition dataset.
- Used smartphone sensor data to recognize human physical activities.
- Performed data cleaning and Exploratory Data Analysis.
- Checked missing values, infinite values, and duplicate rows.
- Performed feature engineering.
- Created two new features:
  - `sensor_mean`
  - `sensor_std`
- Compared three machine learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost

### Model Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 96.13% |
| Random Forest | 92.53% |
| XGBoost | 94.20% |

Logistic Regression achieved the best accuracy of approximately **96.13%** and was selected as the final model.

### Activities Recognized

The system recognizes six physical activities:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

### Deployment

A Streamlit web application was created to demonstrate the Human Activity Recognition model.

Users can upload sensor data and receive a predicted physical activity.

**Tools:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Joblib, Streamlit, GitHub

### Live App

Add your deployed Streamlit app link here:

`https://human-activity-app.streamlit.app/`

---

# Overall Skills Developed

During the NeuroFive Solutions Machine Learning Fundamentals Internship, I gained practical experience in:

- Python Programming
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Data Visualization
- Feature Engineering
- Classification
- Regression
- Logistic Regression
- Linear Regression
- Decision Trees
- Random Forest
- XGBoost
- Model Evaluation
- Hyperparameter Tuning
- Handling Imbalanced Data
- Machine Learning Pipelines
- Model Deployment
- Streamlit
- GitHub
- Joblib

---

# Conclusion

This internship provided hands-on experience with the complete machine learning workflow, starting from data exploration and cleaning to model training, evaluation, tuning, and deployment.

Through multiple practical projects, I worked with classification, regression, ensemble learning, imbalanced datasets, feature engineering, and machine learning pipelines.

The final capstone project, Human Activity Recognition, demonstrated my ability to take a real-world dataset through the complete machine learning lifecycle and deploy a working machine learning application.

**Thank you to NeuroFive Solutions for providing this valuable learning opportunity.**
