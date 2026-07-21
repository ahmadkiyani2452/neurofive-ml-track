# Titanic Survival Prediction — Logistic Regression

## Project Overview

This project is part of the Machine Learning Fundamentals track at NeuroFive Solutions.

The goal of this project is to build a classification model that predicts whether a Titanic passenger survived or did not survive.

## Dataset

The Titanic dataset was used for this project. The dataset contains information about passengers, including:

* Passenger class
* Sex
* Age
* Number of siblings and spouses
* Number of parents and children
* Fare
* Port of Embarkation

## Data Preprocessing

The following preprocessing steps were performed:

* Missing Age values were filled using the median.
* Missing Embarked values were filled using the mode.
* The Cabin column was removed because it contained a large number of missing values.
* Categorical columns such as Sex and Embarked were converted into numerical features using pandas get_dummies().

## Model

A Logistic Regression classification model was trained using scikit-learn.

The dataset was divided into:

* 80% Training Data
* 20% Testing Data

The model was evaluated using accuracy_score and a confusion matrix.

## Final Accuracy

The final model achieved an accuracy of approximately **[80.45]%** on the test dataset.

## Conclusion

The Logistic Regression model successfully predicted Titanic passenger survival based on selected passenger features. The confusion matrix was also used to understand correct predictions and classification errors.
