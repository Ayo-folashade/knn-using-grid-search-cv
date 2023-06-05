# Breast Cancer Classification with K-Nearest Neighbors (KNN)

This repository contains code for classifying breast cancer using the K-Nearest Neighbors (KNN) algorithm. The code performs a grid search to find the best set of hyperparameters for the KNN classifier and evaluates its performance on the breast cancer dataset.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, which contains features computed from digitized images of breast mass.

## Dependencies

The required dependencies can be installed using:
```
pip install -r requirements.txt
```

## Technical Overview

- Data Preprocessing: The code reads the dataset from a CSV file and performs preprocessing steps such as dropping unnecessary columns and splitting the data into input features and the target variable.

- Grid Search: The code defines a parameter grid that specifies different values to be explored for each hyperparameter of the KNN classifier. It then creates a KNN classifier and a GridSearchCV object. The GridSearchCV performs an exhaustive search over the parameter grid, fitting the training data and evaluating different hyperparameter combinations using cross-validation.

- Best Parameters and Score: The code retrieves the best set of hyperparameters found by the grid search and the corresponding best score (accuracy). It prints these values as the output.

## Results

The code outputs the best set of hyperparameters found by the grid search and the corresponding accuracy score achieved on the training data. The best set of hyperparameters can be used to train a KNN classifier for breast cancer classification.

## Conclusion

This project demonstrates how to perform hyperparameter tuning using grid search for a KNN classifier on the breast cancer dataset. By finding the best hyperparameter values, we can improve the model's performance and make more accurate predictions for breast cancer diagnosis.

