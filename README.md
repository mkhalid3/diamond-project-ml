# Diamond Price Prediction

This project focuses on predicting diamond prices using various machine learning algorithms. It involves data extraction, preprocessing, model training, evaluation, and deployment. The entire workflow is tracked using DVC and MLflow, with the final model deployed on AWS Elastic Beanstalk.

## Project Overview

The goal of this project is to build a machine learning model that accurately predicts the price of diamonds based on various features. The dataset is sourced from Kaggle and is processed using SQL for extraction. Multiple regression algorithms are applied, and their performance is evaluated using standard metrics.

## Dataset

- **Source**: Kaggle
- **Extraction**: SQL
- **Train-Test Split**: 80% for training and 20% for testing

## Data Preprocessing

The data was preprocessed using the following steps:

1. **Data Extraction**: The dataset was extracted from the database using SQL.
2. **Train-Test Split**: The data was split into 80% for training and 20% for testing.
3. **Feature Scaling**: Standard scaling techniques were applied to normalize the data.

## Models Implemented

The following machine learning algorithms were implemented:

- **Linear Regression**
- **Lasso**
- **Ridge**
- **K-Neighbors Regressor**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **XGBRegressor**
- **CatBoosting Regressor**
- **GradientBoosting Regressor**
- **AdaBoost Regressor**

## Evaluation Metrics

The models were evaluated using the following metrics:

- **R2 Score**
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

## Experiment Tracking

- **DVC**: Used for tracking data and model versioning.
- **MLflow**: Used for tracking experiments, including hyperparameters, metrics, and models.

## Deployment

The final model was deployed on **AWS Elastic Beanstalk**. The deployment process involves creating an environment on AWS, packaging the application, and uploading it to the environment for deployment.
