# Import requirred libraries
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
     RandomForestRegressor,AdaBoostRegressor,
     GradientBoostingRegressor, VotingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.utils import save_object
from src.utils import evaluate_models
from src.utils import print_evaluated_results
from src.utils import model_metrics

# from urllib.parse import urlparse
# import mlflow
# import mlflow.sklearn

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join ('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                np.delete(train_array,6,axis=1),
                train_array[:,-1],
                np.delete(test_array,6,axis=1),
                test_array[:,-1]
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "GradientBoosting Regressor":GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                logging.info('Best model has r2 Score less than 60%')
                raise CustomException('No Best Model Found')
            
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            logging.info('Hyperparameter tuning started for catboosting Regressor')

            # Hyperparameter tuning on Catboost regressor
            # Initializing catboost regressor
            cbr = CatBoostRegressor(verbose=False)

            # Creating the hyperparameter grid
            param_dist = {'depth'          : [4,5,6,7,8,9, 10],
                          'learning_rate' : [0.01,0.02,0.03,0.04],
                          'iterations'    : [300,400,500,600]}

            #Initiate RandomSearchCV on catboost regerssor
            rscv = RandomizedSearchCV(cbr , param_dist, scoring='r2', cv =5, n_jobs=-1)

            # Fit the model
            rscv.fit(X_train, y_train)

            # Print the tuned parameters and score
            print(f'Best Catboost parameters : {rscv.best_params_}')
            print(f'Best Catboost Score : {rscv.best_score_}')
            print('\n====================================================================================\n')

            best_cbr = rscv.best_estimator_

            logging.info('Hyperparameter tuning completed for Catboost regressor')

            logging.info('Hyperparameter tuning started for KNN')

            # Initialize knn
            knn = KNeighborsRegressor()

            # parameters
            k_range = list(range(2, 31))
            param_grid = dict(n_neighbors=k_range)

            # Initiate and Fit the Grid search CV on knn
            grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2',n_jobs=-1)
            grid.fit(X_train, y_train)

            # Print the tuned parameters and score
            print(f'Best KNN Parameters : {grid.best_params_}')
            print(f'Best KNN Score : {grid.best_score_}')
            print('\n====================================================================================\n')

            best_knn = grid.best_estimator_

            logging.info('Hyperparameter tuning Complete for KNN')

            logging.info('Voting Regressor model training started')

            # Creating final Voting regressor
            vr = VotingRegressor([('cbr',best_cbr),('xgb',XGBRegressor()),('knn',best_knn)], weights=[3,2,1])
            vr.fit(X_train, y_train)

            print('Final Model Evaluation :\n')
            print_evaluated_results(X_train,y_train,X_test,y_test,vr)
            logging.info('Voting Regressor Training Completed')

            #this code need for mlflow
            #set url for mlflow
            # mlflow.set_registry_uri("https://dagshub.com/codemaestro908/diamond-project-ml.mlflow")
            # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            # #start MLFlow from here because we want to track experiments

            # with mlflow.start_run():
            #     predicted_qualities = best_model.predict(X_test)
                
            #     (rmse,mae,r2) = model_metrics(y_test, predicted_qualities)
                
            #     # mlflow.log_params(best_params_)

            #     mlflow.log_metric("rmse",rmse)
            #     mlflow.log_metric("r2", r2)
            #     mlflow.log_metric("mae",mae)

            #     #model registry does not work with file store
            #     if tracking_url_type_store !="file":
            #         #Register the model
            #         #These are other ways to use the model registry,
            #         #please refer to the more information

            #         mlflow.sklearn.log_model(best_model, "model", registered_model_name = best_model_name)
                
            #     else:
            #         mlflow.sklearn.log_model(best_model, "model")
            #end mlflow code

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = vr
            )
            logging.info('Model pickle file created')
            # Evaluating Ensemble Regressor (Voting Classifier on test data)
            y_test_pred = vr.predict(X_test)

            mae, rmse, r2 = model_metrics(y_test, y_test_pred)
            logging.info(f'test MAE : {mae}')
            logging.info(f'test RMSE : {rmse}')
            logging.info(f'Test R2 Score : {r2}')
            logging.info('Model Training Completed')
            
            return mae, rmse, r2 
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
