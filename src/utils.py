import os
import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pandas as pd

# import pymysql
from dotenv import load_dotenv
# from joblib import dump, load
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#data reading from sql database(.env file has been also created for basic database info)
#used in data_ingestion
load_dotenv()
host=os.getenv("host")
user=os.getenv("user")
password = os.getenv("password")
db = os.getenv('db')

def read_sql_data():
    logging.info("reading data from sql started")

    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("connection established with database",mydb)
        df = pd.read_sql_query('Select * from diamonds',mydb)
        print(df.head())
        return df

    except Exception as e:
        raise CustomException(e,sys)


# for saving directories
# used in data transformation 
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            # Predict Training data
            y_train_pred = model.predict(X_train)

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def model_metrics(original, predicted):
    try :
        mae = mean_absolute_error(original, predicted)
        mse = mean_squared_error(original, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(original, predicted)
        return mae, rmse, r2_square
    except Exception as e:
        logging.info('Exception Occured at model metric in utils')
        raise CustomException(e,sys)
    

def print_evaluated_results(X_train,y_train,X_test,y_test,model):
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluate Train and Test dataset
        train_mae , train_rmse, train_r2 = model_metrics(y_train, y_train_pred)
        test_mae , test_rmse, test_r2 = model_metrics(y_test, y_test_pred)

        # Printing results
        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(train_mae))
        print("- R2 Score: {:.4f}".format(train_r2))

        print('----------------------------------')
    
        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(test_mae))
        print("- R2 Score: {:.4f}".format(test_r2))
    
    except Exception as e:
        logging.info('Exception occured during printing of evaluated results')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

