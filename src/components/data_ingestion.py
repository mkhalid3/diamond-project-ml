# Import all the required libraries
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.utils import read_sql_data

from src.components.data_transformation import DataTransformation, DataTransformationConfig
# from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

# Initialize Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts','data.csv')
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    
# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info('Data ingestion Started')
        try:
            # dataset reading from my sql database
            # df = read_sql_data()

            #read from file
            df = pd.read_csv('notebook/data/diamonds.csv')
            logging.info('Dataset read from my sql database')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            logging.info('splitting data into train and test datasets')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=0)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('data_ingestion is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)
    
# Run Data ingestion
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initate_data_transformation(train_data,test_data)

    # modeltrainer = ModelTrainer()
    # modeltrainer.initate_model_training(train_arr, test_arr)