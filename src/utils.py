import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import pymysql
from dotenv import load_dotenv

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
    except Exception as ex:
        raise CustomException(ex,sys)

