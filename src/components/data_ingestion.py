import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
#from src.components.eda import EDA
#from src.components.encoding import encodingclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:

            logging.info("Entered data ingestion method")
            df = pd.read_csv('D:\\COMPUTER VISOIN\\PROJECT\\Airine3\\Airline\\Data_Train.csv')
            logging.info("Read the train data")
            print(df.head())

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path)
            print(self.ingestion_config.raw_data_path)

            return(self.ingestion_config.raw_data_path)

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    dataIngestionObj = DataIngestion()
    dataIngestionObj.initiate_data_ingestion()

