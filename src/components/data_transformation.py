import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numPipelineColumns = ['JourneyDay','JourneyMonth','JourneyYear','Dep_Time_Hr','Dep_Time_Min','Arr_Time_Hr','Arr_Time_Min','DurationOnlyInMinutes']

            catOne = ['Airline','Source','Destination']
            catTwo = ['Total_Stops']

            numPipeline = Pipeline(
                    steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                    ])


            catOnePipeline = Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ])

            catTwoPipeline = Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("label_encoder",LabelEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ])


            logging.info(f"Categorical columns: {catOnePipeline}")
            logging.info(f"Categorical columns: {catTwoPipeline}")
            logging.info(f"Numerical columns: {numPipeline}")

            preprocessor = ColumnTransformer([
                ("catOnePipeline", catOnePipeline, catOne),
                ("catTwoPipeline", catTwoPipeline, catTwo),
                ("numPipeline", numPipeline, numPipelineColumns),

            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Price"
            numPipelineColumns = ['JourneyDay','JourneyMonth','JourneyYear','Dep_Time_Hr','Dep_Time_Min','Arr_Time_Hr','Arr_Time_Min','DurationOnlyInMinutes']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            print("input_feature_train_df")
            print(input_feature_train_df)
            print("input_feature_train_df")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

