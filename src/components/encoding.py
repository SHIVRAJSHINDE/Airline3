import sys
import os
from dataclasses import dataclass

import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import MinMaxScaler

from src.exception import CustomException
from src.logger import logging

#Since airline is nominal categorical data we will perform OneHotEncoding

# Since airline is nominal categorical data we will perform OneHotEncoding
class encodingclass:
    def __init__(self, encodedData):
        self.encodedData = encodedData

    def trainDataencoding(self):
        try:
            Airline = self.encodedData[["Airline"]]
            Source = self.encodedData[['Source']]
            Destination = self.encodedData[['Destination']]

            Airline = pd.get_dummies(Airline, drop_first=True)
            Source = pd.get_dummies(Source, drop_first=True)
            Destination = pd.get_dummies(Destination, drop_first=True)

            self.encodedData['Total_Stops'] = self.encodedData['Total_Stops'].replace(
                {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
            self.encodedData = pd.concat([self.encodedData, Airline, Source, Destination], axis=1)
            self.encodedData.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)

            self.encodedData.to_csv("D:\\COMPUTER VISOIN\\PROJECT\\Airine3\\artifacts\\encodedData.csv", index=False, header=True)

            print("Final Data")
            print(self.encodedData)
            return self.encodedData

        except Exception as e:
            raise CustomException(e, sys)



