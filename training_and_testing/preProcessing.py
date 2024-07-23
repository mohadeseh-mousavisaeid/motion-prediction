import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from pandas import DataFrame
from enums.motion_object import MotionObject
import logging

class DataPreprocessor:
    def __init__(self, data:DataFrame, meta_data:DataFrame):
        self.data = data
        self.meta_data = meta_data

    def downsample(self, fraction:float):
       
        if not (0 < fraction <= 1):
            raise ValueError("Fraction must be between 0 and 1.")
        
        logging.info(f"Downsampling data to {fraction*100}% of its original size.")
        self.data = self.data.sample(frac=fraction, random_state=42).reset_index(drop=True)
    
    def label_encode(self, join_on:str , join_method:str, motion_obj:MotionObject):
        
        le = LabelEncoder()
        item_types = np.array(self.meta_data['class'])
        self.meta_data['class'] = le.fit_transform(item_types)
        merged_data = self.data.merge(self.meta_data, on=join_on, how=join_method)
        self.data = self.data[(merged_data["class"] == motion_obj.value)]
                            
        encoded_values = list(le.classes_)
        actual_values = sorted(list(self.meta_data['class'].unique()))

    def normalize(self):
        numerical_features = self.data.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_features) == 0:
            raise ValueError("No numerical features found to normalize.")
        
        # ---------------------------- Normalization Steps------------------------------------
        
        # Min-Max
        scaler = MinMaxScaler()
        # columns_to_normalize = self.data.columns[1:9]
        self.data[numerical_features]= scaler.fit_transform(self.data[numerical_features])   
                         
        # Z-Score Normalization
        scaler = StandardScaler()
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])
        
    def get_processed_data(self):
        return self.data
