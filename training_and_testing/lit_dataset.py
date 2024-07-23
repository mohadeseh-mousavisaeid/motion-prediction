import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from enums.motion_object import MotionObject
from rich.text import Text
from preProcessing import DataPreprocessor

class inD_RecordingDataset(Dataset):
    def __init__(self, path, recording_id, sequence_length, features_tracks, features_tracksmeta, motion_obj:MotionObject=None,  train=True):
        """Dataset for inD dataset.
        Parameters
        ----------
        path : str
            Path to the data.
        recording_id : int
            Recording id of the data.
        sequence_length : int
            Length of the sequence.
        features : list
            List of features to use.
        train : bool
            Whether to use the training set or not.
        """
        super(inD_RecordingDataset).__init__()
        self.path = path
        self.recording_id = recording_id
        self.sequence_length = sequence_length
        self.features_tracks = features_tracks
        self.features_tracksmeta = features_tracksmeta
        self.motion_obj = motion_obj
        self.train = train
        self.transform = self.get_transform()
        if type(self.recording_id) == list:
            self.data = pd.DataFrame()
            tracks_data = pd.DataFrame()
            tracksMeta_data = pd.DataFrame()
            # TODO: Here we are simply loading the csv and stack them into one pandas dataframe.
            # You have to change this to load your data. This is just meant as a dummy example!!!
            for id in self.recording_id:

                if(motion_obj==None):
                    all_report_txt = Text("Trial and Errors of the entire dataset!", style="bold green")
                    print(all_report_txt)
                    with open(f"{path}/{id}_tracks.csv", 'rb') as f:
                        self.data = pd.concat([self.data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracks, dtype='float16')])
                        print(self.data)
                else:
                    report_txt = Text("Trial and Errors on: " + str(motion_obj.name), style="bold green")
                    print(report_txt)
                    with open(f"{path}/{id}_tracks.csv", 'rb') as f:
                        tracks_data = pd.concat([tracks_data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracks, dtype='float64')])

                    with open(f"{path}/{id}_tracksMeta.csv", 'rb') as f:
                        tracksMeta_data = pd.concat([tracksMeta_data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracksmeta)])                  
                    
                    # ------- Preprocessing Steps -----------
                    
                    preprocessor = DataPreprocessor(data=tracks_data,meta_data=tracksMeta_data)
                    # preprocessor.downsample(fraction=0.8)
                    preprocessor.label_encode(join_on='trackId',join_method='left', motion_obj= self.motion_obj)
                    preprocessor.normalize()
                    self.data = preprocessor.get_processed_data()
                    


        else:
            self.data = pd.DataFrame()
            with open(f"{path}/{recording_id}_tracks.csv", 'rb') as f:
                tracks_data = pd.concat([self.data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracks, dtype='float64')])
                # self.data = pd.read_csv(f, delimiter=',', header=0, usecols=self.features, dtype='str')

            with open(f"{path}/{id}_tracksMeta.csv", 'rb') as f:
                preprocessor = DataPreprocessor(data=tracks_data,meta_data=tracksMeta_data)
                preprocessor.label_encode(join_on='trackId',join_method='left', motion_obj= self.motion_obj)
                preprocessor.normalize()
                self.data = preprocessor.get_processed_data()

    def __len__(self):
        """Returns the length of the dataset.
        """
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
                 Returns the item at index idx.
        Parameters
        ----------
        idx : int
            Index of the item.
        Returns
        -------
        data : torch.Tensor
            The data at index idx.
        """
        if idx <= self.__len__():
            # for each step this function will be called
            data = self.data[idx:idx + self.sequence_length]
            # data type tensor specific for pytorh is like and array
            if self.transform:
                data = self.transform(np.array(data)).squeeze()
            return data
        else:
            print("wrong index")
            return None

    def get_transform(self):
        """
        Returns the transform for the data.
        """
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        return data_transforms