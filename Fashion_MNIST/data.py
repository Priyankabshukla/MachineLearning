import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import cfg

# TODO: Define your data path (the directory containing the 4 np array files)
DATA_PATH = 'fashion-mnist/data/processed/'

class FMNIST(Dataset):
    def __init__(self, set_name):
        super(FMNIST, self).__init__()
        # TODO: Retrieve all the images and the labels, and store them
        # as class variables. Maintaing any other class variables that 
        # you might need for the other class methods. Note that the 
        # methods depends on the set (train or test) and thus maintaining
        # that is essential.
        self.set_name = set_name
        self.images = np.load(DATA_PATH+f'{set_name}_images.npy')
        self.labels=np.load(DATA_PATH+f'{set_name}_labels.npy')
        
#         raise NotImplementedError
    
    def __len__(self):
        # TODO: Complete this
        return len(self.images)
#         raise NotImplementedError
    
    def __getitem__(self, index):
        # TODO: Complete this
        ##convert to tensor
        image = torch.from_numpy(np.asarray(self.images[index]/255.0))
        label = torch.from_numpy(np.asarray(self.labels[index]))
        return image, label
#         raise NotImplementedError

def get_data_loader(set_name):
    # TODO: Create the dataset class tailored to the set (train or test)
    # provided as argument. Use it to create a dataloader. Use the appropriate
    # hyper-parameters from cfg

    '''
    Parameters:
            set_name - the dataset to be loaded with the DataLoader

    Returns:
            The DataLoader witth the given parameters
    '''
    
    
    if set_name == 'train':
        train_loader = DataLoader(dataset = FMNIST('train'),
                                  batch_size = cfg['batch_size'],
                                  num_workers = cfg['num_workers'],
                                  shuffle=True)
        return train_loader
    
    elif set_name == 'test':
        test_loader =  DataLoader(dataset = FMNIST('test'),
                                  batch_size = cfg['batch_size'],
                                  num_workers = cfg['num_workers'],
                                  shuffle = False)
        return test_loader
        
#     raise NotImplementedError
