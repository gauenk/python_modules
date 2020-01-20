import os
import numpy as np
from numpy import random as npr

class mnist():
    def __init__(self,path='/home/kent/Documents/data/mnist/'):
        self.path = path
        load_path = os.path.join(self.path,'train_images.npy')
        self.train_samples = np.load(load_path)
        self.num_train = len(self.train_samples)
        load_path = os.path.join(self.path,'train_labels.npy')
        self.train_labels = np.load(load_path)
        load_path = os.path.join(self.path,'test_images.npy')
        self.test_samples = np.load(load_path)
        self.num_test = len(self.test_samples)
        load_path = os.path.join(self.path,'test_labels.npy')
        self.test_labels = np.load(load_path)
        self.samples = self.train_samples
        self.labels = self.train_labels
        self.classes = [i for i in range(10)]
        self.mode = 'train'
        self.sample_index = 0
        
    def sample(self):
        sample = self.samples[self.sample_index]
        self.sample_index += 1
        return sample

    def shuffle(self):
        train_idx = npr.permutation(self.num_train)
        test_idx = npr.permutation(self.num_test)
        self.train_samples = self.train_samples[train_idx]
        self.train_labels = self.train_labels[train_idx]
        self.test_samples = self.test_samples[test_idx]
        self.test_labels = self.test_labels[test_idx]
        self.reset_mode()

    def reset_mode(self):
        self.set_mode(self.mode)

    def augment(self,aug_typ):
        if aug_type == 'rotate':
            pass
        else:
            raise NotImplementedError("unknown augmentation type: {}".format(aug_type))
    
    def set_mode(self,mode):
        self.mode = mode
        if self.mode is 'train':
            self.samples = self.train_samples
            self.labels = self.train_labels
        elif self.mode is 'test':
            self.samples = self.test_samples
            self.labels = self.test_labels
    
        
