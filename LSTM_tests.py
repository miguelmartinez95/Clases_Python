import sys
import time
import urllib.request
from git import Repo
import pytest
import numpy as np
import unittest
import pandas as pd

Repo.clone_from('https://github.com/miguelmartinez95/Clases_Python',r'E:\Documents\repo1')
sys.path.insert(1, 'E:\\Documents\\repo1\\DL')
from LSTM_model_v2 import LSTM_model


class Test_LSTM(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame(np.random.randint(0, 1000, size=(1000, 4)), columns=list('ABCD'))
        self.scalar_y=[]
        self.scalar_x=[]
        self.times = pd.Series(np.arange(self.data.shape[0]))
        self.pos_y = np.array([0,1])
        self.mask = True
        self.mask_value=-1
        self.inf_limit=np.array([0,0])
        self.sup_limit=np.array([1000,1000])
        self.names=['A','B','C','D']
        self.extract_cero = True
        self.repeat_vector=False
        self.dropout = 0
        self.weights=[]
        self.type='regression'

#    def test_supervised_horizonts(self):
#        d=self.data
#        for t in [0,1,5,10]:
#            lstm_model = LSTM_model(d, t, self.scalar_x, self.scalar_y, 'nothing', np.array([6, 20, 'nothing']), self.times,
#                                self.pos_y, self.mask, self.mask_value, 3, 1, self.inf_limit,
#                                self.sup_limit, self.names, self.extract_cero, self.repeat_vector, self.dropout, self.weights,
#                                self.type)
#
#            # LSTM
#            lstm_model.adjust_limits()
#            lstm_model.missing_values_masking_onehot()
#            lstm_model.missing_values_masking()
#
#            train =lstm_model.data.iloc[range(int(lstm_model.data.shape[0]/2))]
#            test =lstm_model.data.drop(lstm_model.data.index[range(int(lstm_model.data.shape[0]/2))])
#            results = lstm_model.train(train,test,np.array([10,10]),np.array([10]), 5, 128,False,[True,False],[],True)
#
#            print('HORIZONT',t)
#            self.assertEqual(len(results['ind_test']),len(results['Y_test']), 'different size between test_index and test')
#            self.assertEqual(len(results['ind_train']),len(results['Y_train']), 'different size between train_index and train')
#            self.assertEqual(len(results['train']),166, 'size of train three_dimension incorrect')
#            self.assertEqual(len(results['test']),166, 'size of test three_dimension incorrect')

    def test_supervised_steps(self):
        d=self.data.iloc[:, np.array([0,1])]
        self.pos_y = np.array([0])
        self.sup_limit=np.array([1000])
        self.inf_limit=np.array([0])
        for t in [1,5,10]:
            lstm_model = LSTM_model(d, 0, self.scalar_x, self.scalar_y, 'nothing', np.array([6, 20, 'nothing']), self.times,
                                self.pos_y, self.mask, self.mask_value, 3, t, self.inf_limit,
                                self.sup_limit, self.names, self.extract_cero, self.repeat_vector, self.dropout, self.weights,
                                self.type)

            # LSTM
            lstm_model.adjust_limits()
            lstm_model.missing_values_masking_onehot()
            lstm_model.missing_values_masking()

            train =lstm_model.data.iloc[range(int(lstm_model.data.shape[0]/2))]
            test =lstm_model.data.drop(lstm_model.data.index[range(int(lstm_model.data.shape[0]/2))])
            results = lstm_model.train(train,test,np.array([10,10]),np.array([10]), 5, 128,False,[True,False],[],True)

            print('STEPS',t)
            self.assertEqual(len(results['ind_test']),len(results['Y_test']), 'different size between test_index and test')
            self.assertEqual(len(results['ind_train']),len(results['Y_train']), 'different size between train_index and train')
            self.assertEqual(len(results['train']),166, 'size of train three_dimension incorrect')
            self.assertEqual(len(results['test']),166, 'size of test three_dimension incorrect')


if __name__ == "__main__":
    unittest.main()

