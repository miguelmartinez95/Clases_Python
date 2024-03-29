import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from errors import Eval_metrics as evals
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from time import time
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import RepeatVector
from keras.layers import Dropout
from keras.constraints import maxnorm
#import skfda
import math
import multiprocessing
from multiprocessing import Process,Manager,Queue
import collections
from pathlib import Path
import random
from datetime import datetime
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.factory import get_decomposition
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.optimize import minimize
from pymoo.core.problem import starmap_parallelized_eval

'''
Conexion con GPUs
'''

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)


class DL:
    def info(self):
        '''
        data: dataframe with the data
        horizont: how many steps are considered to the future
        scalar x: scalar object empty to be fill with the trained scalar for x
        scalar y: scalar object empty to be fill with the trained scalar for y
        zero_problem: relation with the night and day schedule. Can be nothing, schedule or radiation
        times: dates object
        limits: threshold to evaluate the outputs: f.e. radiation > 0.001
        pos_y: number of colunm or columns where the output data are
        n_lags: time lags considered to train the model
        mask: masking true or false
        mask_value: if mask is true the value for masking
        sup_limit:upper limit wihitn the output of the model must be
        inf_limit: lower limit wihitn the output of the model must be
        namnes: columns labels
        :return:
        '''

        print('Super class to built different deep learning models. This class has other more specific classes associated with it ')

    def __init__(self, data, horizont,scalar_y, scalar_x,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,  inf_limit,sup_limit,names,extract_cero):
        self.data = data
        self.horizont = horizont
        self.scalar_y = scalar_y
        self.scalar_x = scalar_x
        self.scalar_x = scalar_x
        self.zero_problem = zero_problem
        self.times = times
        self.limits =limits
        self.pos_y = pos_y
        self.n_lags  =n_lags
        self.mask = mask
        self.mask_value = mask_value
        self.sup_limit = sup_limit
        self.inf_limit = inf_limit
        self.names=names
        self.extract_cero = extract_cero

    @staticmethod
    def cv_division(x,y, fold):
        '''
        Division de la muestra en trozos según fold para datos normales y no recurrentes
        Para datos recurrentes más abajo está cv_division_lstm

        :param fold: division for cv_analysis
        :return: data divided into train, test and validations in addition to the indexes division
        '''
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X_test = []
        X_train = []
        X_val = []
        Y_test = []
        Y_train = []
        Y_val = []

        step = int(x.shape[0]/fold)
        w = 0
        w2 = step
        indexes = []
        try:
            while w2 < x.shape[0]:
                a = x.iloc[range(w, w2)]
                X_val.append(a.iloc[range(len(a) - math.ceil(len(a) / 2), len(a) - 1)])
                X_test.append(a.drop(a.index[range(len(a) - math.floor(len(a) / 2), len(a))]))
                X_train.append(x.drop(range(w, w2)))

                a = y.iloc[range(w, w2)]
                Y_val.append(a.iloc[range(len(a) - math.ceil(len(a) / 2), len(a) - 1)])
                Y_test.append(a.drop(a.index[range(len(a) - math.floor(len(a) / 2), len(a))]))
                Y_train.append(y.drop(range(w, w2)))
                indexes.append(np.array([w2 - math.ceil(len(a) / 2) + 1, w2]))
                w = w2
                w2 += step
                if (w2 > x.shape[0] and w < x.shape[0]):
                    w2 = x.shape[0]
        except:
            raise NameError('Problems with the sample division in the cv classic')
        res = {'x_test': X_test, 'x_train':X_train,'X_val':X_val, 'y_test':Y_test, 'y_train':Y_train, 'y_val':Y_val,'indexes':indexes}
        return res

    @staticmethod
    def fix_values_0(restriction, zero_problem, limit):
        '''
        Function to fix incorrect values based on some restriction

        :param restriction: schedule hours or irradiance variable depending on the zero_problem
        :param zero_problem: schedule or radiation
        :param limit: limit hours or radiation limit
        :return: the indexes where the data are not in the correct time schedule or is below the radiation limit
        '''
        if zero_problem == 'schedule':
            try:
                limit1 = int(limit[0])
                limit2 = int(limit[1])

                if limit[2]=='weekend':
                    wday = pd.Series(restriction).dt.weekday
                    ii1 = np.where(wday>4)[0]

                    hours = pd.Series(restriction).dt.hour
                    ii = np.where((hours < limit1) | (hours > limit2))[0]
                    print(ii)
                    ii = np.union1d(ii1, ii)

                else:
                    hours = pd.Series(restriction).dt.hour
                    ii = np.where((hours < limit1) | (hours > limit2))[0]


            except:
                raise NameError('Zero_problem and restriction incompatibles')
        elif zero_problem == 'radiation':
            try:
                rad = np.array([restriction])
                ii = np.where(rad <= limit)[0]
            except:
                raise NameError('Zero_problem and restriction incompatibles')
        else:
            ii=[]
            'Unknown situation with nights'

        res = {'indexes_out': ii}
        return res


    @staticmethod
    def cortes(x, D, lim):
        '''
        :param x:
        :param D: length of data
        :param lim: dimension of the curves
        :return: data divided in curves of specific length
        '''
        Y = np.zeros((lim, int(D/lim)))
        i=0
        s=0
        while i<=D:
            if D-i < lim:
                Y = np.delete(Y, s-1, 1)
                break
            else:
                Y[:,s] = x[i:(i+lim)]
                i +=lim
                s += 1
                if i==D:
                    break
        return(Y)


    @staticmethod
    def ts(new_data, look_back, pred_col, names, lag):
        '''
        :param look_back: amount of lags
        :param pred_col: variables to lagged
        :param names: name variables
        :param lag: name of lag-- variable lag1
        :return: dataframe with lagged data
        '''

        t = new_data.copy()
        t['id'] = range(0, len(t))
        t = t.iloc[look_back:, :]
        t.set_index('id', inplace=True)
        pred_value = new_data.copy()
        pred_value = pred_value.iloc[:-look_back, pred_col]
        names_lag = [0 for x in range(len(pred_col))]
        for i in range(len(pred_col)):
            names_lag[i] = names[i] + '.' + str(lag)
        pred_value.columns = names_lag
        pred_value = pd.DataFrame(pred_value)

        pred_value['id'] = range(1, len(pred_value) + 1)
        pred_value.set_index('id', inplace=True)
        final_df = pd.concat([t, pred_value], axis=1)

        return final_df

    def introduce_lags(self, lags, var_lag):
        '''
        Introduction of lags moving the sample

        :param lags: amount of lags
        :param var_lag: label of lagged variables
        :return: data lagged
        '''
        d1 = self.data.copy()
        dim = d1.shape[1]
        selec = range(dim - var_lag, dim)
        try:
            names1 = self.data.columns[selec]
            for i in range(self.n_lags):
                self.data = self.ts(self.data, lags, selec,  names1, i + 1)

                selec = range(dim, dim + var_lag)
                dim += var_lag
            self.times = self.data.index
        except:
            raise NameError('Problems introducing time lags')

    def adjust_limits(self):
        '''
        Adjust the data or the variable to certain upper or lower limits
        '''
        if isinstance(self.pos_y, collections.abc.Sized):
            for t in range(len(self.pos_y)):
                inf = np.where(self.data.iloc[:, self.pos_y[t]] < self.inf_limit[t])[0]
                sup = np.where(self.data.iloc[:, self.pos_y[t]] > self.sup_limit[t])[0]
                if len(inf) > 0:
                    self.data.iloc[inf, self.pos_y[t]] = np.repeat(self.inf_limit[t], len(inf))
                if len(sup) > 0:
                    self.data.iloc[sup, self.pos_y[t]] = np.repeat(self.sup_limit[t], len(sup))
        else:
            inf = np.where(self.data.iloc[:,self.pos_y] < self.inf_limit)[0]
            sup = np.where(self.data.iloc[:,self.pos_y] > self.sup_limit)[0]
            if len(inf)>0:
                self.data.iloc[inf, self.pos_y] = np.repeat(self.inf_limit, len(inf))
            if len(sup)>0:
                self.data.iloc[sup, self.pos_y] = np.repeat(self.sup_limit, len(sup))

    def adapt_horizont(self, onebyone):
        '''
        Move the data sample to connected the y with the x based on the future selected
        onebyone: the movement is to match the future and past moving one step at a time (True) or considering certain lags (False)
        '''
        if self.n_steps == 0:
            self.data = self.data
        else:
            if self.type == 'series':
                X = self.data.drop(self.data.columns[self.pos_y], axis=1)
                y = self.data.iloc[:, self.pos_y]
                y = y.drop(y.index[0], axis=0)
                X = X.drop(X.index[X.shape[0] - 1], axis=0)
                index1 = X.index

                if onebyone[0] == True:
                    y, gap = self.cortes_onebyone(y, len(y), self.n_steps)
                    y = pd.DataFrame(y.transpose())
                    if gap > 0:
                        X = X.drop(X.index[range(X.shape[0] - 1, X.shape[0])], axis=0)
                        index1 = np.delete(index1, range(X.shape[0] - 1, X.shape[0]))
                    X = X.drop(X.index[range(X.shape[0] - self.n_steps + 1, X.shape[0])], axis=0)
                    index1 = np.delete(index1, range(len(index1) - self.n_steps + 1, len(index1)))

                else:
                    y, gap = self.cortes(y, len(y), self.n_steps)
                    y = pd.DataFrame(y.transpose())

                    seq = np.arange(0, X.shape[0] - self.n_steps + 1, self.n_steps)
                    X = X.iloc[seq]
                    index1 = index1[seq]

                    if gap > 0:
                        fuera = 1 + gap + self.n_steps + self.n_lags
                        X = X.drop(X.index[range(X.shape[0] - 1, X.shape[0])], axis=0)
                        index1 = np.delete(index1, range(X.shape[0] - 1, X.shape[0]))

                    else:
                        fuera = 1 + self.n_lags
                    print('El total a quitar de time_val es:', fuera)

                X = X.reset_index(drop=True)

                print('X-shape in adapt_horizont', X.shape)
                print('y-shape in adapt_horizont', y.shape)

                X.index = index1
                y.index = index1

                if self.pos_y == 0:
                    self.data = pd.concat([y, X], axis=1)
                else:
                    self.data = pd.concat([X, y], axis=1)

            else:
                X = self.data.drop(self.data.columns[self.pos_y], axis=1)
                y = self.data.iloc[:, self.pos_y]
                for t in range(self.horizont):
                    y = y.drop(y.index[0], axis=0)
                    X = X.drop(X.index[X.shape[0] - 1], axis=0)

                X = X.reset_index(drop=True)
                X.index = y.index

                if self.pos_y == 0:
                    self.data = pd.concat([y, X.set_index(y.index)], axis=1)
                else:
                    self.data = pd.concat([X.set_index(y.index), y], axis=1)
        print('Horizont adjusted!')

    def scalating(self, scalar_limits, groups, x, y):
        '''
        Scalate date bsed on MinMax scaler and certain limits

        :param scalar_limits: limit of the scalar data
        :param groups: groups defining the different variable groups to be scaled together
        :return: data scaled depending if x, y or both are scaled
        '''
        scalars = dict()
        names = list(groups.keys())
        if x == True and y == True:
            try:
                for i in range(len(groups)):
                    scalars[names[i]] = MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1]))
                    selec = groups[names[i]]
                    d = self.data.iloc[:, selec]
                    if (len(selec) > 1):
                        scalars[names[i]].fit(np.concatenate(np.array(d)).reshape(-1, 1))
                    else:
                        scalars[names[i]].fit(np.array(d).reshape(-1, 1))
                    for z in range(len(selec)):
                        self.data.iloc[:, selec[z]] = scalars[names[i]].transform(pd.DataFrame(d.iloc[:, z]))[:, 0]
            except:
                raise NameError('Problems with the scalar by groups of variables')
            scalar_y = MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1]))
            scalar_y.fit(pd.DataFrame(self.data.iloc[:, self.pos_y]))

            if isinstance(self.pos_y, collections.abc.Sized):
                self.data.iloc[:, self.pos_y] = scalar_y.transform(pd.DataFrame(self.data.iloc[:, self.pos_y]))
            else:
                self.data.iloc[:, self.pos_y] = scalar_y.transform(pd.DataFrame(self.data.iloc[:, self.pos_y]))[:, 0]

            self.scalar_y = scalar_y
            self.scalar_x = scalars
        elif x == True and y == False:
            try:
                for i in range(len(groups)):
                    scalars[names[i]] = MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1]))
                    selec = groups[names[i]]
                    d = self.data.iloc[:, selec]
                    if (len(selec) > 1):
                        scalars[names[i]].fit(np.concatenate(np.array(d)).reshape(-1, 1))
                    else:
                        scalars[names[i]].fit(np.array(d).reshape(-1, 1))
                    for z in range(len(selec)):
                        self.data.iloc[:,selec[z]]= scalars[names[i]].transform(pd.DataFrame(d.iloc[:,z]))[:,0]
                self.scalar_x = scalars
            except:
                raise NameError('Problems with the scalar by groups of variables')
        elif y == True and x == False:
            scalar_y = MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1]))
            scalar_y.fit(pd.DataFrame(self.data.iloc[:, self.pos_y]))

            if isinstance(self.pos_y, collections.abc.Sized):
                self.data.iloc[:, self.pos_y] = scalar_y.transform(pd.DataFrame(self.data.iloc[:, self.pos_y]))
            else:
                self.data.iloc[:, self.pos_y] = scalar_y.transform(pd.DataFrame(self.data.iloc[:, self.pos_y]))[:, 0]

            self.scalar_y = scalar_y
        print('Data scaled!!')

    def missing_values_remove(self):
        self.data = self.data.dropna()
        self.times = self.data.index

    def missing_values_masking_onehot(self):
        d=self.data.drop(self.data.columns[self.pos_y],axis=1)
        places=np.where(d.isnull().any(axis = 1))[0]
        if len(places)<1:
            print('No rows with missing values')
        else:
            binary_var= np.array([1 for x in range(self.data.shape[0])])
            binary_var[places]=0
            binary_var=pd.DataFrame(binary_var,columns=['onehot'])
            self.data = pd.concat([self.data, binary_var.set_index(self.data.index)], axis=1)

            print('Onehot Encoder applied to missing values')

    def missing_values_masking(self):
        self.data = self.data.replace(np.nan, self.mask_value)

        m = self.data.isna().sum()
        if np.array(m > 0).sum() == 0:
            print('Missing values masked')
        else:
            w = np.where(m > 0)
            print('CAREFUL!!! There are still missing values \n'
                  'The variables with missing values are', self.data.columns[w])

    def missing_values_interpolate(self, delete_end, delete_start, mode, limit, order=2):
        '''
        :param delete_end: delete missing data at the last row
        :param delete_start: delete missing data at the first row
        :param mode: linear, spline, polinimial..
        :param limit: amount of missing values accepted
        :param order: if spline or polinomial
        :return: data interpolated
        '''
        if delete_end==True:
            while(any(self.data.iloc[self.data.index[self.data.shape[0]-1]].isna())):
                self.data.drop(self.data.index[self.data.shape[0]-1], axis=0, inplace=True)
        if delete_start==True:
            while(any(self.data.iloc[self.data.index[0]].isna())):
                self.data.drop(self.data.index[0], axis=0, inplace=True)

        if mode=='spline' or mode=='polynomial':
            self.data = self.data.interpolate(method=mode, order=order, axis=0, limit=limit)
        else:
            self.data = self.data.interpolate(method=mode, axis=0, limit=limit)
        m = self.data.isna().sum()
        if np.array(m>0).sum() == 0:
            print('Missing values interpolated')
        else:
            w = np.where(m>0)
            print('CAREFUL!!! There are still missing values; increase the limit? \n'
                  'The variables with missing values are', self.data.columns[w])

        a = self.data.iloc[:,self.pos_y]
        a[a<self.inf_limit] = self.inf_limit
        a[a>self.sup_limit] = self.sup_limit
        self.data.iloc[:,self.pos_y] = a


class LSTM_model(DL):
    def info(self):
        print('Class to built LSTM models.')
        '''
        Subclass of DL 
        repeat_vector:True or False (specific layer). Repeat the inputs n times (batch, 12) -- (batch, n, 12). n would be the timesteps considered as inertia
        dropout: between 0 and 1. regularization technique where randomly selected neurons are ignored during training. They are “dropped out” 
        randomly. This means that their contribution to the activation of downstream neurons is temporally removed.
        type: regression or classification
        weights: weights for the outputs. mainly for multivriate output
        '''

    def __init__(self, data, horizont,scalar_y, scalar_x,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,n_steps,  inf_limit,sup_limit,names,extract_cero, repeat_vector,dropout,weights, type):
        super().__init__(data, horizont,scalar_y, scalar_x,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,  inf_limit,sup_limit,names,extract_cero)
        self.repeat_vector = repeat_vector
        self.dropout = dropout
        self.type = type
        self.weights=weights
        self.n_steps=n_steps

    @staticmethod
    def complex(neurons_lstm, neurons_dense, max_N, max_H):
        '''
        :param max_N: maximun neurons in the network
        :param max_H: maximum hidden layers in the network
        :return: complexity of the model
        '''

        if any(neurons_lstm == 0):
            neurons_lstm = neurons_lstm[neurons_lstm > 0]
        if any(neurons_dense == 0):
            neurons_dense = neurons_dense[neurons_dense > 0]

        u = len(neurons_lstm) + len(neurons_dense)

        F = 0.25 * (u / max_H) + 0.75 * np.sum(np.concatenate((neurons_lstm, neurons_dense))) / max_N

        return F

    @staticmethod
    def three_dimension(data_new, n_inputs):
        '''
        :param data_new: data
        :param n_inputs: the lags considered (n_lags)
        :return: data converted in three dimension based on the lags and the variables
        '''
        data_new =np.array(data_new)
        rest2 = data_new.shape[0] % n_inputs
        ind_out = 0
        while rest2 != 0:
            data_new = np.delete(data_new,0, axis=0)
            rest2 = data_new.shape[0] % n_inputs
            ind_out += 1

        # restructure into windows of  data
        data_new = np.array(np.split(data_new, len(data_new) / n_inputs))
        res={'data':data_new, 'ind_out':ind_out}
        print('Eliminados en three deimensions', ind_out, 'values at the beginning of the sample')
        return res

    @staticmethod
    def split_dataset(data_new1, n_inputs,cut1, cut2):
       '''
        :param data_new1: data
        :param n_inputs:n lags selected to create the blocks lstm
        :param cut1: lower limit to divided into train - test
        :param cut2: upper limit to divided into train - test
        :return: data divided into train - test in three dimensions
        Also returnin the index modified if some index was left over
       '''

       index=data_new1.index
       data_new1=data_new1.reset_index(drop=True)
       data_new = data_new1.copy()

       train, test = data_new.drop(range(cut1, cut2)), data_new.iloc[range(cut1, cut2)]
       index_test = index[range(cut1, cut2)]

       ###################################################################################
       #Evalaute that the datasets match with a number of timesteps sets
       rest1 = train.shape[0] % n_inputs
       ind_out1 = 0
       while rest1 != 0:
           train = train.drop(train.index[0], axis=0)
           rest1 = train.shape[0] % n_inputs
           ind_out1 += 1

       rest2 = test.shape[0] % n_inputs
       ind_out = 0
       while rest2 != 0:
           test = test.drop(test.index[0], axis=0)
           rest2 = test.shape[0] % n_inputs
           ind_out+=1

       ###################################################################################
       if ind_out>0:
            index_test=np.delete(index_test, range(ind_out), axis=0)

       # restructure into windows of  data
       train1 = np.array(np.split(train, len(train) / n_inputs))
       test1 = np.array(np.split(test, len(test) / n_inputs))

       return train1, test1, index_test


    @staticmethod
    def to_supervised(train,pos_y, n_lags,n_steps, horizont, onebyone):
        '''
        Relate x and y based on lags and future horizont

        :param
        train: dataset for transforming
        horizont: horizont to the future selected
        onebyone: [0] if we want to move the sample one by one [1] (True)although the horizont is 0 we want to move th sample lags by lags
        :return: x (past) and y (future horizont) considering the past-future relations selected
        '''

        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        data=pd.DataFrame(data)
        X, y, timesF= list(), list(),list()

        in_start = 0
        # step over the entire history one time step at a time
        if onebyone[0]==True:
            if n_steps==1:
                for _ in range(len(data)-(n_lags + horizont)):
                    #timesF.append(data.index[_ + n_lags-1+horizont])

                    timesF.append(data.index[_ + n_lags+horizont])
                    # define the end of the input sequence
                    in_end = in_start + n_lags
                    out_end=in_end+horizont

                    xx = data.drop(data.columns[pos_y], axis=1)
                    yy = data.iloc[:,pos_y]
                    # ensure we have enough data for this instance
                    if out_end <= len(data):
                        x_input = xx.iloc[in_start:in_end,:]
                        X.append(x_input)
                        if horizont==0:
                            y.append(yy.iloc[out_end-1])
                        else:
                            y.append(yy.iloc[in_end:out_end])
                        #se selecciona uno
                    # move along one time step
                    in_start += 1
                dd=len(data)-len(data)-(n_lags + horizont)+1
            else:
                print('With n_steps > 1 it is not feasible go 1 by 1')
                raise NameError('Problems with n_steps and onebyone')

        else:
            if onebyone[1]==True:
                while in_start <= data.shape[0] - (n_steps - 1) - horizont - n_lags:
                    if n_steps == 1:
                        timesF.append(data.index[in_start + n_lags + horizont-1])
                    else:
                        timesF.append(data.index[(in_start + n_lags + horizont):(in_start + n_lags + horizont + (n_steps - 1))])
                    # define the end of the input sequence
                    in_end = in_start + n_lags
                    out_end = in_end + horizont + (n_steps - 1)

                    xx = data.drop(data.columns[pos_y], axis=1)
                    yy = data.iloc[:, pos_y]
                    # ensure we have enough data for this instance
                    if out_end <= len(data):
                        x_input = xx.iloc[in_start:in_end, :]
                        X.append(x_input)
                        if horizont == 0 and n_steps == 1:
                            y.append(yy.iloc[out_end - 1])
                        elif horizont == 0 and n_steps > 1:
                            y.append(yy.iloc[range(out_end - 1, out_end - 1 + (n_steps - 1))])
                        else:
                            y.append(yy.iloc[in_end:out_end])
                    # se selecciona uno
                    # move along one time step
                    in_start += n_lags
                    dd = len(data) - len(data) - (n_lags + horizont) + 1

            else:
                while in_start <= data.shape[0] - (n_steps - 1) - horizont - n_lags:
                    if n_steps == 1:
                        timesF.append(data.index[in_start + n_lags + horizont-1])
                    else:
                        timesF.append(
                            data.index[(in_start + n_lags + horizont):(in_start + n_lags + horizont + (n_steps - 1))])
                    # define the end of the input sequence
                    in_end = in_start + n_lags
                    out_end = in_end + horizont + (n_steps - 1)

                    xx = data.drop(data.columns[pos_y], axis=1)
                    yy = data.iloc[:, pos_y]
                    # ensure we have enough data for this instance
                    if out_end <= len(data):
                        x_input = xx.iloc[in_start:in_end, :]
                        X.append(x_input)
                        if horizont == 0 and n_steps == 1:
                            y.append(yy.iloc[out_end - 1])
                        elif horizont == 0 and n_steps > 1:
                            y.append(yy.iloc[range(out_end - 1, out_end - 1 + (n_steps - 1))])
                        else:
                            y.append(yy.iloc[in_end:out_end])
                    # se selecciona uno
                    # move along one time step
                    in_start += n_steps
                    dd = len(data) - len(data) - (n_lags + horizont) + 1
                print('Data supervised')

        return(np.array(X), np.array(y),timesF, dd)

    @staticmethod
    def built_model_classification(train_x1, train_y1, neurons_lstm, neurons_dense, mask, mask_value, repeat_vector, dropout):
        '''
        WORK IN PROGRESS!!!

        :param mask: True or False
        :param repeat_vector: True or False
        :return: the model architecture built to be trained
        '''

        print('No finished')


        layers_lstm = len(neurons_lstm)
        layers_neurons = len(neurons_dense)

        n_timesteps, n_features, n_outputs = train_x1.shape[1], train_x1.shape[2], train_y1.shape[1]  # define model

        model = Sequential()
        for k in range(layers_lstm):
            if k == 0 and repeat_vector == True:
                if mask == True:
                    model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                    model.add(LSTM(neurons_lstm[k], activation='relu'))
                    model.add(RepeatVector(n_timesteps))
                else:
                    model.add(LSTM(neurons_lstm[k], input_shape=(n_timesteps, n_features), activation='relu'))
                    model.add(RepeatVector(n_timesteps))

            elif k == 0 and repeat_vector == False:
                if mask == True:
                    model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                    model.add(LSTM(neurons_lstm[k], return_sequences=True, activation='relu'))
                else:
                    model.add(LSTM(neurons_lstm[k], input_shape=(n_timesteps, n_features), return_sequences=True,
                                   activation='relu'))
            elif k == layers_lstm - 1:
                model.add(LSTM(neurons_lstm[k], activation='relu'))
            else:
                model.add(LSTM(neurons_lstm[k], return_sequences=True, activation='relu'))

        if layers_neurons > 0:
            if dropout > 0:
                for z in range(layers_neurons):
                    if neurons_dense[z] == 0:
                        pass
                    else:
                        model.add(Dense(neurons_dense[z], activation='relu', kernel_constraint=maxnorm(3)))
                        model.add(Dropout(dropout))
            else:
                for z in range(layers_neurons):
                    if neurons_dense[z] == 0:
                        pass
                    else:
                        model.add(Dense(neurons_dense[z], activation='relu'))

        model.add(Dense(n_outputs), kernel_initializer='normal', activation='softmax')
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        model.summary()

        return model


    @staticmethod
    def built_model_regression(train_x1, train_y1, neurons_lstm, neurons_dense, mask,mask_value, repeat_vector,dropout):
        '''
        :param
        trains: datasets
        neurons_lstm: array with the LSTM neurons that define the LSTM layers
        neurons_dense: array with the dense neurons that define the dense layers
        mask: True or False
        repeat_vector: True or False
        dropout: between 0 and 1
        :return: the model architecture built to be trained

        CAREFUL: with masking at least two lstm layers are required
        '''

        if any(neurons_lstm==0):
            neurons_lstm = neurons_lstm[neurons_lstm>0]
        if any(neurons_dense==0):
            neurons_dense = neurons_dense[neurons_dense>0]

        layers_lstm = len(neurons_lstm)
        layers_neurons = len(neurons_dense)
        n_timesteps, n_features, n_outputs = train_x1.shape[1], train_x1.shape[2], train_y1.shape[1]

        model = Sequential()

        if layers_lstm<2:
            if mask == True:
                model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                model.add(LSTM(neurons_lstm[0], activation='relu'))
            else:
                model.add(LSTM(neurons_lstm[0], input_shape=(n_timesteps, n_features),
                               activation='relu'))
        else:
            for k in range(layers_lstm):
                if k==0 and repeat_vector==True:
                    if mask==True:
                        model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                        model.add(LSTM(neurons_lstm[k],activation='relu'))
                        model.add(RepeatVector(n_timesteps))
                    else:
                        model.add(LSTM(neurons_lstm[k], input_shape=(n_timesteps, n_features),activation='relu'))
                        model.add(RepeatVector(n_timesteps))

                elif k==0 and repeat_vector==False:
                    if mask == True:
                        model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                        model.add(LSTM(neurons_lstm[k], return_sequences=True,activation='relu'))
                    else:
                        model.add(LSTM(neurons_lstm[k], input_shape=(n_timesteps, n_features), return_sequences=True,activation='relu'))
                elif k==layers_lstm-1:
                    model.add(LSTM(neurons_lstm[k],activation='relu'))
                else:
                    model.add(LSTM(neurons_lstm[k],return_sequences=True,activation='relu'))

        if layers_neurons>0:
            if dropout>0:
                for z in range(layers_neurons):
                    if neurons_dense[z]==0:
                        pass
                    else:
                        model.add(Dense(neurons_dense[z], activation='relu', kernel_constraint=maxnorm(3)))
                        model.add(Dropout(dropout))
            else:
                for z in range(layers_neurons):
                    if neurons_dense[z]==0:
                        pass
                    else:
                        model.add(Dense(neurons_dense[z], activation='relu'))

        model.add(Dense(n_outputs,kernel_initializer='normal', activation='linear'))

        model.compile(loss='mse', optimizer='adam',metrics=['mse'])
        model.summary()

        return model

    @staticmethod
    def train_model(model,train_x1, train_y1, test_x1, test_y1, pacience, batch):
        '''
        :param
        train: train data set
        test: test data set
        pacience: stopping criterion
        bath: batchsize
        model: model architecture built
        :return: model trained based on pacience
        '''
        print('SHAPE TRAIN:', train_x1.shape)
        print('SHAPE TRAIN_Y:',train_y1.shape)
        print('SHAPE TEST:', test_x1.shape)
        print('SHAPE TEST_Y:', test_y1.shape)

        h_path = Path('./best_models')
        h_path.mkdir(exist_ok=True)
        h = h_path / f'best_{random.randint(0, 1000000)}_model.h5'

        # Checkpoitn callback
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pacience)
        mc = ModelCheckpoint(str(h), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # Train the model
        history = model.fit(train_x1, train_y1, epochs=2000, validation_data=(test_x1, test_y1), batch_size=batch,
                           callbacks=[es, mc])

        return model, history


    @staticmethod
    def predict_model(model,n_lags, x_val,batch, n_outputs):
        '''
        :param model: trained model
        :param n_lags: lags to built lstm block
        :param n_outputs: how many variables want to estimate
        :return: predictions in the validation sample, considering the selected moving window
        CAREFUL: the predictions here taking into account movements by n_lags
        '''

        data = np.array(x_val)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        predictions = list()
        l1 = 0
        l2 = n_lags
        for i in range(x_val.shape[0]):
            # flatten data
            input_x = data[l1:l2, :]
            input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
            print(input_x.shape)
            # forecast the next step
            yhat = model.predict(input_x, verbose=0, batch_size=batch)
            if n_outputs>1:
                yhat=yhat
            else:
                yhat = yhat[0]
            predictions.append(yhat)
            #history.append(tt[i,:])
            l1 =l2
            l2 += n_lags

        predictions  =np.array(predictions)
        if n_outputs>1:
            y_pred=predictions.reshape((predictions.shape[0], predictions.shape[2]))
        else:
            y_pred = predictions.reshape((predictions.shape[0] * predictions.shape[1], 1))
        res = {'y_pred': y_pred}
        return res

    @staticmethod
    def cv_division_lstm(data, horizont, fold, pos_y,n_lags,n_steps, onebyone, values):
        '''
            MODIFICATION TO DIVIDE INTO MORE PIECES????

        :return: Division to cv analysis considering that with lstm algorithm the data can not be divided into simple pieces.
        The validation sample is extracted from the first test sample
        If values the division is previously defined
        It can only be divided with a initial part and a final part
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        return: train, test, and validation samples. Indexes for test samples (before division into test-validation)
        '''
        X_test = []
        X_train = []
        X_val = []
        Y_test = []
        Y_train = []
        Y_val = []

        if values:
            times_val = []
            place = values[2]
            var= data.iloc[:,place]
            for t in range(values[0]):
                if len(place)==1:
                    w = np.where(var==values[1])[0][0]
                    w2 = np.where(var==values[1])[0][len(np.where(var==values[1])[0])-1]
                elif len(place)==2:
                    w = np.where((var.iloc[:,0]==values[1][t][0])&(var.iloc[:,1]==values[1][t][1]))[0][0]
                    w2 = np.where((var.iloc[:,0]==values[1][t][0])&(var.iloc[:,1]==values[1][t][1]))[0][len(np.where((var.iloc[:,0]==values[1][t][0])&(var.iloc[:,1]==values[1][t][1]))[0])-1]
                elif len(place)==3:
                    w = np.where((var.iloc[:, 0] == values[1][t][0]) & (var.iloc[:, 1] == values[1][t][1])&(var.iloc[:, 2] == values[1][t][2]))[0][0]
                    w2 = np.where((var.iloc[:, 0] == values[1][t][0]) & (var.iloc[:, 1] == values[1][t][1])&(var.iloc[:, 2] == values[1][t][2]))[0][
                        len(np.where((var.iloc[:, 0] == values[1][t][0]) & (var.iloc[:, 1] == values[1][t][1])&(var.iloc[:, 2] == values[1][t][2]))[0]) - 1]
                else:
                    raise(NameError('Not considered'))

                train, val, index_val = LSTM_model.split_dataset(data, n_lags, w, w2)

                index_val = index_val[range(index_val.shape[0] - math.ceil(index_val.shape[0] / 2))]
                st = int(train.shape[0]/3)
                test= train[range(train.shape[0]-st, train.shape[0]),:,:]
                train=np.delete(train, list(range(train.shape[0]-st, train.shape[0])), 0)

                x_train, y_train, ind_train, dif = LSTM_model.to_supervised(train, pos_y, n_lags,n_steps, horizont, onebyone)
                x_test, y_test, ind_test, dif = LSTM_model.to_supervised(test, pos_y, n_lags,n_steps, horizont, onebyone)
                x_val, y_val, ind_val, dif = LSTM_model.to_supervised(val, pos_y, n_lags,n_steps, horizont, onebyone)

                if onebyone[0] == True:
                    if horizont == 0:
                        index_val = np.delete(index_val, range(n_lags - 1), axis=0)
                    else:
                        index_val = np.delete(index_val, range(n_lags), axis=0)
                else:
                    index_val = ind_val

                times_val.append(index_val)

                print(len(index_val))
                print(y_val.shape)
                X_test.append(x_test)
                X_train.append(x_train)
                X_val.append(x_val)
                Y_test.append(y_test)
                Y_train.append(y_train)
                Y_val.append(y_val)
                print('cv_division done')

        else:

            ###################################################################################
            step = int(data.shape[0] / fold)
            w = 0
            w2 = step
            times_val = []
            ####################################################################################

            try:
               for i in range(2):
                    train, test, index_val = LSTM_model.split_dataset(data, n_lags,w, w2)

                    print('####',test.shape)
                    print(len(index_val))

                    r = LSTM_model.three_dimension(index_val, n_lags)
                    index_val=r['data']

                    print(index_val.shape)
                    index_val = index_val[range(test.shape[0]-math.ceil(test.shape[0]/2), test.shape[0]),:]
                    val = test[range(test.shape[0]-math.ceil(test.shape[0]/2), test.shape[0]),:,:]
                    test = test[range(0, math.ceil(test.shape[0] / 2)), :, :]

                    print(len(index_val))
                    print(val.shape)
                    index_val= index_val.reshape(index_val.shape[0]*index_val.shape[1],1)

                    x_train, y_train,ind_train,dif = LSTM_model.to_supervised(train, pos_y, n_lags,n_steps,horizont, onebyone)
                    x_test, y_test,ind_test,dif = LSTM_model.to_supervised(test, pos_y, n_lags,n_steps,horizont,onebyone)
                    x_val, y_val,ind_val,dif = LSTM_model.to_supervised(val, pos_y, n_lags,n_steps,horizont, onebyone)


                    if onebyone[0]==True:
                        index_val = np.delete(index_val, range(n_lags+horizont), axis=0)
                    else:
                        if isinstance(ind_val, list):
                            index_val = index_val[np.concatenate(ind_val)]
                        else:
                            index_val=index_val[ind_val]

                    times_val.append(index_val)

                    print(len(index_val))
                    print(y_val.shape)
                    X_test.append(x_test)
                    X_train.append(x_train)
                    X_val.append(x_val)
                    Y_test.append(y_test)
                    Y_train.append(y_train)
                    Y_val.append(y_val)

                    w = data.shape[0]-w2
                    w2 = data.shape[0]
            except:
                raise NameError('Problems with the sample division in the cv classic')
        res = {'x_test': X_test, 'x_train': X_train, 'x_val':X_val, 'y_test': Y_test, 'y_train': Y_train, 'y_val':Y_val,  'time_val':times_val}
        return res

    def cv_analysis(self, fold,rep, neurons_lstm, neurons_dense,onebyone, pacience, batch,mean_y,values,plot, q=[], model=[]):
        '''
        :param fold: the assumed size of divisions
        :param rep: In this case, the analysis repetitions of each of the two possile division considered in lstm analysis
        :param onebyone: [0] if we want to move the sample one by one [1] (True)although the horizont is 0 we want to move th sample lags by lags
        :param values specific values to divide the sample. specific values of a variable to search division
        :param plot: True plots
        :param q: queue that inform us if paralyse or not
        :param model if model we have a pretrained model

        if mean_y is size 0 the evaluation will be with variation rates

        :return: Considering what zero_problem is mentioned, return thre predictions, real values, errors and computational times needed to train the models
        '''

        names = self.names
        names = np.delete(names ,self.pos_y)
        layers_lstm = len(neurons_lstm)
        layers_neurons = len(neurons_dense)

        res = LSTM_model.cv_division_lstm(self.data, self.horizont, fold, self.pos_y, self.n_lags,self.n_steps, onebyone,values)

        x_test =np.array(res['x_test'])
        x_train=np.array(res['x_train'])
        x_val=np.array(res['x_val'])
        y_test=np.array(res['y_test'])
        y_train =np.array(res['y_train'])
        y_val =np.array(res['y_val'])

        times_val = res['time_val']

        print('Shape dates at beginning of cv_analysis',times_val[0].shape)
        print('Shape y at beginning of cv_analysis', y_val[0].shape)

        if self.type=='regression':
            if isinstance(model, list):
                if len(y_train[0].shape)>1:
                    y_trainO=y_train[0]
                else:
                    y_trainO=y_train[0].reshape(-1, 1)

                model1 = self.__class__.built_model_regression(x_train[0],y_trainO,neurons_lstm, neurons_dense, self.mask,self.mask_value, self.repeat_vector, self.dropout)

            else:
                model1=model
            # Train the model
            times = [0 for x in range(rep*2)]
            cv = [0 for x in range(rep*2)]
            rmse = [0 for x in range(rep*2)]
            nmbe = [0 for x in range(rep*2)]
            zz= 0
            predictions = []
            reales = []
            for z in range(len(x_train)):
                print('Fold number', z)
                for zz2 in range(rep):
                    modelF = model1
                    time_start = time()
                    modelF, history = self.__class__.train_model(modelF,x_train[z], y_train[z], x_test[z], y_test[z], pacience, batch)
                    times[zz] = round(time() - time_start, 3)

                    res = self.__class__.predict_model(modelF, self.n_lags, x_val[z], batch, len(self.pos_y))
                    y_pred = res['y_pred']

                    y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
                    for t in range(y_pred.shape[1]):
                        inf = np.where(y_pred[:,t] < self.inf_limit[t])[0]
                        upp = np.where(y_pred[:,t] > self.sup_limit[t])[0]
                        if len(inf) > 0:
                            y_pred[inf, t] = self.inf_limit[t]
                        if len(upp) > 0:
                            y_pred[upp, t] = self.sup_limit[t]

                    if len(y_val[z].shape)>1:
                        y_real = y_val[z]
                    else:
                        y_real = y_val[z].reshape((y_val[z].shape[0] * y_val[z].shape[1], 1))
                    y_real = np.array(self.scalar_y.inverse_transform(y_real))

                    if plot == True:

                        if self.horizont>1:
                            y_predP = y_pred.reshape(int(y_pred.shape[0] / self.horizont), self.horizont)
                            y_realP = y_real.reshape(int(y_real.shape[0] / self.horizont), self.horizont)

                            y_predP = pd.DataFrame(y_predP[:,0])
                            y_predP.index = times_val[z]
                            y_realP = pd.DataFrame(y_realP[:,0])
                            y_realP.index = times_val[z]

                            s = np.max(y_realP).astype(int) + 12
                            i = np.min(y_realP).astype(int) - 12
                            plt.figure()
                            plt.ylim(i, s)
                            plt.plot(y_realP, color='black', label='Real')
                            plt.plot(y_predP, color='blue', label='Prediction')
                            plt.legend()
                            plt.title("Subsample {} ".format(z))
                            plt.show()
                        else:
                            y_realP = pd.DataFrame(y_real)
                            y_predP = pd.DataFrame(y_pred)
                            y_realP.index=times_val[z]
                            y_predP.index=times_val[z]

                            s = np.max(y_real).astype(int) + 12
                            i = np.min(y_real).astype(int) - 12
                            plt.figure()
                            plt.ylim(i, s)
                            plt.plot(y_realP, color='black', label='Real')
                            plt.plot(y_predP, color='blue', label='Prediction')
                            plt.legend()
                            plt.title("Subsample {} ".format(z))
                            plt.show()

                    y_predF = y_pred.copy()
                    y_predF = pd.DataFrame(y_predF)
                    y_realF = y_real.copy()
                    y_realF = pd.DataFrame(y_realF)

                    if self.zero_problem == 'schedule':
                        print('*****Night-schedule fixed******')

                        res = super().fix_values_0(times_val[z][:,0],
                                                      self.zero_problem, self.limits)


                        index_hour = res['indexes_out']

                        predictions.append(y_predF)
                        reales.append(y_realF)
                        if len(y_pred) <= 1:
                            y_pred1 = np.nan
                            y_real1 = y_real
                        else:
                            if len(index_hour) > 0 and self.horizont == 0:
                                y_pred1 = np.delete(y_pred, index_hour, 0)
                                y_real1 = np.delete(y_real, index_hour, 0)
                            elif len(index_hour) > 0 and self.horizont > 0:
                                y_pred1 = np.delete(y_pred, index_hour - self.horizont, 0)
                                y_real1 = np.delete(y_real, index_hour - self.horizont, 0)
                            else:
                                y_pred1 = y_pred
                                y_real1 = y_real

                        # Outliers and missing values
                        if self.mask == True and len(y_pred1) > 0:
                            if mean_y.size == 0:
                                o = np.where(y_real1 < self.inf_limit)[0]
                                if len(o) > 0:
                                    y_pred1 = np.delete(y_pred1, o, 0)
                                    y_real1 = np.delete(y_real1, o, 0)
                            else:
                                o=list()
                                for t in range(len(mean_y)):
                                    o.append(np.where(y_real1[:,t] < self.inf_limit[t])[0])

                                oT=np.unique(np.concatenate(o))
                                y_pred1 = np.delete(y_pred1, oT, 0)
                                y_real1 = np.delete(y_real1, oT, 0)

                        if self.extract_cero == True and len(y_pred1) > 0:
                            if mean_y.size == 0:
                                o = np.where(y_real1 == 0)[0]
                                if len(o) > 0:
                                    y_pred1 = np.delete(y_pred1, o, 0)
                                    y_real1 = np.delete(y_real1, o, 0)
                            else:
                                o = list()
                                for t in range(len(mean_y)):
                                    o.append(np.where(y_real1[:, t] == 0)[0])

                                oT = np.unique(np.concatenate(o))
                                y_pred1 = np.delete(y_pred1, oT, 0)
                                y_real1 = np.delete(y_real1, oT, 0)

                        #After  checking we have data to evaluate:
                        # if the mean_y is empty we use variation rate with or witout weights
                        # on the other hand, we compute the classic error metrics

                        if len(y_pred1) > 0:
                            if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0 and len(y_pred1)>0 and len(y_real1)>0:
                                if mean_y.size == 0:
                                    e = evals(y_pred1, y_real1).variation_rate()
                                    if isinstance(self.weights, list):
                                        cv[zz] = np.mean(e)
                                    else:
                                        print(e)
                                        print(self.weights)
                                        cv[zz] = np.sum(e * self.weights)
                                    rmse[zz] = np.nan
                                    nmbe[zz] = np.nan
                                else:
                                    e_cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                                    e_r = evals(y_pred1, y_real1).rmse()
                                    e_n = evals(y_pred1, y_real1).nmbe(mean_y)
                                    if isinstance(self.weights, list):
                                        cv[zz] = np.mean(e_cv)
                                        rmse[zz] = np.mean(e_r)
                                        nmbe[zz] = np.mean(e_n)
                                    else:
                                        cv[zz] = np.sum(e_cv * self.weights)
                                        rmse[zz] = np.sum(e_r * self.weights)
                                        nmbe[zz] = np.sum(e_n * self.weights)
                            else:
                                print('Missing values are detected when we are evaluating the predictions')
                                cv[zz] = 9999
                                rmse[zz] = 9999
                                nmbe[zz] = 9999
                        else:
                            raise NameError('Empty prediction')

                    elif self.zero_problem == 'radiation':
                        print('*****Night-radiation fixed******')
                        place = np.where(names == 'radiation')[0]
                        scalar_x = self.scalar_x
                        scalar_rad = scalar_x['radiation']
                        res = super().fix_values_0(scalar_rad.inverse_transform(x_val[zz][:, x_val[zz].shape[1] - 1, place]),
                                                   self.zero_problem, self.limits)
                        index_rad = res['indexes_out']
                        index_rad2 = np.where(y_real <= self.inf_limit)[0]
                        index_rad = np.union1d(np.array(index_rad), np.array(index_rad2))

                        predictions.append(y_predF)
                        reales.append(y_realF)

                        if len(y_pred) <= 1:
                            y_pred1 = np.nan
                            y_real1 = y_real
                        else:
                            if len(index_rad) > 0 and self.horizont == 0:
                                y_pred1 = np.delete(y_pred, index_rad, 0)
                                y_real1 = np.delete(y_real, index_rad, 0)
                            elif len(index_rad) > 0 and self.horizont > 0:
                                y_pred1 = np.delete(y_pred, np.array(index_rad) - self.horizont, 0)
                                y_real1 = np.delete(y_real, np.array(index_rad) - self.horizont, 0)
                            else:
                                y_pred1 = y_pred
                                y_real1 = y_real

                        # Outliers and missing values
                        if self.mask == True and len(y_pred1) > 0:
                            if mean_y.size == 0:
                                o = np.where(y_real1 < self.inf_limit)[0]
                                if len(o) > 0:
                                    y_pred1 = np.delete(y_pred1, o, 0)
                                    y_real1 = np.delete(y_real1, o, 0)
                            else:
                                o = list()
                                for t in range(len(mean_y)):
                                    o.append(np.where(y_real1[:, t] < self.inf_limit[t])[0])

                                oT = np.unique(np.concatenate(o))
                                y_pred1 = np.delete(y_pred1, oT, 0)
                                y_real1 = np.delete(y_real1, oT, 0)

                        if self.extract_cero == True and len(y_pred1) > 0:
                            if mean_y.size == 0:
                                o = np.where(y_real1 == 0)[0]
                                if len(o) > 0:
                                    y_pred1 = np.delete(y_pred1, o, 0)
                                    y_real1 = np.delete(y_real1, o, 0)
                            else:
                                o = list()
                                for t in range(len(mean_y)):
                                    o.append(np.where(y_real1[:, t] == 0)[0])

                                oT = np.unique(np.concatenate(o))
                                y_pred1 = np.delete(y_pred1, oT, 0)
                                y_real1 = np.delete(y_real1, oT, 0)

                        #After  checking we have data to evaluate:
                        # if the mean_y is empty we use variation rate with or witout weights
                        # on the other hand, we compute the classic error metrics

                        if len(y_pred1) > 0:
                            if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0 and len(y_pred1)>0 and len(y_real1)>0:
                                if mean_y.size == 0:
                                    e = evals(y_pred1, y_real1).variation_rate()
                                    if isinstance(self.weights, list):
                                        cv[zz] = np.mean(e)
                                    else:
                                        cv[zz] = np.sum(e * self.weights)
                                    rmse[zz] = np.nan
                                    nmbe[zz] = np.nan
                                else:
                                    e_cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                                    e_r = evals(y_pred1, y_real1).rmse()
                                    e_n = evals(y_pred1, y_real1).nmbe(mean_y)
                                    if isinstance(self.weights, list):
                                        cv[zz] = np.mean(e_cv)
                                        rmse[zz] = np.mean(e_r)
                                        nmbe[zz] = np.mean(e_n)
                                    else:
                                        cv[zz] = np.sum(e_cv * self.weights)
                                        rmse[zz] = np.sum(e_r * self.weights)
                                        nmbe[zz] = np.sum(e_n * self.weights)
                            else:
                                print('Missing values are detected when we are evaluating the predictions')
                                cv[zz] = 9999
                                rmse[zz] = 9999
                                nmbe[zz] = 9999
                        else:
                            raise NameError('Empty prediction')
                    else:
                        predictions.append(y_predF)
                        reales.append(y_realF)

                        # Outliers and missing values
                        if self.mask == True and len(y_pred) > 0:
                            if mean_y.size == 0:
                                o = np.where(y_real < self.inf_limit)[0]
                                if len(o) > 0:
                                    y_pred = np.delete(y_pred, o, 0)
                                    y_real = np.delete(y_real, o, 0)
                            else:
                                o = list()
                                for t in range(len(mean_y)):
                                    o.append(np.where(y_real[:, t] < self.inf_limit[t])[0])

                                oT = np.unique(np.concatenate(o))
                                y_pred = np.delete(y_pred, oT, 0)
                                y_real = np.delete(y_real, oT, 0)

                        if self.extract_cero == True and len(y_pred) > 0:
                            if mean_y.size == 0:
                                o = np.where(y_real == 0)[0]
                                if len(o) > 0:
                                    y_pred = np.delete(y_pred, o, 0)
                                    y_real = np.delete(y_real, o, 0)
                            else:
                                o = list()
                                for t in range(len(mean_y)):
                                    o.append(np.where(y_real[:, t] == 0)[0])

                                oT = np.unique(np.concatenate(o))
                                y_pred = np.delete(y_pred, oT, 0)
                                y_real = np.delete(y_real, oT, 0)

                        #After  checking we have data to evaluate:
                        # if the mean_y is empty we use variation rate with or witout weights
                        # on the other hand, we compute the classic error metrics

                        if len(y_pred) > 0:
                            if np.sum(np.isnan(y_pred)) == 0 and np.sum(np.isnan(y_real)) == 0 and len(y_pred)>0 and len(y_real)>0:
                                if mean_y.size == 0:
                                    e = evals(y_pred, y_real).variation_rate()
                                    if isinstance(self.weights, list):
                                        cv[zz] = np.mean(e)
                                    else:
                                        print(e)
                                        print(self.weights)
                                        cv[zz] = np.sum(e * self.weights)
                                    rmse[zz]=np.nan
                                    nmbe[zz]=np.nan
                                else:
                                    e_cv = evals(y_pred, y_real).cv_rmse(mean_y)
                                    e_r = evals(y_pred, y_real).rmse()
                                    e_n = evals(y_pred, y_real).nmbe(mean_y)
                                    if isinstance(self.weights, list):
                                        cv[zz] = np.mean(e_cv)
                                        rmse[zz] = np.mean(e_r)
                                        nmbe[zz] = np.mean(e_n)
                                    else:
                                        cv[zz] = np.sum(e_cv * self.weights)
                                        rmse[zz] = np.sum(e_r * self.weights)
                                        nmbe[zz] = np.sum(e_n * self.weights)
                            else:
                                print('Missing values are detected when we are evaluating the predictions')
                                cv[zz] = 9999
                                rmse[zz] = 9999
                                nmbe[zz] = 9999
                        else:
                            raise NameError('Empty prediction')
                    zz +=1

            res_final = {'preds': predictions, 'reals':reales, 'times_val':times_val, 'cv_rmse':cv,
                 'nmbe':nmbe, 'rmse':rmse,
                 'times_comp':times}

            print('The model with {} LSTM layers, {} Dense layers, {} LSTM neurons, {} Dense neurons and a patience of {} \n'
                  'has an Average CV(RMSE): {}, has an Average NMBE: {} \n, an Average RMSE: {} and the average time training is {} s'.format(layers_lstm,layers_neurons,neurons_lstm,neurons_dense,
            pacience, np.nanmean(cv),np.nanmean(nmbe),np.nanmean(rmse),np.mean(times)))

            z = Queue()
            if type(q) == type(z):
                #q.put(np.array([np.mean(cv), np.std(cv)]))
                q.put(np.array([np.mean(cv), self.complex(layers_lstm,layers_neurons,2000,12)]))
            else:
                return (res_final)

    ###################################################################################################
        #FALTARÍA CLASIFICACION !!!!!!!!!!!!!!!!!
        ###################################################################################################


    def train(self, train, test, neurons_lstm, neurons_dense, pacience, batch, save_model,onebyone,model=[]):
        '''
        onebyone: [0] if we want to move the sample one by one [1] (True)although the horizont is 0 we want to move th sample lags by lags
        if model we have a pretrained model
        Instance to train model outside these classes
        :return: the trained model and the time required to be trained
        '''

        now = str(datetime.now().microsecond)

        res = self.__class__.three_dimension(train, self.n_lags)
        train = res['data']
        res = self.__class__.three_dimension(test, self.n_lags)
        test = res['data']

        print('Data in three dimensions')

        x_test, y_test, ind_test, dif = LSTM_model.to_supervised(test, self.pos_y, self.n_lags,self.n_steps, self.horizont,
                                                                     onebyone)
        x_train, y_train, ind_train, dif = LSTM_model.to_supervised(train, self.pos_y, self.n_lags,self.n_steps, self.horizont,
                                                                        onebyone)

        print('X_train SHAPE in training', x_train.shape)
        print('X_test SHAPE in training', x_test.shape)
        print('Y SHAPE in training',y_train.shape)
        y_train=pd.DataFrame(y_train)
        if isinstance(model, list):
            if self.type=='regression':
                model = self.__class__.built_model_regression(x_train, y_train,neurons_lstm, neurons_dense, self.mask, self.mask_value, self.repeat_vector, self.dropout)
                time_start = time()
                model_trained, history = self.__class__.train_model(model, x_train, y_train, x_test, y_test, pacience, batch)
                times = round(time() - time_start, 3)
            else:
                model = self.__class__.built_model_classification(x_train, y_train,neurons_lstm, neurons_dense,self.mask, self.mask_value, self.repeat_vector, self.dropout)
                time_start = time()
                model_trained, history = self.__class__.train_model(model, x_train, y_train, x_test, y_test, pacience, batch)
                times = round(time() - time_start, 3)
        else:
            time_start = time()
            model_trained, history = self.__class__.train_model(model, x_train, y_train, x_test, y_test, pacience,batch)
            times = round(time() - time_start, 3)

        if save_model==True:
            name='mlp'+now+'.h5'
            model_trained.save(name, save_format='h5')
        res = {'model': model_trained, 'times': times, 'history':history}
        return res


    def predict(self, model, val,mean_y,batch,times, onebyone, scalated,daily,plotting):
        '''
        :param model: trained model
        times: dates for plot
        :param scalated: 0 prediction sample 1 real sample
        daily: option to generate results day by day
        :param onebyone: [0] if we want to move the sample one by one [1] (True)although the horizont is 0 we want to move th sample lags by lags
        :return: prediction with the built metrics
        Instance to predict certain samples outside these classes
        '''
        shape1 = val.shape[0]
        res = self.__class__.three_dimension(val, self.n_lags)

        val = res['data']
        i_out = res['ind_out']
        if i_out>0:
            times=np.delete(times, range(i_out),0)


        if self.horizont == 0 and onebyone[1] == True:
            seq=list()
            cont = -1+ self.n_lags+self.horizont
            while cont <= len(times) - self.horizont:
                if self.n_steps == 1:
                    seq.append(times[cont + (self.n_steps - 1)])
                else:
                    seq.append(times[range(cont, cont + (self.n_steps - 1) + 1)])
                cont += self.n_lags
            times = seq
        elif self.horizont == 0 and onebyone[1] == False:
            times = np.delete(times, range(self.n_lags), 0)
        else:
            times = np.delete(times, range(self.n_lags), 0)

        x_val, y_val,ind_val,dif = self.__class__.to_supervised(val, self.pos_y, self.n_lags,self.n_steps, self.horizont, onebyone)

        print('Diferencia entre time and y:',dif)

        print('X_val SHAPE in predicting',x_val.shape)
        print('Y_val SHAPE',y_val.shape)

        if isinstance(self.pos_y, collections.abc.Sized):
            outputs = len(self.pos_y)
        else:
            outputs = 1

        res = self.__class__.predict_model(model, self.n_lags,  x_val,batch, outputs)

        y_pred = res['y_pred']
        print('SHAPE of y_pred in predicting',y_pred.shape)

        if scalated[0]==True:
            y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
        if scalated[1]==True:
            if len(y_val.shape)>1:
                y_val = np.array(self.scalar_y.inverse_transform(y_val))
            else:
                y_val=np.array(self.scalar_y.inverse_transform(y_val.reshape(-1,1)))

        if isinstance(self.pos_y, collections.abc.Sized):
            for t in range(len(self.pos_y)):
                y_pred[np.where(y_pred[:,t] < self.inf_limit[t])[0],t] = self.inf_limit[t]
                y_pred[np.where(y_pred[:,t] > self.sup_limit[t])[0], t] = self.sup_limit[t]
            y_real=y_val
        else:
            y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
            y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit
            y_real = y_val.reshape((len(y_val), 1))

        print('PREDICTION:', y_pred)

        y_predF = y_pred.copy()
        y_predF = pd.DataFrame(y_predF)
        y_predF.index = times
        y_realF = pd.DataFrame(y_real.copy())
        y_realF.index = y_predF.index

        if self.zero_problem == 'schedule':
            print('*****Night-schedule fixed******')

            res = super().fix_values_0(times,
                                       self.zero_problem, self.limits)

            index_hour = res['indexes_out']

            if len(y_pred) <= 1:
                y_pred1 = np.nan
                y_real1 = y_real
            else:

                if len(index_hour) > 0 and self.horizont == 0:
                    y_pred1 = np.delete(y_pred, index_hour, 0)
                    y_real1 = np.delete(y_real, index_hour, 0)
                elif len(index_hour) > 0 and self.horizont > 0:
                    y_pred1 = np.delete(y_pred, index_hour - self.horizont, 0)
                    y_real1 = np.delete(y_real, index_hour - self.horizont, 0)
                else:
                    y_pred1 = y_pred
                    y_real1 = y_real
            # Outliers and missing values
            if self.mask == True and len(y_pred1) > 0:
                if mean_y.size == 0:
                    o = np.where(y_real1 < self.inf_limit)[0]
                    if len(o) > 0:
                        y_pred1 = np.delete(y_pred1, o, 0)
                        y_real1 = np.delete(y_real1, o, 0)
                else:
                    o = list()
                    for t in range(len(mean_y)):
                        o.append(np.where(y_real1[:, t] < self.inf_limit[t])[0])

                    oT = np.unique(np.concatenate(o))
                    y_pred1 = np.delete(y_pred1, oT, 0)
                    y_real1 = np.delete(y_real1, oT, 0)

            if self.extract_cero == True and len(y_pred1) > 0:
                if mean_y.size == 0:
                    o = np.where(y_real1 == 0)[0]
                    if len(o) > 0:
                        y_pred1 = np.delete(y_pred1, o, 0)
                        y_real1 = np.delete(y_real1, o, 0)
                else:
                    o = list()
                    for t in range(len(mean_y)):
                        o.append(np.where(y_real1[:, t] == 0)[0])

                    oT = np.unique(np.concatenate(o))
                    y_pred1 = np.delete(y_pred1, oT, 0)
                    y_real1 = np.delete(y_real1, oT, 0)

            if len(y_pred1)>0:
                if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                    if daily==True:
                        cv, std_cv = evals(y_pred1, y_real1).cv_rmse_daily(mean_y, times)
                        nmbe, std_nmbe = evals(y_pred1, y_real1).nmbe_daily(mean_y, times)
                        rmse, std_rmse = evals(y_pred1, y_real1).rmse_daily(times)
                        r2 = evals(y_pred1, y_real1).r2()
                        cv = np.mean(cv)
                        nmbe=np.mean(nmbe)
                        res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv,'std_cv': std_cv, 'nmbe': nmbe,'std_nmbe': std_nmbe, 'rmse': rmse,'std_rmse': std_rmse, 'r2': r2}
                    else:
                        if mean_y.size == 0:
                            e = evals(y_pred1, y_real1).variation_rate()
                            if isinstance(self.weights, list):
                                cv= np.mean(e)
                            else:
                                #print(e)
                                #print(self.weights)
                                cv= np.sum(e * self.weights)
                            rmse= np.nan
                            nmbe= np.nan
                        else:
                            e_cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                            e_r = evals(y_pred1, y_real1).rmse()
                            e_n = evals(y_pred1, y_real1).nmbe(mean_y)
                            r2 = evals(y_pred1, y_real1).r2()
                            if isinstance(self.weights, list):
                                cv = np.mean(e_cv)
                                rmse = np.mean(e_r)
                                nmbe = np.mean(e_n)
                            else:
                                cv= np.sum(e_cv * self.weights)
                                rmse = np.sum(e_r * self.weights)
                                nmbe= np.sum(e_n * self.weights)

                        res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv, 'nmbe': nmbe,
                               'rmse': rmse, 'r2': r2, 'ind_out':i_out}
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
                    res = {'y_pred': y_predF, 'cv_rmse': cv, 'nmbe': nmbe,
                           'rmse': rmse, 'r2': r2,'ind_out':i_out}
            else:
                raise NameError('Empty prediction')

        elif self.zero_problem == 'radiation':
            print('*****Night-radiation fixed******')
            place = np.where(self.names == 'radiation')[0]
            scalar_x = self.scalar_x
            scalar_rad = scalar_x['radiation']
            res = super().fix_values_0(scalar_rad.inverse_transform(x_val[:,x_val.shape[1]-1, place]),
                                       self.zero_problem, self.limits)
            index_rad = res['indexes_out']
            index_rad2 = np.where(y_real <= self.inf_limit)[0]
            index_rad = np.union1d(np.array(index_rad), np.array(index_rad2))

            if len(y_pred) <= 1:
                y_pred1 = np.nan
                y_real1 = y_real
            else:
                if len(index_rad) > 0 and self.horizont == 0:
                    y_pred1 = np.delete(y_pred, index_rad, 0)
                    y_real1 = np.delete(y_real, index_rad, 0)
                elif len(index_rad) > 0 and self.horizont > 0:
                    y_pred1 = np.delete(y_pred, np.array(index_rad) - self.horizont, 0)
                    y_real1 = np.delete(y_real, np.array(index_rad) - self.horizont, 0)
                else:
                    y_pred1 = y_pred
                    y_real1 = y_real

                # Outliers and missing values
                if self.mask == True and len(y_pred1) > 0:
                    if mean_y.size == 0:
                        o = np.where(y_real1 < self.inf_limit)[0]
                        if len(o) > 0:
                            y_pred1 = np.delete(y_pred1, o, 0)
                            y_real1 = np.delete(y_real1, o, 0)

                    else:
                        o = list()
                        for t in range(len(mean_y)):
                            o.append(np.where(y_real1[:, t] < self.inf_limit[t])[0])

                        oT = np.unique(np.concatenate(o))
                        y_pred1 = np.delete(y_pred1, oT, 0)
                        y_real1 = np.delete(y_real1, oT, 0)

                if self.extract_cero == True and len(y_pred1) > 0:
                    if mean_y.size == 0:
                        o = np.where(y_real1 == 0)[0]
                        if len(o) > 0:
                            y_pred1 = np.delete(y_pred1, o, 0)
                            y_real1 = np.delete(y_real1, o, 0)
                    else:
                        o = list()
                        for t in range(len(mean_y)):
                            o.append(np.where(y_real1[:, t] == 0)[0])

                        oT = np.unique(np.concatenate(o))
                        y_pred1 = np.delete(y_pred1, oT, 0)
                        y_real1 = np.delete(y_real1, oT, 0)

            if len(y_pred1)>0:
                if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                    if daily == True:
                        print(times)
                        cv, std_cv = evals(y_pred1, y_real1).cv_rmse_daily(mean_y, times)
                        nmbe, std_nmbe = evals(y_pred1, y_real1).nmbe_daily(mean_y, times)
                        rmse, std_rmse = evals(y_pred1, y_real1).rmse_daily(times)
                        r2 = evals(y_pred1, y_real1).r2()
                        cv = np.mean(cv)
                        nmbe=np.mean(nmbe)
                        res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv, 'std_cv': std_cv, 'nmbe': nmbe, 'std_nmbe': std_nmbe,
                               'rmse': rmse, 'std_rmse': std_rmse, 'r2': r2}
                    else:
                        if mean_y.size == 0:
                            e = evals(y_pred1, y_real1).variation_rate()
                            if isinstance(self.weights, list):
                                cv = np.mean(e)
                            else:
                                cv = np.sum(e * self.weights)
                            rmse = np.nan
                            nmbe = np.nan
                        else:
                            e_cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                            e_r = evals(y_pred1, y_real1).rmse()
                            e_n = evals(y_pred1, y_real1).nmbe(mean_y)
                            r2 = evals(y_pred1, y_real1).r2()
                            if isinstance(self.weights, list):
                                cv = np.mean(e_cv)
                                rmse = np.mean(e_r)
                                nmbe = np.mean(e_n)
                            else:
                                cv = np.sum(e_cv * self.weights)
                                rmse = np.sum(e_r * self.weights)
                                nmbe = np.sum(e_n * self.weights)
                        res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv, 'nmbe': nmbe,
                               'rmse': rmse, 'r2': r2,'ind_out':i_out}
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
                    res = {'y_pred': y_predF, 'cv_rmse': cv, 'nmbe': nmbe,
                           'rmse': rmse, 'r2': r2,'ind_out':i_out}

            else:
                raise NameError('Empty prediction')
        else:
            # Outliers and missing values
            if self.mask == True and len(y_pred) > 0:
                if mean_y.size == 0:
                    o = np.where(y_real < self.inf_limit)[0]
                    if len(o) > 0:
                        y_pred = np.delete(y_pred, o, 0)
                        y_real = np.delete(y_real, o, 0)
                else:
                    o = list()
                    for t in range(len(mean_y)):
                        o.append(np.where(y_real[:, t] < self.inf_limit[t])[0])

                    oT = np.unique(np.concatenate(o))
                    y_pred = np.delete(y_pred, oT, 0)
                    y_real = np.delete(y_real, oT, 0)

            if self.extract_cero == True and len(y_pred) > 0:
                if mean_y.size == 0:
                    o = np.where(y_real == 0)[0]
                    if len(o) > 0:
                        y_pred = np.delete(y_pred, o, 0)
                        y_real = np.delete(y_real, o, 0)
                else:
                    o = list()
                    for t in range(len(mean_y)):
                        o.append(np.where(y_real[:, t] == 0)[0])

                    oT = np.unique(np.concatenate(o))
                    y_pred = np.delete(y_pred, oT, 0)
                    y_real = np.delete(y_real, oT, 0)
            if len(y_pred)>0:
                if np.sum(np.isnan(y_pred)) == 0 and np.sum(np.isnan(y_real)) == 0:
                    if daily == True:
                        cv, std_cv = evals(y_pred, y_real).cv_rmse_daily(mean_y, times)
                        nmbe, std_nmbe = evals(y_pred, y_real).nmbe_daily(mean_y, times)
                        rmse, std_rmse = evals(y_pred, y_real).rmse_daily(times)
                        r2 = evals(y_pred, y_real).r2()
                        cv = np.mean(cv)
                        nmbe=np.mean(nmbe)
                        res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv, 'std_cv': std_cv, 'nmbe': nmbe, 'std_nmbe': std_nmbe,
                               'rmse': rmse, 'std_rmse': std_rmse, 'r2': r2,'ind_out':i_out}
                    else:
                        if mean_y.size == 0:
                            e = evals(y_pred, y_real).variation_rate()
                            if isinstance(self.weights, list):
                                cv = np.mean(e)
                            else:
                                cv = np.sum(e * self.weights)
                            rmse = np.nan
                            nmbe = np.nan
                            r2=np.nan
                        else:
                            e_cv = evals(y_pred, y_real).cv_rmse(mean_y)
                            e_r = evals(y_pred, y_real).rmse()
                            e_n = evals(y_pred, y_real).nmbe(mean_y)
                            r2 = evals(y_pred, y_real).r2()
                            if isinstance(self.weights, list):
                                cv = np.mean(e_cv)
                                rmse = np.mean(e_r)
                                nmbe = np.mean(e_n)
                            else:
                                cv = np.sum(e_cv * self.weights)
                                rmse = np.sum(e_r * self.weights)
                                nmbe = np.sum(e_n * self.weights)
                        res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv, 'nmbe': nmbe,
                               'rmse': rmse, 'r2': r2,'ind_out':i_out}
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
                    res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv, 'nmbe': nmbe,
                           'rmse': rmse, 'r2': r2,'ind_out':i_out}
            else:
                raise NameError('Empty prediction')

        y_realF = pd.DataFrame(y_realF)
        y_realF.index = y_predF.index

        if plotting==True:
            a = np.round(cv, 2)
            up =int(np.max(y_realF)) + int(np.max(y_realF)/4)
            low = int(np.min(y_realF)) - int(np.min(y_realF)/4)
            plt.figure()
            plt.ylim(low, up)
            plt.plot(y_realF, color='black', label='Real')
            plt.plot(y_predF, color='blue', label='Prediction')
            plt.legend()
            plt.title("CV(RMSE)={}".format(str(a)))
            plt.show()
            plt.savefig('plot1.png')

        return res

    def optimal_search(self, fold, rep, neurons_dense, neurons_lstm, paciences, onebyone,batch, mean_y,parallel,weights,values):
        '''
        Parallelisation is not work tested!!!

        :param fold: assumed division of data sample
        :param rep: repetitions of cv analysis considering the intial or the final of sample
        :param parallel: 0 no paralyse
        :param top: number of best solution selected
        :return: errors obtained with the options considered together  with the best solutions
        '''

        error = [0 for x in range(len(neurons_lstm) * len(neurons_dense) * len(paciences))]
        complexity = [0 for x in range(len(neurons_lstm) * len(neurons_dense) * len(paciences))]

        options = {'neurons_dense': [], 'neurons_lstm': [], 'pacience': []}
        w = 0
        contador=len(neurons_lstm) * len(neurons_dense) * len(paciences)-1
        if parallel<2:
            for t in range(len(neurons_dense)):
                print('##################### Option ####################', w)
                neuron_dense = neurons_dense[t]
                for j in range(len(neurons_lstm)):
                    neuron_lstm = neurons_lstm[j]
                    for i in range(len(paciences)):
                        options['neurons_dense'].append(neuron_dense)
                        options['neurons_lstm'].append(neuron_lstm)
                        options['pacience'].append(paciences[i])
                        res = self.cv_analysis(fold, rep, neuron_lstm, neuron_dense,onebyone, paciences[i], batch, mean_y,values,False)
                        error[w] = np.mean(res['cv_rmse'])
                        complexity[w] = LSTM_model.complex(neuron_lstm, neuron_dense,2000,12)
                        w += 1
        elif parallel>=2:
            processes = []
            res2 = []
            dev2 = []
            z = 0

            q = Queue()
            for t in range(len(neurons_dense)):

                neuron_dense = neurons_dense[t]
                for j in range(len(neurons_lstm)):
                    neuron_lstm = neurons_lstm[j]

                    for i in range(len(paciences)):
                        print('##################### Option ####################', w)
                        options['neurons_dense'].append(neuron_dense)
                        options['neurons_lstm'].append(neuron_lstm)
                        options['pacience'].append(paciences[i])
                        if z < parallel and w<contador:
                            p = Process(target=self.cv_analysis,
                                        args=(fold,rep, neuron_lstm, neuron_dense, paciences[i], batch, mean_y,values,False, q))
                            p.start()

                            processes.append(p)
                            z1 =z+ 1
                        if z == parallel and w < contador:
                            p.close()
                            for p in processes:
                                p.join()

                            for v in range(len(processes)):
                                res2.append(q.get()[0])
                                res2.append(q.get()[1])

                            processes=[]
                            q = Queue()
                            p = Process(target=self.cv_analysis,
                                        args=(fold, rep, neuron_lstm, neuron_dense, paciences[i], batch, mean_y,values,False, q))
                            p.start()

                            processes.append(p)
                            z1 = 1

                        elif w==contador:
                            p = Process(target=self.cv_analysis,
                                        args=(fold, rep, neuron_lstm, neuron_dense, paciences[i], batch, mean_y,values,False, q))
                            p.start()

                            processes.append(p)
                            p.close()
                            for p in processes:
                                p.join()

                            for v in range(len(processes)):
                                res2.append(q.get()[0])
                                dev2.append(q.get()[1])
                        z=z1

                        w += 1
            error = res2
            complexity = dev2
        else:
            raise NameError('Option not considered')

        r1 = error.copy()
        d1 = complexity.copy()
        print('Resultados search', r1)

        scal_cv = MinMaxScaler(feature_range=(0, 1))
        scal_com = MinMaxScaler(feature_range=(0, 1))

        scal_cv.fit(np.array(r1).reshape(-1, 1))
        scal_com.fit(np.array(d1).reshape(-1, 1))

        cv = scal_cv.transform(np.array(r1).reshape(-1, 1))
        com = scal_com.transform(np.array(d1).reshape(-1, 1))

        r_final = np.array([cv[:, 0], com[:, 0]]).T

        I = get_decomposition("aasf", beta=5).do(r_final, weights).argmin()
        #I = get_decomposition("pbi").do(r_final, weights).argmin()

        top_result = {'error': [], 'complexity': [], 'neurons_dense': [],'neurons_lstm':[], 'pacience': []}
        top_result['error']=r1[I]
        top_result['complexity']=d1[I]
        top_result['neurons_dense']=options['neurons_dense'][I]
        top_result['neurons_lstm']=options['neurons_lstm'][I]
        top_result['pacience']=options['pacience'][I]

        print(top_result['error'])
        print(top_result['complexity'])
        print(top_result['neurons_lstm'])
        print(top_result['neurons_dense'])
        print(top_result['pacience'])

        np.savetxt('objectives_selected_brute.txt', np.array([top_result['error'],top_result['complexity']]))
        np.savetxt('x_selected_brute.txt', np.concatenate((top_result['neurons_lstm'],top_result['neurons_dense'],np.array([top_result['pacience']]))))

        plt.figure(figsize=(12,9))
        plt.scatter(r_final[:, 0], r_final[:, 1], color='black')
        plt.xlabel('Normalised CV (RMSE)', fontsize=20, labelpad=10)
        plt.ylabel('Normalised Complexity', fontsize=20, labelpad=10)
        plt.scatter(r_final[I, 0], r_final[I, 1], s=175, color='red', alpha=1, marker='o', facecolors='none',
                    label='Optimum')
        plt.legend(borderpad=1.25)
        plt.savefig('optimisation_plot.png')


        print('Process finished!!!')
        res = {'errors': r1, 'complexity':d1, 'options': options, 'best': top_result}

        return res

    def nsga2_individual(self,med, contador,n_processes,l_lstm, l_dense, batch,pop_size,tol,n_last, nth_gen, xlimit_inf, xlimit_sup,dictionary,onebyone,values,weights):
        '''
        :param med:
        :param contador: a operator to count the attempts
        :param n_processes: how many processes are parallelise
        :param l_lstm:maximun number of layers lstm
        :param l_dense:maximun number of layers dense
        :param batch: batch size
        :param pop_size: population size selected for NSGA2
        :param tol: tolearance selected to terminate the process
        :param xlimit_inf: array with the lower limits to the neuron  lstm , neurons dense and pacience
        :param xlimit_sup:array with the upper limits to the neuron  lstm , neurons dense and pacience
        :param dictionary: dictionary to stored the options tested
        :return: options in Pareto front, the optimal selection and the total results. Consider the option of parallelisation with runners
        '''

        if n_processes>1:
            pool = multiprocessing.Pool(n_processes)
            problem = MyProblem(self.names,self.extract_cero,self.horizont, self.scalar_y, self.zero_problem, self.limits,self.times,self.pos_y,self.mask,
                                self.mask_value, self.n_lags,self.n_steps,self.inf_limit, self.sup_limit, self.repeat_vector, self.type, self.data,
                                self.scalar_x,self.dropout,self.weights,med, contador,len(xlimit_inf),l_lstm, l_dense, batch, xlimit_inf, xlimit_sup,dictionary,onebyone,values,runner = pool.starmap,func_eval=starmap_parallelized_eval)
        else:
            problem = MyProblem(self.names,self.extract_cero,self.horizont, self.scalar_y, self.zero_problem, self.limits,self.times,self.pos_y,self.mask,
                                self.mask_value, self.n_lags,self.n_steps, self.inf_limit, self.sup_limit, self.repeat_vector, self.type, self.data,
                                self.scalar_x, self.dropout,self.weights,med, contador,len(xlimit_inf),l_lstm, l_dense, batch, xlimit_inf, xlimit_sup,dictionary,onebyone,values)

        algorithm = NSGA2(pop_size=pop_size, repair=MyRepair(l_lstm, l_dense), eliminate_duplicates=True,
                          sampling=get_sampling("int_random"),
                          crossover=get_crossover("int_sbx"),
                          mutation=get_mutation("int_pm", prob=0.1))

        termination = MultiObjectiveSpaceToleranceTermination(tol=tol,
                                                              n_last=n_last, nth_gen=nth_gen, n_max_gen=None,
                                                              n_max_evals=6000)
        '''
        Termination can be with tolerance or with generations limit
        '''
        res = minimize(problem,
                       algorithm,
                       termination,
                       # ("n_gen", 20),
                       pf=True,
                       verbose=True,
                       seed=7)

        if res.F.shape[0] > 1:
            rf=res.F
            rx=res.X
            scal_cv = MinMaxScaler(feature_range=(0, 1))
            scal_com = MinMaxScaler(feature_range=(0, 1))

            scal_cv.fit(res.F[:,0].reshape(-1,1))
            scal_com.fit(res.F[:,1].reshape(-1,1))

            cv=scal_cv.transform(res.F[:,0].reshape(-1,1))
            com=scal_com.transform(res.F[:,1].reshape(-1,1))

            r_final = np.array([cv[:,0], com[:,0]]).T

            I = get_decomposition("aasf", beta=5).do(r_final, weights).argmin()
            #I = get_decomposition("pbi").do(r_final, weights).argmin()

            obj_T = res.F
            struct_T = rx
            obj = res.F[I, :]
            struct = rx[I, :]
            print(rf.shape)
            print(rx.shape)
            print(r_final.shape)

            plt.figure(figsize=(10, 7))
            plt.scatter(r_final[:, 0], r_final[:, 1], color='black')
            plt.xlabel('Normalised CV (RMSE)', fontsize=20, labelpad=10)
            plt.ylabel('Normalised Complexity', fontsize=20, labelpad=10)
            plt.scatter(r_final[I, 0], r_final[I, 1], s=200, color='red', alpha=1, marker='o', facecolors='none',
                        label='Optimum')
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(borderpad=1, fontsize=15)
            plt.savefig('optimisation_plot.png')
        else:
            obj_T = res.F
            struct_T = res.X
            obj = res.F
            struct = res.X

        print('The number of evaluations were:', contador)
        if n_processes>1:
            pool.close()
        else:
            pass

        return (obj, struct,obj_T, struct_T,  res,contador)

    def optimal_search_nsga2(self,l_lstm, l_dense, batch, pop_size, tol,xlimit_inf, xlimit_sup, mean_y,parallel, onebyone, values, weights, n_last=5, nth_gen=5):
        '''
        :param l_lstm: maximun layers lstm (first layer never 0 neurons (input layer))
        :param l_dense: maximun layers dense
        :param batch: batch size
        :param pop_size: population size for NSGA2
        :param tol: tolerance to built the pareto front
        :param xlimit_inf: array with lower limits for neurons lstm (range of number multiplied by 10), dense (range of number multiplied by 10) and
        pacience (range of number multiplied by 10)
        :param xlimit_sup: array with upper limits for neurons lstm, dense and pacience
        :param parallel: how many processes are parallelise
        n_last: more robust, we consider the last n generations and take the maximum
        nth_gen: whenever the termination criterion is calculated
        :return: the options selected for the pareto front, the optimal selection and the total results
        '''

        manager = multiprocessing.Manager()
        dictionary = manager.dict()
        contador = manager.list()
        contador.append(0)
        print('start optimisation!!!')
        obj, x_obj, obj_total, x_obj_total,res,evaluations = self.nsga2_individual(mean_y, contador,parallel,l_lstm, l_dense, batch,pop_size,tol, n_last, nth_gen,xlimit_inf, xlimit_sup,dictionary, onebyone,values, weights)

        np.savetxt('objectives_selected.txt', obj)
        np.savetxt('x_selected.txt', x_obj)
        np.savetxt('objectives.txt', obj_total)
        np.savetxt('x.txt', x_obj_total)
        np.savetxt('evaluations.txt', evaluations)

        print('Process finished!!!')
        print('The selection is \n', x_obj, 'with a result of \n', obj)
        res = {'total_x': x_obj_total, 'total_obj': obj_total, 'opt_x': x_obj, 'opt_obj':obj, 'res':res,'evaluations':evaluations}
        return res


    def rnsga2_individual(self,med, contador,n_processes,l_lstm, l_dense, batch,pop_size,tol,n_last, nth_gen,xlimit_inf, xlimit_sup,dictionary,onebyone,values,weights,epsilon):
        '''
        :param med:
        :param contador: a operator to count the attempts
        :param n_processes: how many processes are parallelise
        :param l_lstm:maximun number of layers lstm
        :param l_dense:maximun number of layers dense
        :param batch: batch size
        :param epsilon: smaller generates solutions tighter
        :param pop_size: population size selected for RVEA
        :param tol: tolearance selected to terminate the process
        :param xlimit_inf: array with the lower limits to the neuron  lstm , neurons dense and pacience
        :param xlimit_sup:array with the upper limits to the neuron  lstm , neurons dense and pacience
        :param dictionary: dictionary to stored the options tested
        :return: options in Pareto front, the optimal selection and the total results. Consider the option of parallelisation with runners
        '''

        if n_processes>1:
            pool = multiprocessing.Pool(n_processes)
            problem = MyProblem(self.names,self.extract_cero,self.horizont, self.scalar_y, self.zero_problem, self.limits,self.times,self.pos_y,self.mask,
                                self.mask_value, self.n_lags,self.n_steps,self.inf_limit, self.sup_limit, self.repeat_vector, self.type, self.data,
                                self.scalar_x,self.dropout,self.weights,med, contador,len(xlimit_inf),l_lstm, l_dense, batch, xlimit_inf, xlimit_sup,dictionary,onebyone,values,runner = pool.starmap,func_eval=starmap_parallelized_eval)
        else:
            problem = MyProblem(self.names,self.extract_cero,self.horizont, self.scalar_y, self.zero_problem, self.limits,self.times,self.pos_y,self.mask,
                                self.mask_value, self.n_lags,self.n_steps,self.inf_limit, self.sup_limit, self.repeat_vector, self.type, self.data,
                                self.scalar_x, self.dropout,self.weights,med, contador,len(xlimit_inf),l_lstm, l_dense, batch, xlimit_inf, xlimit_sup,dictionary,onebyone,values)


        ref_points = np.array([[0.3, 0.1], [0.1, 0.3]])

        algorithm = RNSGA2(ref_points, pop_size=pop_size, sampling=get_sampling("int_random"),
                          crossover=get_crossover("int_sbx"),
                          mutation=get_mutation("int_pm", prob=0.1),
                           normalization='front',
                           extreme_points_as_reference_points=False,
                           weights=weights,
                           epsilon=epsilon)

        termination = MultiObjectiveSpaceToleranceTermination(tol=tol,
                                                              n_last=n_last, nth_gen=nth_gen, n_max_gen=None,
                                                              n_max_evals=6000)
        res = minimize(problem,
                       algorithm,
                       termination,
                       pf=True,
                       verbose=True,
                       seed=7)

        if res.F.shape[0] > 1:
            rf=res.F
            rx=res.X
            scal_cv = MinMaxScaler(feature_range=(0, 1))
            scal_com = MinMaxScaler(feature_range=(0, 1))

            scal_cv.fit(res.F[:,0].reshape(-1,1))
            scal_com.fit(res.F[:,1].reshape(-1,1))

            cv=scal_cv.transform(res.F[:,0].reshape(-1,1))
            com=scal_com.transform(res.F[:,1].reshape(-1,1))

            r_final = np.array([cv[:,0], com[:,0]]).T

            I = get_decomposition("aasf", beta=5).do(r_final, weights).argmin()
            #I = get_decomposition("pbi").do(r_final, weights).argmin()

            obj_T = res.F
            struct_T = rx
            obj = res.F[I, :]
            struct = rx[I, :]
            print(rf.shape)
            print(rx.shape)

            plt.figure(figsize=(10, 7))
            plt.scatter(r_final[:, 0], r_final[:, 1], color='black')
            plt.xlabel('Normalised CV (RMSE)', fontsize=20, labelpad=10)
            plt.ylabel('Normalised Complexity', fontsize=20, labelpad=10)
            plt.scatter(r_final[I, 0], r_final[I, 1], s=200, color='red', alpha=1, marker='o', facecolors='none',
                        label='Optimum')
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(borderpad=1, fontsize=15)
            plt.savefig('optimisationR_plot.png')

        else:
            obj_T = res.F
            struct_T = res.X
            obj = res.F
            struct = res.X

        print('The number of evaluations were:', contador)
        if n_processes>1:
            pool.close()
        else:
            pass

        return (obj, struct,obj_T, struct_T,  res,contador)



    def optimal_search_rnsga2(self,l_lstm, l_dense, batch, pop_size, tol,xlimit_inf, xlimit_sup, mean_y,parallel, onebyone, values, weights, epsilon=0.01,n_last=5, nth_gen=5):
        '''
        :param l_lstm: maximun layers lstm (first layer never 0 neurons (input layer))
        :param l_dense: maximun layers dense
        :param batch: batch size
        :param pop_size: population size for RVEA
        :param tol: tolerance to built the pareto front
        :param epsilon: smaller generates solutions tighter
        :param xlimit_inf: array with lower limits for neurons lstm (range of number multiplied by 10), dense (range of number multiplied by 10) and
        pacience (range of number multiplied by 10)
        :param xlimit_sup: array with upper limits for neurons lstm, dense and pacience
        :param parallel: how many processes are parallelise
        n_last: more robust, we consider the last n generations and take the maximum
        nth_gen: whenever the termination criterion is calculated
        :return: the options selected for the pareto front, the optimal selection and the total results
        '''

        manager = multiprocessing.Manager()
        dictionary = manager.dict()
        contador = manager.list()
        contador.append(0)
        print('start optimisation!!!')
        obj, x_obj, obj_total, x_obj_total,res,evaluations = self.rnsga2_individual(mean_y, contador,parallel,l_lstm, l_dense, batch,pop_size,tol,n_last, nth_gen,xlimit_inf, xlimit_sup,dictionary, onebyone,values, weights,epsilon)

        np.savetxt('objectives_selectedR.txt', obj)
        np.savetxt('x_selectedR.txt', x_obj)
        np.savetxt('objectivesR.txt', obj_total)
        np.savetxt('xR.txt', x_obj_total)
        np.savetxt('evaluationsR.txt', evaluations)

        print('Process finished!!!')
        print('The selection is \n', x_obj, 'with a result of \n', obj)
        res = {'total_x': x_obj_total, 'total_obj': obj_total, 'opt_x': x_obj, 'opt_obj':obj, 'res':res,'evaluations':evaluations}
        return res

from pymoo.core.repair import Repair
class MyRepair(Repair):
    def info(self):
        '''
        l_lstm: number of LSTM layers
        l_dense: number of Dense layers
        :return:
        '''
        print('Class defining a function to repair the possible error of the genetic algorithm. If a layer is zero the next layer cannot have positive neurons')

    def __init__(self,l_lstm, l_dense):
        self.l_lstm=l_lstm
        self.l_dense = l_dense

    def _do(self, problem, pop, **kwargs):
        print('FIXING X')
        for k in range(len(pop)):
            x = pop[k].X
            xx = x[range(self.l_lstm + self.l_dense)]
            x1 = xx[range(self.l_lstm)]
            x2 = xx[range(self.l_lstm, self.l_lstm + self.l_dense)]
            r_lstm, r_dense = MyProblem.bool4(xx, self.l_lstm, self.l_dense)

            if len(r_lstm) == 1:
                if r_lstm == 0:
                    pass
                elif r_lstm != 0:
                    x1[r_lstm] = 0
            elif len(r_lstm) > 1:
                x1[r_lstm] = 0

            if len(r_dense) == 1:
                if r_dense == 0:
                    pass
                elif r_dense != 0:
                    x2[r_dense] = 0
            elif len(r_dense) > 1:
                x2[r_dense] = 0
            x = np.concatenate((x1, x2, np.array([x[len(x) - 1]])))
            pop[k].X = x

        print('X FIXED')
        return pop



from pymoo.core.problem import ElementwiseProblem
class MyProblem(ElementwiseProblem):
    def info(self):
        print('Class to create a specific problem to use NSGA2 in architectures search. Two objectives and a constraint (Repair) concerning the neurons in each layer')


    def __init__(self,names,extract_cero, horizont,scalar_y,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,n_steps,  inf_limit,sup_limit, repeat_vector, type,data,scalar_x,dropout, weights, med, contador,
                 n_var,l_lstm, l_dense,batch,xlimit_inf, xlimit_sup,dictionary,onebyone,values, **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_constr=2,
                         xl=xlimit_inf,
                         xu=xlimit_sup,
                         type_var=np.int,
                         #elementwise_evaluation=True,
                         **kwargs)

        self.names=names
        self.extract_cero=extract_cero
        self.data=data
        self.horizont = horizont
        self.scalar_y = scalar_y
        self.scalar_x = scalar_x
        self.zero_problem = zero_problem
        self.limits = limits
        self.times = times
        self.pos_y = pos_y
        self.mask = mask
        self.mask_value = mask_value
        self.n_lags=n_lags
        self.n_steps=n_steps
        self.inf_limit = inf_limit
        self.sup_limit = sup_limit
        self.repeat_vector = repeat_vector
        self.dropout = dropout
        self.type = type
        self.med = med
        self.contador = contador
        self.l_lstm = l_lstm
        self.l_dense = l_dense
        self.batch = batch
        self.xlimit_inf = xlimit_inf
        self.xlimit_sup = xlimit_sup
        self.n_var = n_var
        self.dictionary =dictionary
        self.onebyone =onebyone
        self.values=values
        self.weights=weights

    def cv_opt(self, data, fold, rep, neurons_lstm, neurons_dense, pacience, batch, mean_y, dictionary):
        '''
        :param fold:assumed division of the sample for cv
        :param rep:repetition of the estimation in each subsample
        :param dictionary: dictionary to fill with the options tested
        :param q:operator to differentiate when there is parallelisation and the results must be a queue
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        :return: cv(rmse) and complexity of the model tested
        '''
        name1 = tuple(np.concatenate((neurons_lstm, neurons_dense, np.array([pacience]))))
        try:
            a0, a1 = dictionary[name1]
            return a0, a1
        except KeyError:
            pass
        cvs = [0 for x in range(rep * 2)]
        names = self.names
        names = np.delete(names, self.pos_y)
        res = LSTM_model.cv_division_lstm(data, self.horizont, fold, self.pos_y, self.n_lags, self.n_steps,
                                          self.onebyone, self.values)
        x_test = np.array(res['x_test'])
        x_train = np.array(res['x_train'])
        x_val = np.array(res['x_val'])
        y_test = np.array(res['y_test'])
        y_train = np.array(res['y_train'])
        y_val = np.array(res['y_val'])
        #
        times_val = res['time_val']
        #
        if self.type == 'regression':
            # Train the model
            zz = 0
            if self.values:
                stop = len(self.values[0])
            else:
                stop = len(x_train)
            for z in range(stop):
                print('Fold number', z)
                time_start = time()
                if len(y_train[z].shape) > 1:
                    ytrain = y_train[z]
                    ytest = y_test[z]
                    yval = y_val[z]
                else:
                    ytrain = y_train[z].reshape(len(y_train[z]), 1)
                    ytest = y_test[z].reshape(len(y_test[z]), 1)
                    yval = y_val[z].reshape(len(y_val[z]), 1)
                model = LSTM_model.built_model_regression(x_train[z], ytrain, neurons_lstm,
                                                          neurons_dense,
                                                          self.mask, self.mask_value, self.repeat_vector,
                                                          self.dropout)
                model, history = LSTM_model.train_model(model, x_train[z], ytrain, x_test[z], ytest, pacience,
                                                        batch)
                print('Teh training spent ', time() - time_start)
                if isinstance(self.pos_y, collections.abc.Sized):
                    outputs = len(self.pos_y)
                else:
                    outputs = 1
                res = LSTM_model.predict_model(model, self.n_lags, x_val[z], batch, outputs)
                y_pred = res['y_pred']
                print('Y_val SHAPE in CV_OPT', yval[z].shape)
                print('X_val SHAPE in CV_OPT', x_val[z].shape)
                y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
                y_real = np.array(self.scalar_y.inverse_transform(yval))
                if isinstance(self.pos_y, collections.abc.Sized):
                    for t in range(len(self.pos_y)):
                        y_pred[np.where(y_pred[:, t] < self.inf_limit[t])[0], t] = self.inf_limit[t]
                        y_pred[np.where(y_pred[:, t] > self.sup_limit[t])[0], t] = self.sup_limit[t]
                    y_real = y_real
                else:
                    y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
                    y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit
                    y_real = y_real.reshape(-1, 1)
                print('Y_pred SHAPE in CV_OPT ', y_pred.shape)
                print('Dates SHAPE this piece', times_val[z].shape)
                y_predF = y_pred.copy()
                y_predF = pd.DataFrame(y_predF)
                y_predF.index = times_val[z]
                y_realF = y_real.copy()
                y_realF = pd.DataFrame(y_realF)
                y_realF.index = times_val[z]
                if self.zero_problem == 'schedule':
                    print('*****Night-schedule fixed******')
                    res = DL.fix_values_0(times_val[z][:, 0],
                                          self.zero_problem, self.limits)
                    index_hour = res['indexes_out']
                    if len(index_hour) > 0 and self.horizont == 0:
                        y_pred1 = np.delete(y_pred, index_hour, 0)
                        y_real1 = np.delete(y_real, index_hour, 0)
                    elif len(index_hour) > 0 and self.horizont > 0:
                        y_pred1 = np.delete(y_pred, index_hour - self.horizont, 0)
                        y_real1 = np.delete(y_real, index_hour - self.horizont, 0)
                    else:
                        y_pred1 = y_pred
                        y_real1 = y_real
                    # Outliers and missing values
                    if self.mask == True and len(y_pred1) > 0:
                        if mean_y.size == 0:
                            o = np.where(y_real1 < self.inf_limit)[0]
                            if len(o) > 0:
                                y_pred1 = np.delete(y_pred1, o, 0)
                                y_real1 = np.delete(y_real1, o, 0)
                        else:
                            o = list()
                            for t in range(len(mean_y)):
                                o.append(np.where(y_real1[:, t] < self.inf_limit[t])[0])

                            oT = np.unique(np.concatenate(o))
                            y_pred1 = np.delete(y_pred1, oT, 0)
                            y_real1 = np.delete(y_real1, oT, 0)

                    if self.extract_cero == True and len(y_pred1) > 0:
                        if mean_y.size == 0:
                            o = np.where(y_real1 == 0)[0]
                            if len(o) > 0:
                                y_pred1 = np.delete(y_pred1, o, 0)
                                y_real1 = np.delete(y_real1, o, 0)
                        else:
                            o = list()
                            for t in range(len(mean_y)):
                                o.append(np.where(y_real1[:, t] == 0)[0])

                            oT = np.unique(np.concatenate(o))
                            y_pred1 = np.delete(y_pred1, oT, 0)
                            y_real1 = np.delete(y_real1, oT, 0)

                    if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0 and len(
                            y_pred1) > 0 and len(y_real1) > 0:
                        if mean_y.size == 0:
                            e = evals(y_pred1, y_real1).variation_rate()
                            if isinstance(self.weights, list):
                                cvs[zz] = np.mean(e)
                            else:
                                cvs[zz] = np.sum(e * self.weights)
                        else:
                            e = evals(y_pred1, y_real1).cv_rmse(mean_y)
                            if isinstance(self.weights, list):
                                cvs[zz] = np.mean(e)
                            else:
                                cvs[zz] = np.sum(e * self.weights)
                    else:
                        print('Missing values are detected when we are evaluating the predictions')
                        cvs[zz] = 9999
                elif self.zero_problem == 'radiation':
                    print('*****Night-radiation fixed******')
                    place = np.where(names == 'radiation')[0]
                    scalar_rad = self.scalar_x['radiation']
                    res = DL.fix_values_0(scalar_rad.inverse_transform(x_val[z][:, self.n_lags - 1, place]),
                                          self.zero_problem, self.limits)
                    index_rad = res['indexes_out']
                    index_rad2 = np.where(y_real <= self.inf_limit)[0]
                    index_rad = np.union1d(np.array(index_rad), np.array(index_rad2))
                    if len(index_rad) > 0 and self.horizont == 0:
                        y_pred1 = np.delete(y_pred, index_rad, 0)
                        y_real1 = np.delete(y_real, index_rad, 0)
                    elif len(index_rad) > 0 and self.horizont > 0:
                        y_pred1 = np.delete(y_pred, np.array(index_rad) - self.horizont, 0)
                        y_real1 = np.delete(y_real, np.array(index_rad) - self.horizont, 0)
                    else:
                        y_pred1 = y_pred
                        y_real1 = y_real

                    # Outliers and missing values
                    if self.mask == True and len(y_pred1) > 0:
                        if mean_y.size == 0:
                            o = np.where(y_real1 < self.inf_limit)[0]
                            if len(o) > 0:
                                y_pred1 = np.delete(y_pred1, o, 0)
                                y_real1 = np.delete(y_real1, o, 0)
                        else:
                            o = list()
                            for t in range(len(mean_y)):
                                o.append(np.where(y_real1[:, t] < self.inf_limit[t])[0])

                            oT = np.unique(np.concatenate(o))
                            y_pred1 = np.delete(y_pred1, oT, 0)
                            y_real1 = np.delete(y_real1, oT, 0)

                    if self.extract_cero == True and len(y_pred1) > 0:
                        if mean_y.size == 0:
                            o = np.where(y_real1 == 0)[0]
                            if len(o) > 0:
                                y_pred1 = np.delete(y_pred1, o, 0)
                                y_real1 = np.delete(y_real1, o, 0)
                        else:
                            o = list()
                            for t in range(len(mean_y)):
                                o.append(np.where(y_real1[:, t] == 0)[0])

                            oT = np.unique(np.concatenate(o))
                            y_pred1 = np.delete(y_pred1, oT, 0)
                            y_real1 = np.delete(y_real1, oT, 0)
                    if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0 and len(
                            y_pred1) > 0 and len(y_real1) > 0:
                        if mean_y.size == 0:
                            e = evals(y_pred1, y_real1).variation_rate()
                            if isinstance(self.weights, list):
                                cvs[zz] = np.mean(e)
                            else:
                                cvs[zz] = np.sum(e * self.weights)
                        else:
                            e = evals(y_pred1, y_real1).cv_rmse(mean_y)
                            if isinstance(self.weights, list):
                                cvs[zz] = np.mean(e)
                            else:
                                cvs[zz] = np.sum(e * self.weights)
                    else:
                        print('Missing values are detected when we are evaluating the predictions')
                        cvs[zz] = 9999
                else:
                    if self.mask == True and len(y_pred) > 0:
                        if mean_y.size == 0:
                            o = np.where(y_real < self.inf_limit)[0]
                            if len(o) > 0:
                                y_pred = np.delete(y_pred, o, 0)
                                y_real = np.delete(y_real, o, 0)
                        else:
                            o = list()
                            for t in range(len(mean_y)):
                                o.append(np.where(y_real[:, t] < self.inf_limit[t])[0])

                            oT = np.unique(np.concatenate(o))
                            y_pred = np.delete(y_pred, oT, 0)
                            y_real = np.delete(y_real, oT, 0)

                    if self.extract_cero == True and len(y_pred) > 0:
                        if mean_y.size == 0:
                            o = np.where(y_real == 0)[0]
                            if len(o) > 0:
                                y_pred = np.delete(y_pred, o, 0)
                                y_real = np.delete(y_real, o, 0)
                        else:
                            o = list()
                            for t in range(len(mean_y)):
                                o.append(np.where(y_real[:, t] == 0)[0])

                            oT = np.unique(np.concatenate(o))
                            y_pred = np.delete(y_pred, oT, 0)
                            y_real = np.delete(y_real, oT, 0)
                    if np.sum(np.isnan(y_pred)) == 0 and np.sum(np.isnan(y_real)) == 0 and len(
                            y_pred) > 0 and len(y_real) > 0:
                        if mean_y.size == 0:
                            e = evals(y_pred, y_real).variation_rate()
                            if isinstance(self.weights, list):
                                cvs[zz] = np.mean(e)
                            else:
                                # print(e)
                                # print(self.weights)
                                cvs[zz] = np.sum(e * self.weights)
                        else:
                            e = evals(y_pred, y_real).cv_rmse(mean_y)
                            if isinstance(self.weights, list):
                                cvs[zz] = np.mean(e)
                            else:
                                cvs[zz] = np.sum(e * self.weights)
                    else:
                        print('Missing values are detected when we are evaluating the predictions')
                        cvs[zz] = 9999
                zz += 1
            #
            complexity = LSTM_model.complex(neurons_lstm, neurons_dense, 2000, 12)
            dictionary[name1] = np.mean(cvs), complexity
            res_final = {'cvs': np.mean(cvs), 'complexity': complexity}
            print(res_final)
            return res_final['cvs'], res_final['complexity']

    #
    @staticmethod
    def bool4(x, l_lstm, l_dense):
        '''
        :x: neurons options
        l_lstm: number of values that represent lstm neurons
        l_dense: number of values that represent dense neurons
        :return: 0 if the constraint is fulfilled
        '''
#
        x1 = x[range(l_lstm)]
        x2 = x[range(l_lstm, l_lstm+l_dense)]
        print('LSTM neurons', x1)
        print('Dense neurons', x2)
#

        if len(x2) == 2:
            if x2[0] == 0 and x2[1] > 0:
                a_dense = np.array([1])
            else:
                a_dense = np.array([0])
        elif len(x2)==3:
            if x2[0] == 0 and x2[1] > 0:
                a_dense = np.array([1])
                if x2[2] > 0:
                    a_dense = np.array([1,2])
            elif x2[1] == 0 and x2[2] > 0:
                a_dense = np.array([2])
            else:
                a_dense = np.array([0])
        elif len(x2)==4:
            if x2[0] == 0 and x2[1] > 0:
                a_dense = np.array([1])
                if x2[2] > 0:
                    a_dense = np.array([1,2])
            elif x2[0] == 0 and x2[2] > 0:
                a_dense = np.array([2])
            elif x2[0] == 0 and x2[3] > 0:
                a_dense = np.array([3])
            elif x2[1] == 0 and x2[2] > 0:
                a_dense = np.array([2])
                if x2[3] > 0:
                    a_dense = np.array([2,3])
            elif x2[1] == 0 and x2[3] > 0:
                a_dense = np.array([3])
            elif x2[2] == 0 and x2[3] > 0:
                a_dense = np.array([3])
            else:
                a_dense = np.array([0])
        else:
            raise NameError('Option not considered')
#
        if len(x1) == 2:
            if x1[0] == 0 and x1[1] > 0:
                a_lstm = np.array([1])
            else:
                a_lstm = np.array([0])
        elif len(x1)==3:
            if x1[1] == 0 and x1[2] > 0:
                a_lstm = np.array([2])
            else:
                a_lstm = np.array([0])
        elif len(x1)==4:
            if x1[1] == 0 and x1[2] > 0:
                a_lstm = np.array([2])
                if x1[3] > 0:
                    a_lstm = np.array([2, 3])
            elif x1[1] == 0 and x1[3] > 0:
                a_lstm=np.array([3])
            elif x1[2] == 0 and x1[3] > 0:
                a_lstm = np.array([3])
            else:
                a_lstm = np.array([0])
        else:
            raise NameError('Option not considered')
        print('checked the x options')
        return a_lstm, a_dense

    def _evaluate(self, x, out, *args, **kwargs):
        g1,g2 = MyProblem.bool4(np.delete(x, len(x)-1), self.l_lstm, self.l_dense)
        out["G"] =np.column_stack([g1, g2])
#
        print('##########################################  X=',x,'##########################################')
#
        n_lstm = x[range(self.l_lstm)]*20
        n_dense = x[range(self.l_lstm, self.l_lstm + self.l_dense)]*20
        n_pacience = x[len(x)-1]*20

        f1, f2 = self.cv_opt(self.data,2,1, n_lstm, n_dense, n_pacience, self.batch, self.med,self.dictionary)
        print(
            '\n ############################################## \n ############################# \n ########################## EVALUATION ',
            self.contador, '\n ######################### \n #####################################')

        self.contador[0] += 1

        print('F1:',f1)
        print('F2:',f2)

        out["F"] = np.column_stack([f1, f2])
