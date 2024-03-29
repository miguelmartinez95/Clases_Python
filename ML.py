from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from errors import Eval_metrics as evals
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Masking, Dropout
from datetime import datetime
from time import time
#import skfda
import collections
import math
import multiprocessing
from multiprocessing import Process,Manager,Queue
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from sklearn import svm
from pymoo.factory import get_decomposition
import tensorflow as tf
from pathlib import Path
import random
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem, get_visualization, get_decomposition
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.optimize import minimize
from pymoo.core.problem import starmap_parallelized_eval

'''
Conecting with GUPs
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)


class ML:
    def info(self):
        print(('Super class to built different machine learning models. This class has other more specific classes associated with it  \n'
              'Positions_y is required to be 0 or len(data) \n'
               'Zero problem is related with the night-day issue \n'
               'Horizont is focused on the future \n'
               'Limit is the radiation limit and schedule is the working hours'
               'IMPORTANT: the variables that can be lagged to the end of data frame'
              ))

    def __init__(self, data,horizont, scalar_y,scalar_x, zero_problem,limits,extract_cero, times, pos_y, n_lags, mask, mask_value, inf_limit,sup_limit):
        self.data = data
        self.horizont = horizont
        self.scalar_y = scalar_y
        self.scalar_x = scalar_x
        self.zero_problem = zero_problem
        self.times = times
        self.limits = limits
        self.extract_cero=extract_cero
        self.pos_y = pos_y
        self.n_lags = n_lags
        self.mask = mask
        self.mask_value = mask_value
        self.sup_limit = sup_limit
        self.inf_limit = inf_limit

    @staticmethod
    def cv_division(x,y, fold):
        '''
        Division de la muestra en trozos según fold para datos normales y no recurrentes

        :param fold: division for cv_analysis
        :return: data divided into train, test and validations in addition to the indexes division for a CV analysis
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
                a = x.iloc[range(w,w2)]
                X_val.append(a.iloc[range(len(a)-math.ceil(len(a)/2), len(a)-1)])
                X_test.append(a.drop(a.index[range(len(a)-math.floor(len(a)/2), len(a))]))
                X_train.append(x.drop(range(w,w2)))

                a = y.iloc[range(w, w2)]
                Y_val.append(a.iloc[range(len(a)-math.ceil(len(a)/2), len(a)-1)])
                Y_test.append(a.drop(a.index[range(len(a)-math.floor(len(a)/2), len(a))]))
                Y_train.append(y.drop(range(w, w2)))
                indexes.append(np.array([w2-math.ceil(len(a)/2)+1, w2]))
                w = w2
                w2 += step
                if w2 > x.shape[0] and w < x.shape[0]:
                    w2 = x.shape[0]
        except:
            raise NameError('Problems with the sample division in the cv classic')

        res = {'x_test': X_test, 'x_train':X_train,'x_val':X_val, 'y_test':Y_test, 'y_train':Y_train, 'y_val':Y_val,
            'indexes':indexes}

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

                if limit[2] == 'weekend':
                    wday = pd.Series(restriction).dt.weekday
                    ii1 = np.where(wday > 4)[0]

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

        Y = np.zeros((lim, int(D / lim)))
        i = 0
        s = 0
        gap=0
        while i <= D:
            if D - i < lim:
                Y = np.delete(Y, s-1, 1)
                gap = D - i
                break
            else:
                Y[:, s] = x[i:(i + lim)]
                i += lim
                s += 1
                if i == D:
                    gap = 0
                    break
        return (Y,gap)


    @staticmethod
    def cortes_onebyone(x, D, lim):
        '''
        :param x:
        :param D: length of data
        :param lim: dimension of the curves
        :return: data divided in curves of specific length but one by one time step
        '''

        Y = np.zeros((lim, D-(lim-1)))
        i = 0
        s = 0
        gap=0
        while i <= D:
            print(i)
            if D - i < lim:
                Y = np.delete(Y, s-1, 1)
                gap=D-i

                break
            else:
                Y[:, s] = x[i:(i + lim)]
                i += 1
                s += 1
                if i == D:
                    gap=0
                    break
        return (Y,gap)

    @staticmethod
    def ts(new_data, look_back, pred_col, names, lag):
        '''
        :param look_back: amount of lags
        :param pred_col: variables to lagged (colum index lagged variables)
        :param names: name variables
        :param lag: name of lag-- variable lag1
        :return: dataframe with lagged data. Intended to have variables to lag in the last columns
        '''
        t = new_data.copy()
        t['id'] = range(0, len(t))
        t = t.iloc[look_back:, :]
        t.set_index('id', inplace=True)
        pred_value = new_data.copy()
        pred_value = pred_value.iloc[:-look_back, pred_col]
        names_lag = [0 for x in range(len(pred_col))]
        for i in range(len(pred_col)):
            names_lag[i]=names[i] + '.'+ str(lag)
        pred_value.columns = names_lag
        pred_value = pd.DataFrame(pred_value)
        pred_value['id'] = range(1, len(pred_value) + 1)
        pred_value.set_index('id', inplace=True)
        final_df = pd.concat([t, pred_value], axis=1)
        return final_df

    def introduce_lags(self, lags, var_lag):
        '''
        Introduction of lags moving the sample

        :param lags: amount of lags for each n_lags selected in MLP
        :param var_lag: label of lagged variables. Amount of variables starting by the end
        :return: data lagged
        '''
        if self.n_lags>0:
            d1 = self.data.copy()
            tt = d1.index
            dim= d1.shape[1]
            selec = range(dim - var_lag, dim)
            try:
                names1= d1.columns[selec]
                for i in range(self.n_lags):
                    d1 = self.ts(d1, lags, selec, names1, i + 1)
                    selec = range(dim, dim +var_lag)
                    dim += var_lag
                    tt =  np.delete(tt, 0)
                self.data = d1
                self.data.index = tt
                self.times = self.data.index
            except:
                raise NameError('Problems introducing time lags')
        else:
            print('No lags selected')

    def adjust_limits(self):
        '''
        Adjust the data or the variable to certain upper or lower limits
        '''
        inf = np.where(self.data.iloc[:,self.pos_y] < self.inf_limit)[0]
        sup = np.where(self.data.iloc[:,self.pos_y] > self.sup_limit)[0]
        if len(inf)>0:
            self.data.iloc[inf, self.pos_y] = np.repeat(self.inf_limit, len(inf))
        if len(sup)>0:
            self.data.iloc[sup, self.pos_y] = np.repeat(self.sup_limit, len(sup))

    def adapt_horizont(self, onebyone):
        '''
        Move the data sample to connected the y with the x based on the future selected
        Adapt to series data!!
        After introduce_lags
        n_steps=0 one by one
        Consideration of horizont 0 o 1
        '''
        if self.horizont==0:
            self.data = self.data
        else:
            X = self.data.drop(self.data.columns[self.pos_y], axis=1)
            y = self.data.iloc[:, self.pos_y]
            y = y.drop(y.index[range(self.horizont)], axis=0)
            X = X.drop(X.index[range(X.shape[0] - self.horizont, X.shape[0])], axis=0)
            self.data = pd.concat([y, X], axis=1)


        if self.type=='series':
            index1 = X.index
            if onebyone==True:
                y,gap = self.cortes_onebyone(y, len(y), self.n_steps)
                y=pd.DataFrame(y.transpose())
                if gap > 0:
                    X = X.drop(X.index[range(X.shape[0] - gap, X.shape[0])], axis=0)
                    index1 = np.delete(index1, range(X.shape[0] - gap, X.shape[0]))
                X = X.drop(X.index[range(X.shape[0] - self.n_steps+1, X.shape[0])], axis=0)
                index1 = np.delete(index1, range(len(index1)-self.n_steps+1, len(index1)))
            else:
                y,gap = self.cortes(y, len(y), self.n_steps)
                y=pd.DataFrame(y.transpose())
                seq = np.arange(0, X.shape[0] - self.n_steps+1, self.n_steps)
                X = X.iloc[seq]
                index1 =index1[seq]
                if gap > 0:
                    fuera = 1 + gap + self.n_steps + self.n_lags
                    X = X.drop(X.index[range(X.shape[0] - 1, X.shape[0])], axis=0)
                    index1 = np.delete(index1, range(X.shape[0] - 1, X.shape[0]))
                else:
                    fuera= 1+self.n_lags
                print('El total a quitar de time_val es:', fuera)
            X = X.reset_index(drop=True)
            print(X.shape)
            print(y.shape)
            X.index = index1
            y.index = index1
            if any(self.pos_y == 0):
                self.data = pd.concat([y, X], axis=1)
            else:
                self.data = pd.concat([X, y], axis=1)
        else:
            print('Not series, nothign to do with n_steps')

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
            if len(self.pos_y) > 1:
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
                        self.data.iloc[:, selec[z]] = scalars[names[i]].transform(pd.DataFrame(d.iloc[:, z]))[:, 0]
                self.scalar_x = scalars
            except:
                raise NameError('Problems with the scalar by groups of variables')
        elif y == True and x == False:
            scalar_y = MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1]))
            scalar_y.fit(pd.DataFrame(self.data.iloc[:, self.pos_y]))

            if len(self.pos_y) > 1:
                self.data.iloc[:, self.pos_y] = scalar_y.transform(pd.DataFrame(self.data.iloc[:, self.pos_y]))
            else:
                self.data.iloc[:, self.pos_y] = scalar_y.transform(pd.DataFrame(self.data.iloc[:, self.pos_y]))[:, 0]

            self.scalar_y = scalar_y

    def missing_values_remove(self):
        self.data = self.data.dropna()
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
    def missing_values_interpolate(self, delete_end, delete_start, mode, limit, order=2):
        '''
        :param delete_end: delete missing data at the last row
        :param delete_start: delete missing data at the first row
        :param mode: linear, spline, polinimial..
        :param limit: amount of missing values accepted
        :param order: if spline or polinomial
        :return: data interpolated
        '''
        dd =  self.data.copy()
        dd = dd.reset_index(drop=True)
        ii = self.data.index
        if delete_end==True:
            while(any(dd.iloc[dd.shape[0]-1].isna())):
                dd.drop(dd.index[dd.shape[0]-1], axis=0, inplace=True)
                ii = np.delete(ii,len(ii)-1)
        if delete_start==True:
            while(any(dd.iloc[0].isna())):
                dd.drop(dd.index[0], axis=0, inplace=True)
                ii = np.delete(ii, 0)
        if mode=='spline' or mode=='polynomial':
            dd = dd.interpolate(method=mode, order=order, axis=0, limit=limit)
        else:
            dd = dd.interpolate(method=mode, axis=0, limit=limit)
        m = dd.isna().sum()
        if np.array(m>0).sum() == 0:
            print('Missing values interpolated')
        else:
            w = np.where(m>0)
            print('CAREFUL!!! There are still missing values; increase the limit? \n'
                  'The variables with missing values are', dd.columns[w])
        self.data = dd
        self.data.index=ii

class MLP(ML):
    def info(self):
        print(('Class to built MLP models. \n'
              'All the parameters comes from the ML class except the activation functions'))
    def __init__(self,data,horizont, scalar_y,scalar_x, zero_problem,limits,extract_cero, times, pos_y, n_lags,n_steps, mask, mask_value, inf_limit,sup_limit,weights, type):
        super().__init__(data,horizont, scalar_y,scalar_x, zero_problem,limits, extract_cero, times, pos_y, n_lags, mask, mask_value, inf_limit,sup_limit)
        self.type = type
        self.weights = weights
        self.n_steps=n_steps
        '''
        horizont: distance to the present
        I want to predict the moment four steps in future
        '''

    @staticmethod
    def complex_mlp(neurons, max_N, max_H):
        '''
        :param max_N: maximun neurons in the network
        :param max_H: maximum hidden layers in the network
        :return: complexity of the model
        '''
        if any(neurons == 0):
            neurons = neurons[neurons > 0]
        u = len(neurons)
        F = 0.25 * (u / max_H) + 0.75 * np.sum(neurons) / max_N
        return F

    @staticmethod
    def mlp_classification(layers, neurons, inputs, outputs, mask, mask__value):
        '''
        :param inputs: amount of inputs
        :param outputs: amount of outputs
        :param mask: True or false
        :return:
        '''
        # activation2 to classification is usually softmax
        try:
            ANN_model = Sequential()
            ANN_model.add(Dense(inputs, kernel_initializer='normal', input_dim=inputs,
                                activation='relu'))
            for i in range(layers):
                ANN_model.add(Dense(neurons[i], kernel_initializer='normal', activation='relu'))
            # The Output Layer :
            ANN_model.add(Dense(outputs, kernel_initializer='normal', activation='softmax'))
            # Compile the network :
            ANN_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            # ANN_model.summary()
            return (ANN_model)
        except:
            raise NameError('Problems building the MLP')

    @staticmethod
    def mlp_regression(layers, neurons,  inputs,mask, mask_value, dropout, outputs):
        '''
        :param inputs:amount of inputs
        :param mask:True or false
        :return: the MLP architecture
        '''
        try:
            ANN_model = Sequential()
            if mask==True and dropout>0:
                ANN_model.add(Masking(mask_value=mask_value, input_shape=np.array([inputs])))
                ANN_model.add(Dense(inputs,kernel_initializer='normal', input_dim=inputs,
                                activation='relu'))
                ANN_model.add(Dropout(dropout))
            elif mask==True and dropout==0:
                ANN_model.add(Masking(mask_value=mask_value, input_shape=np.array([inputs])))
                ANN_model.add(Dense(inputs,kernel_initializer='normal', input_dim=inputs,
                                activation='relu'))
                ANN_model.add(Dropout(dropout))
            elif mask==False and dropout>0:
                ANN_model.add(Dense(inputs, kernel_initializer='normal', input_dim=inputs,
                                activation='relu'))
                ANN_model.add(Dropout(dropout))
            else:
                ANN_model.add(Dense(inputs, kernel_initializer='normal', input_dim=inputs,
                                activation='relu'))
            for i in range(layers):
                if dropout>0:
                    ANN_model.add(Dense(neurons[i], kernel_initializer='normal', activation='relu'))
                    ANN_model.add(Dropout(dropout))
                else:
                    ANN_model.add(Dense(neurons[i], kernel_initializer='normal', activation='relu'))

            # The Output Layer :
            ANN_model.add(Dense(outputs, kernel_initializer='normal', activation='linear'))
            # Compile the network :
            ANN_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
            # ANN_model.summary()
            return(ANN_model)
        except:
            raise NameError('Problems building the MLP')

    @staticmethod
    def mlp_series(layers, neurons,  inputs,mask, mask_value,dropout,n_steps):
        '''
        The main diference is the n_steps. Focused on estimate one variable to several steps in future
        n_steps:amount of time steps estimated
        :param inputs:amount of inputs
        :param mask:True or false
        :return: the MLP architecture
        '''
        try:
            ANN_model = Sequential()
            if mask == True and dropout >0:
                ANN_model.add(Masking(mask_value=mask_value, input_shape=(inputs)))
                ANN_model.add(Dense(inputs, kernel_initializer='normal', input_dim=inputs,
                                    activation='relu'))
                ANN_model.add(Dropout(dropout))
            elif mask == True and dropout ==0:
                ANN_model.add(Masking(mask_value=mask_value, input_shape=(inputs)))
                ANN_model.add(Dense(inputs, kernel_initializer='normal', input_dim=inputs,
                                    activation='relu'))
                ANN_model.add(Dropout(dropout))
            elif mask == False and dropout >0:
                ANN_model.add(Dense(inputs, kernel_initializer='normal', input_dim=inputs,
                                    activation='relu'))
                ANN_model.add(Dropout(dropout))
            else:
                ANN_model.add(Dense(inputs, kernel_initializer='normal', input_dim=inputs,
                                    activation='relu'))
            for i in range(layers):
                if dropout >0:
                    ANN_model.add(Dense(neurons[i], kernel_initializer='normal', activation='relu'))
                    ANN_model.add(Dropout(dropout))
                else:
                    ANN_model.add(Dense(neurons[i], kernel_initializer='normal', activation='relu'))

            # The Output Layer :
            ANN_model.add(Dense(n_steps, kernel_initializer='normal', activation='linear'))
            # Compile the network :
            ANN_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
            # ANN_model.summary()
            return(ANN_model)
        except:
            raise NameError('Problems building the MLP')

    def cv_analysis(self, fold, neurons, pacience, batch, mean_y, dropout, plot, q=[], model=[]):
        '''
        :param fold: divisions in cv analysis
        :param q: a Queue to paralelyse or empty list to do not paralyse
        :param plot: True plots
        :return: predictions, real values, errors and the times needed to train
        '''

        names = self.data.drop(self.data.columns[self.pos_y], axis=1).columns
        print('##########################'
              '################################'
              'CROSS-VALIDATION'
              '#################################'
              '################################')
        layers = len(neurons)
        x = pd.DataFrame(self.data.drop(self.data.columns[self.pos_y], axis=1))
        if self.type == 'series':
            y = pd.DataFrame(self.data.iloc[:, range(self.n_steps)])
        else:
            y = pd.DataFrame(self.data.iloc[:, self.pos_y])
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        res = super().cv_division(x, y, fold)
        x_test = res['x_test']
        x_train = res['x_train']
        x_val = res['x_val']
        y_test = res['y_test']
        y_train = res['y_train']
        y_val = res['y_val']
        indexes = res['indexes']
        times_test = []
        tt = self.times
        for t in range(len(indexes)):
            times_test.append(tt[indexes[t][0]:indexes[t][1]])
        if self.type == 'classification':
            data2 = self.data
            yy = data2.iloc[:, self.pos_y]
            yy = pd.Series(yy, dtype='category')
            n_classes = len(yy.cat.categories.to_list())
            model = self.__class__.mlp_classification(layers, neurons, x_train[0].shape[1], n_classes, self.mask,
                                                      self.mask_value)
            ####################################################################
            # EN PROCESOO ALGÚN DíA !!!!!!!
            ##########################################################################
        else:
            # Train the model
            times = [0 for x in range(fold)]
            cv = [0 for x in range(fold)]
            rmse = [0 for x in range(fold)]
            nmbe = [0 for x in range(fold)]
            predictions = []
            reales = []
            for z in range(fold):
                h_path = Path('./best_models')
                h_path.mkdir(exist_ok=True)
                h = h_path / f'best_{random.randint(0, 1000000)}_model.h5'
                if isinstance(model, list):
                    if self.type == 'regression':
                        model1 = self.__class__.mlp_regression(layers, neurons, x_train[0].shape[1], self.mask,
                                                               self.mask_value, dropout, len(self.pos_y))
                        # Checkpoitn callback
                        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pacience)
                        mc = ModelCheckpoint(str(h), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
                    else:
                        model1 = self.__class__.mlp_series(layers, neurons, x_train[0].shape[1], self.mask,
                                                           self.mask_value, dropout, self.n_steps)
                        # Checkpoitn callback
                        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pacience)
                        mc = ModelCheckpoint(str(h), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
                else:
                    model1 = model
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pacience)
                    mc = ModelCheckpoint(str(h), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
                modelF = model1
                print('Fold number', z)
                x_t = pd.DataFrame(x_train[z]).reset_index(drop=True)
                y_t = pd.DataFrame(y_train[z]).reset_index(drop=True)
                test_x = pd.DataFrame(x_test[z]).reset_index(drop=True)
                test_y = pd.DataFrame(y_test[z]).reset_index(drop=True)
                val_x = pd.DataFrame(x_val[z]).reset_index(drop=True)
                val_y = pd.DataFrame(y_val[z]).reset_index(drop=True)
                time_start = time()
                modelF.fit(x_t, y_t, epochs=2000, validation_data=(test_x, test_y), callbacks=[es, mc],
                           batch_size=batch)
                times[z] = round(time() - time_start, 3)
                y_pred = modelF.predict(val_x)
                y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
                y_real = np.array(self.scalar_y.inverse_transform(val_y))
                if isinstance(self.pos_y, collections.abc.Sized):
                    for t in range(len(self.pos_y)):
                        y_pred[np.where(y_pred[:, t] < self.inf_limit[t])[0], t] = self.inf_limit[t]
                        y_pred[np.where(y_pred[:, t] > self.sup_limit[t])[0], t] = self.sup_limit[t]
                else:
                    y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
                    y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit
                y_predF = y_pred.copy()
                y_predF = pd.DataFrame(y_predF)
                y_predF.index = times_test[z]
                y_realF = y_real.copy()
                y_realF = pd.DataFrame(y_realF)
                y_realF.index = times_test[z]
                predictions.append(y_predF)
                reales.append(y_realF)
                predictions.append(y_predF)
                reales.append(y_realF)
                if self.zero_problem == 'schedule':
                    print('*****Night-schedule fixed******')
                    res = super().fix_values_0(times_test[z],
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
                    if self.type == 'series':
                        y_pred1 = np.concatenate(y_pred1)
                        y_real1 = np.concatenate(y_real1)
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

                    if len(y_pred1) > 0:
                        if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                            if mean_y.size == 0:
                                e = evals(y_pred1, y_real1).variation_rate()
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e)
                                else:
                                    print(e)
                                    print(self.weights)
                                    cv[z] = np.sum(e * self.weights)
                                rmse[z] = np.nan
                                nmbe[z] = np.nan
                            else:
                                e_cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                                e_r = evals(y_pred1, y_real1).rmse()
                                e_n = evals(y_pred1, y_real1).nmbe(mean_y)
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e_cv)
                                    rmse[z] = np.mean(e_r)
                                    nmbe[z] = np.mean(e_n)
                                else:
                                    cv[z] = np.sum(e_cv * self.weights)
                                    rmse[z] = np.sum(e_r * self.weights)
                                    nmbe[z] = np.sum(e_n * self.weights)
                        else:
                            print('Missing values are detected when we are evaluating the predictions')
                            cv[z] = 9999
                            rmse[z] = 9999
                            nmbe[z] = 9999
                    else:
                        raise NameError('Empty prediction')
                elif self.zero_problem == 'radiation':
                    print('*****Night-radiation fixed******')
                    place = np.where(names == 'radiation')[0]
                    scalar_rad = self.scalar_x['radiation']
                    res = super().fix_values_0(scalar_rad.inverse_transform(x_val[z].iloc[:, place]),
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
                    #
                    if self.type == 'series':
                        y_pred1 = np.concatenate(y_pred1)
                        y_real1 = np.concatenate(y_real1)
                    #
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

                    if len(y_pred1) > 0:
                        if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                            if mean_y.size == 0:
                                e = evals(y_pred1, y_real1).variation_rate()
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e)
                                else:
                                    print(e)
                                    print(self.weights)
                                    cv[z] = np.sum(e * self.weights)
                                rmse[z] = np.nan
                                nmbe[z] = np.nan
                            else:
                                e_cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                                e_r = evals(y_pred1, y_real1).rmse()
                                e_n = evals(y_pred1, y_real1).nmbe(mean_y)
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e_cv)
                                    rmse[z] = np.mean(e_r)
                                    nmbe[z] = np.mean(e_n)
                                else:
                                    cv[z] = np.sum(e_cv * self.weights)
                                    rmse[z] = np.sum(e_r * self.weights)
                                    nmbe[z] = np.sum(e_n * self.weights)
                        else:
                            print('Missing values are detected when we are evaluating the predictions')
                            cv[z] = 9999
                            rmse[z] = 9999
                            nmbe[z] = 9999
                    else:
                        raise NameError('Empty prediction')
                else:
                    if self.type == 'series':
                        y_pred = np.concatenate(y_pred)
                        y_real = np.concatenate(y_real)
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

                    if len(y_pred) > 0:
                        if np.sum(np.isnan(y_pred)) == 0 and np.sum(np.isnan(y_real)) == 0:
                            if mean_y.size == 0:
                                e = evals(y_pred, y_real).variation_rate()
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e)
                                else:
                                    print(e)
                                    print(self.weights)
                                    cv[z] = np.sum(e * self.weights)
                                rmse[z] = np.nan
                                nmbe[z] = np.nan
                            else:
                                e_cv = evals(y_pred, y_real).cv_rmse(mean_y)
                                e_r = evals(y_pred, y_real).rmse()
                                e_n = evals(y_pred, y_real).nmbe(mean_y)
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e_cv)
                                    rmse[z] = np.mean(e_r)
                                    nmbe[z] = np.mean(e_n)
                                else:
                                    cv[z] = np.sum(e_cv * self.weights)
                                    rmse[z] = np.sum(e_r * self.weights)
                                    nmbe[z] = np.sum(e_n * self.weights)
                        else:
                            print('Missing values are detected when we are evaluating the predictions')
                            cv[z] = 9999
                            rmse[z] = 9999
                            nmbe[z] = 9999
                    else:
                        raise NameError('Empty prediction')
                if plot == True and len(y_realF.shape) > 1:
                    s = np.max(y_realF.iloc[:, 0]).astype(int) + 15
                    i = np.min(y_realF.iloc[:, 0]).astype(int) - 15
                    a = np.round(cv[z], 2)
                    plt.figure()
                    plt.ylim(i, s)
                    plt.plot(y_realF.iloc[:, 0], color='black', label='Real')
                    plt.plot(y_predF.iloc[:, 0], color='blue', label='Prediction')
                    plt.legend()
                    plt.title("Subsample {} - CV(RMSE)={}".format(z, str(a)))
                    a = 'Subsample-'
                    b = str(z) + '.png'
                    plot_name = a + b
                    plt.show()
                    plt.savefig(plot_name)
                elif plot == True and len(y_realF.shape) < 2:
                    s = np.max(y_realF).astype(int) + 15
                    i = np.min(y_realF).astype(int) - 15
                    a = np.round(cv[z], 2)
                    plt.figure()
                    plt.ylim(i, s)
                    plt.plot(y_realF, color='black', label='Real')
                    plt.plot(y_predF, color='blue', label='Prediction')
                    plt.legend()
                    plt.title("Subsample {} - CV(RMSE)={}".format(z, str(a)))
                    a = 'Subsample-'
                    b = str(z) + '.png'
                    plot_name = a + b
                    plt.show()
                    plt.savefig(plot_name)
            res = {'preds': predictions, 'reals': reales, 'times_test': times_test, 'cv_rmse': cv,
                   'std_cv': np.std(cv),
                   'nmbe': nmbe, 'rmse': rmse,
                   'times_comp': times}
            print(("The model with", layers, " layers", neurons, "neurons and a pacience of", pacience, "has: \n"
                                                                                                        "The average CV(RMSE) is",
                   np.mean(cv), " \n"
                                "The average NMBE is", np.mean(nmbe), "\n"
                                                                      "The average RMSE is", np.mean(rmse), "\n"
                                                                                                            "The average time to train is",
                   np.mean(times)))
            z = Queue()
            if type(q) == type(z):
                # q.put(np.array([np.mean(cv), np.std(cv)]))
                q.put(np.array([np.mean(cv), MLP.complex_mlp(neurons, 2000, 8)]))
            else:
                return (res)

    def optimal_search(self, neurons, paciences,batch, fold,mean_y, parallel,dropout, weights):
        '''
        :param fold: division in cv analyses
        :param parallel: True or false (True to linux)
        :param top: the best options yielded
        :return: the options with their results and the top options
        '''
        error = [0 for x in range(len(neurons) * len(paciences))]
        complexity = [0 for x in range(len(neurons) * len(paciences))]
        options = {'neurons':[], 'pacience':[]}
        w=0
        contador= len(neurons) * len(paciences)-1
        if parallel <2:
            for t in range(len(neurons)):
                print('##################### Option ####################', w)
                neuron = neurons[t]
                for i in range(len(paciences)):
                    options['neurons'].append(neuron)
                    options['pacience'].append(paciences[i])
                    res = self.cv_analysis(fold, neuron , paciences[i],batch,mean_y,dropout,False)
                    error[w]=np.mean(res['cv_rmse'])
                    complexity[w]=MLP.complex_mlp(neuron,2000,8)
                    w +=1
        elif parallel>=2:
            processes = []
            res2 = []
            dev2 = []
            z = 0
            q = Queue()
            for t in range(len(neurons)):
                neuron = neurons[t]
                for i in range(len(paciences)):
                    print('##################### Option ####################', w)
                    options['neurons'].append(neuron)
                    options['pacience'].append(paciences[i])
                    if z < parallel and w < contador:
                        multiprocessing.set_start_method('fork')
                        p = Process(target=self.cv_analysis,
                                    args=(fold, neuron, paciences[i], batch, mean_y,dropout,False, q))
                        p.start()
                        processes.append(p)
                        z1 =z+ 1
                    elif z == parallel and w < contador:
                        p.close()
                        for p in processes:
                            p.join()
                        for v in range(len(processes)):
                            res2.append(q.get()[0])
                            res2.append(q.get()[1])
                        processes = []
                        # multiprocessing.set_start_method('fork')
                        # multiprocessing.set_start_method('spawn')
                        q = Queue()
                        p = Process(target=self.cv_analysis,
                                    args=(fold, neuron, paciences[i], batch, mean_y,dropout,False, q))
                        p.start()
                        processes.append(p)
                        z1 = 1
                    elif w == contador:
                        p = Process(target=self.cv_analysis,
                                    args=(fold, neuron, paciences[i], batch, mean_y,dropout,False, q))
                        p.start()
                        processes.append(p)
                        p.close()
                        for p in processes:
                            p.join()
                            #res2.append(q.get())
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
        print(r1)

        scal_cv = MinMaxScaler(feature_range=(0, 1))
        scal_com = MinMaxScaler(feature_range=(0, 1))

        scal_cv.fit(np.array(r1).reshape(-1, 1))
        scal_com.fit(np.array(d1).reshape(-1, 1))

        cv = scal_cv.transform(np.array(r1).reshape(-1, 1))
        com = scal_com.transform(np.array(d1).reshape(-1, 1))

        r_final = np.array([cv[:, 0], com[:, 0]]).T

        I = get_decomposition("aasf", beta=5).do(r_final, weights).argmin()
        #I = get_decomposition("pbi").do(r_final, weights).argmin()

        top_result = {'error': [], 'complexity': [], 'nuerons': [], 'pacience': []}
        top_result['error'] = r1[I]
        top_result['complexity'] = d1[I]
        top_result['neurons'] = options['neurons'][I]
        top_result['pacience'] = options['pacience'][I]

        print(top_result['error'])
        print(top_result['complexity'])
        print(top_result['neurons'])
        print(top_result['pacience'])

        np.savetxt('objectives_selected_brute.txt', np.array([top_result['error'],top_result['complexity']]))
        np.savetxt('x_selected_brute.txt', np.concatenate((top_result['neurons'],np.array([top_result['pacience']]))))


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
        return(res)

    def train(self, type,neurons, pacience, batch,data_train, data_test, dropout, save_model, model=[]):
        '''
        :param x_train: x to train
        :param x_test: x to early stopping
        :param y_train: y to train
        :param y_test: y to early stopping
        :param model: loaded model
        :return: trained model and the time needed to train
        '''
        data_train = pd.DataFrame(data_train)
        data_test = pd.DataFrame(data_test)

        x_train = data_train.drop(data_train.columns[self.pos_y], axis=1)
        x_test = data_test.drop(data_test.columns[self.pos_y], axis=1)
        y_train = data_train.iloc[:,self.pos_y]
        y_test = data_test.iloc[:,self.pos_y]

        from datetime import datetime

        now = str(datetime.now().microsecond)

        if type=='regression':
            if isinstance(model, list):
                layers = len(neurons)
                if self.type=='series':
                    model = self.__class__.mlp_series(layers, neurons,x_train.shape[1], self.mask, self.mask_value,dropout, self.n_steps)
                else:
                    model = self.__class__.mlp_regression(layers, neurons, x_train.shape[1], self.mask, self.mask_value, dropout, len(self.pos_y))
            else:
                model=model
            # Checkpoint callback
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pacience)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            time_start = time()
            history = model.fit(x_train, y_train, epochs=2000, validation_data=(x_test, y_test),
                      callbacks=[es, mc],batch_size=batch)
            times = round(time() - time_start, 3)
        else:

            'clasification'

        if save_model==True:
            name='mlp'+now+'.h5'
            model.save(name, save_format='h5')
        res = {'model':model, 'times':times, 'history':history}
        return(res)

    def predict(self, model, val, mean_y, times, plotting):
        '''
        :param model: trained model
        :param x_val: x to predict
        :param y_val: y to predict
        if mean_y is empty a variation rate will be applied as cv in result. The others relative will be nan
        :return: predictions with the errors depending of zero_problem
        '''

        y_val = val.iloc[:, self.pos_y]
        x_val = val.drop(val.columns[self.pos_y], axis=1)

        x_val = x_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        y_pred = model.predict(pd.DataFrame(x_val))
        y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
        y_real = np.array(self.scalar_y.inverse_transform(y_val))

        if len(self.pos_y) > 1:
            for t in range(len(self.pos_y)):
                y_pred[np.where(y_pred[:, t] < self.inf_limit[t])[0], t] = self.inf_limit[t]
                y_pred[np.where(y_pred[:, t] > self.sup_limit[t])[0], t] = self.sup_limit[t]
            y_predF = pd.DataFrame(y_pred.copy())
            y_realF = pd.DataFrame(y_real).copy()

        elif self.n_steps > 1:
            for t in self.n_steps:
                y_pred[np.where(y_pred[:, t] < self.inf_limit[t])[0], t] = self.inf_limit
                y_pred[np.where(y_pred[:, t] > self.sup_limit[t])[0], t] = self.sup_limit
            y_predF = pd.DataFrame(np.concatenate(y_pred.copy()))
            y_realF = pd.DataFrame(np.concatenate(y_real).copy())
        else:
            y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
            y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit
            y_predF = pd.DataFrame(y_pred.copy())
            y_realF = pd.DataFrame(y_real).copy()

        y_predF.index = times
        y_realF.index = y_predF.index

        if self.zero_problem == 'schedule':
            print('*****Night-schedule fixed******')
            res = super().fix_values_0(times, self.zero_problem, self.limits)
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

                if self.type == 'series':
                    y_pred1 = np.concatenate(y_pred1)
                    y_real1 = np.concatenate(y_real1)

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

            if len(y_pred1) > 1:
                if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
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
                    res = {'y_pred': y_predF, 'y_real':y_realF,'cv_rmse': cv, 'nmbe': nmbe,
                           'rmse': rmse, 'r2': r2}
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
            else:
                raise NameError('Empty prediction')
        elif self.zero_problem == 'radiation':
            print('*****Night-radiation fixed******')
            place = np.where(x_val.columns == 'radiation')[0]
            scalar_x = self.scalar_x
            scalar_rad = scalar_x['radiation']
            res = super().fix_values_0(scalar_rad.inverse_transform(x_val.iloc[:, place]),
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

                if self.type == 'series':
                    y_pred1 = np.concatenate(y_pred1)
                    y_real1 = np.concatenate(y_real1)

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
            if len(y_pred1) > 1:
                if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
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
                           'rmse': rmse, 'r2': r2}
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
            else:
                raise NameError('Empty prediction')
        else:
            if self.type == 'series':
                y_pred = np.concatenate(y_pred)
                y_real = np.concatenate(y_real)

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

            if len(y_pred) > 1:
                if np.sum(np.isnan(y_pred)) == 0 and np.sum(np.isnan(y_real)) == 0:
                    if mean_y.size == 0:
                        e = evals(y_pred, y_real).variation_rate()
                        if isinstance(self.weights, list):
                            cv = np.mean(e)
                        else:
                            cv = np.sum(e * self.weights)
                        rmse = np.nan
                        nmbe = np.nan
                        r2 = np.nan
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
                           'rmse': rmse, 'r2': r2}
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
                    res = {'y_pred': y_predF, 'cv_rmse': cv, 'nmbe': nmbe,
                           'rmse': rmse, 'r2': r2}
            else:
                raise NameError('Empty prediction')

        if plotting == True:
            a = np.round(cv, 2)
            up = int(np.max(y_realF)) + int(np.max(y_realF) / 4)
            low = int(np.min(y_realF)) - int(np.min(y_realF) / 4)
            plt.figure()
            plt.ylim(low, up)
            plt.plot(y_realF, color='black', label='Real')
            plt.plot(y_predF, color='blue', label='Prediction')
            plt.legend()
            plt.title("CV(RMSE)={}".format(str(a)))
            plt.savefig('plot1.png')

        return res

    def nsga2_individual(self, med, contador, n_processes, l_dense, batch, pop_size, tol, xlimit_inf,
                         xlimit_sup,dropout, dictionary, weights):
        '''
        :param med:
        :param contador: a operator to count the attempts
        :param n_processes: how many processes are parallelise
        :param l_dense:maximun number of layers dense
        :param batch: batch size
        :param pop_size: population size selected for NSGA2
        :param tol: tolearance selected to terminate the process
        :param xlimit_inf: array with the lower limits to the neuron  lstm , neurons dense and pacience
        :param xlimit_sup:array with the upper limits to the neuron  lstm , neurons dense and pacience
        :param dictionary: dictionary to stored the options tested
        :return: options in Pareto front, the optimal selection and the total results
        '''

        print('DATA is', type(self.data))
        if n_processes > 1:
            pool = multiprocessing.Pool(n_processes)
            problem = MyProblem_mlp(self.horizont, self.scalar_y, self.zero_problem,self.extract_cero, self.limits, self.times, self.pos_y,
                                self.mask,
                                self.mask_value, self.n_lags, self.inf_limit, self.sup_limit,
                                self.type, self.data,self.scalar_x,
                                med, contador,len(xlimit_inf), l_dense, batch, xlimit_inf, xlimit_sup,dropout,dictionary,self.weights,runner = pool.starmap,func_eval=starmap_parallelized_eval)
        else:
            problem = MyProblem_mlp(self.horizont, self.scalar_y, self.zero_problem, self.extract_cero, self.limits, self.times, self.pos_y,
                                self.mask,
                                self.mask_value, self.n_lags, self.inf_limit, self.sup_limit,
                                self.type, self.data,self.scalar_x,
                                med, contador, len(xlimit_inf), l_dense, batch, xlimit_inf, xlimit_sup,dropout, dictionary, self.weights)
        algorithm = NSGA2(pop_size=pop_size, repair=MyRepair(l_dense), eliminate_duplicates=True,
                          sampling=get_sampling("int_random"),
                          # sampling =g,
                          # crossover=0.9,
                          # mutation=0.1)
                          crossover=get_crossover("int_sbx",prob=0.95),
                          mutation=get_mutation("int_pm", prob=0.4))
        termination = MultiObjectiveSpaceToleranceTermination(tol=tol,
                                                              n_last=int(pop_size / 2), nth_gen=int(pop_size / 4),
                                                              n_max_gen=None,
                                                              n_max_evals=5000)
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

            I = get_decomposition("pbi").do(r_final, weights).argmin()

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
            plt.savefig('optimisation_plot.png')
        else:
            obj_T = res.F
            struct_T = res.X
            obj = res.F
            struct = res.X
        print('The number of evaluations were:', contador)
        if n_processes > 1:
            pool.close()
        else:
            pass
        return (obj, struct, obj_T, struct_T, res,contador)

    def optimal_search_nsga2(self, l_dense, batch, pop_size, tol, xlimit_inf, xlimit_sup, mean_y,dropout, parallel,weights):
        '''
        :param l_dense: maximun layers dense
        :param batch: batch size
        :param pop_size: population size for RVEA
        :param tol: tolerance to built the pareto front
        :param xlimit_inf: array with lower limits for neurons lstm, dense and pacience
        :param xlimit_sup: array with upper limits for neurons lstm, dense and pacience
        :param parallel: how many processes are parallelise
        if mean_y is empty a variation rate will be applied
        :return: the options selected for the pareto front, the optimal selection and the total results
        '''
        manager = multiprocessing.Manager()
        dictionary = manager.dict()
        contador = manager.list()
        contador.append(0)
        print('Start the optimization!!!!!')
        obj, x_obj, obj_total, x_obj_total, res,evaluations = self.nsga2_individual(mean_y, contador, parallel, l_dense,
                                                                            batch, pop_size, tol, xlimit_inf,
                                                                            xlimit_sup, dropout,dictionary, weights)
        np.savetxt('objectives_selected.txt', obj)
        np.savetxt('x_selected.txt', x_obj)
        np.savetxt('objectives.txt', obj_total)
        np.savetxt('x.txt', x_obj_total)
        np.savetxt('evaluations.txt', evaluations)

        print('Process finished!!!')
        print('The selection is', x_obj, 'with a result of', obj)
        res = {'total_x': x_obj_total, 'total_obj': obj_total, 'opt_x': x_obj, 'opt_obj': obj, 'res': res,'evaluations':evaluations}
        return res

    def rnsga2_individual(self, med, contador, n_processes, l_dense, batch, pop_size, tol, xlimit_inf,
                         xlimit_sup,dropout, dictionary, weights,epsilon):
        '''
        :param med:
        :param contador: a operator to count the attempts
        :param n_processes: how many processes are parallelise
        :param l_dense:maximun number of layers dense
        :param batch: batch size
        :param pop_size: population size selected for NSGA2
        :param tol: tolearance selected to terminate the process
        :param xlimit_inf: array with the lower limits to the neuron  lstm , neurons dense and pacience
        :param xlimit_sup:array with the upper limits to the neuron  lstm , neurons dense and pacience
        :param dictionary: dictionary to stored the options tested
        :return: options in Pareto front, the optimal selection and the total results
        '''
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.factory import get_problem, get_visualization, get_decomposition
        from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
        from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
        from pymoo.optimize import minimize
        from pymoo.core.problem import starmap_parallelized_eval
        print('DATA is', type(self.data))
        if n_processes > 1:
            pool = multiprocessing.Pool(n_processes)
            problem = MyProblem_mlp(self.horizont, self.scalar_y, self.zero_problem,self.extract_cero, self.limits, self.times, self.pos_y,
                                self.mask,
                                self.mask_value, self.n_lags, self.inf_limit, self.sup_limit,
                                self.type, self.data,self.scalar_x,
                                med, contador,len(xlimit_inf), l_dense, batch, xlimit_inf, xlimit_sup,dropout,dictionary,self.weights,runner = pool.starmap,func_eval=starmap_parallelized_eval)
        else:
            problem = MyProblem_mlp(self.horizont, self.scalar_y, self.zero_problem,self.extract_cero, self.limits, self.times, self.pos_y,
                                self.mask,
                                self.mask_value, self.n_lags, self.inf_limit, self.sup_limit,
                                self.type, self.data,self.scalar_x,
                                med, contador, len(xlimit_inf), l_dense, batch, xlimit_inf, xlimit_sup,dropout, dictionary, self.weights)

        ref_points = np.array([[0.3, 0.1], [0.1, 0.3]])

        algorithm = RNSGA2(ref_points, pop_size=pop_size, sampling=get_sampling("int_random"),
                          crossover=get_crossover("int_sbx", prob=0.95),
                          mutation=get_mutation("int_pm", prob=0.4),
                           normalization='front',
                           extreme_points_as_reference_points=False,
                           weights=weights,
                           epsilon=epsilon)

        termination = MultiObjectiveSpaceToleranceTermination(tol=tol,
                                                              n_last=int(pop_size / 2), nth_gen=int(pop_size / 4),
                                                              n_max_gen=None,
                                                              n_max_evals=5000)
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

            plt.figure(figsize=(10, 7))
            plt.scatter(r_final[:, 0], r_final[:, 1], color='black')
            plt.xlabel('Normalised CV (RMSE)', fontsize=20, labelpad=10)
            plt.ylabel('Normalised Complexity', fontsize=20, labelpad=10)
            plt.scatter(r_final[I, 0], r_final[I, 1], s=200, color='red', alpha=1, marker='o', facecolors='none',
                        label='Optimum')
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(borderpad=1, fontsize=15)
            plt.savefig('optimisation_plotR.png')
        else:
            obj_T = res.F
            struct_T = res.X
            obj = res.F
            struct = res.X
        print('The number of evaluations were:', contador)
        if n_processes > 1:
            pool.close()
        else:
            pass
        return (obj, struct, obj_T, struct_T, res,contador)

    def optimal_search_rnsga2(self, l_dense, batch, pop_size, tol, xlimit_inf, xlimit_sup, mean_y,dropout, parallel,weights,epsilon=0.01):
        '''
        :param l_dense: maximun layers dense
        :param batch: batch size
        :param pop_size: population size for RVEA
        :param tol: tolerance to built the pareto front
        :param xlimit_inf: array with lower limits for neurons lstm, dense and pacience
        :param xlimit_sup: array with upper limits for neurons lstm, dense and pacience
        :param parallel: how many processes are parallelise
        if mean_y is empty a variation rate will be applied
        :return: the options selected for the pareto front, the optimal selection and the total results
        '''
        manager = multiprocessing.Manager()
        dictionary = manager.dict()
        contador = manager.list()
        contador.append(0)
        print('Start the optimization!!!!!')
        obj, x_obj, obj_total, x_obj_total, res,evaluations = self.rnsga2_individual(mean_y, contador, parallel, l_dense,
                                                                            batch, pop_size, tol, xlimit_inf,
                                                                            xlimit_sup, dropout,dictionary, weights,epsilon)
        np.savetxt('objectives_selectedR.txt', obj)
        np.savetxt('x_selectedR.txt', x_obj)
        np.savetxt('objectivesR.txt', obj_total)
        np.savetxt('xR.txt', x_obj_total)
        np.savetxt('evaluationsR.txt', evaluations)

        print('Process finished!!!')
        print('The selection is', x_obj, 'with a result of', obj)
        res = {'total_x': x_obj_total, 'total_obj': obj_total, 'opt_x': x_obj, 'opt_obj': obj, 'res': res,'evaluations':evaluations}
        return res


from pymoo.core.repair import Repair
class MyRepair(Repair):
    def info(self):
        print('Class defining a function to repair the possible error of the genetic algorithm. If a layer is zero the next layer cannot have positive neurons')
    def __init__(self,l_dense):
        self.l_dense = l_dense
    def _do(self, problem, pop, **kwargs):
        for k in range(len(pop)):
            x = pop[k].X
            x2 = x[range(self.l_dense)]
            r_dense = MyProblem_mlp.bool4(x, self.l_dense)
            if len(r_dense) == 1:
                if r_dense == 0:
                    pass
                elif r_dense != 0:
                    x2[r_dense] = 0
            elif len(r_dense) > 1:
                x2[r_dense] = 0
            x = np.concatenate((x2, np.array([x[len(x) - 1]])))
            pop[k].X = x
        return pop
from pymoo.core.problem import ElementwiseProblem
class MyProblem_mlp(ElementwiseProblem):
    def info(self):
        print('Class to create a specific problem to use NSGA2 in architectures search.')
    def __init__(self, horizont, scalar_y, zero_problem, extract_cero,limits, times, pos_y, mask, mask_value, n_lags, inf_limit,
                 sup_limit, type, data,scalar_x, med, contador,
                 n_var,l_dense, batch, xlimit_inf, xlimit_sup,dropout, dictionary,weights, **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_constr=1,
                         xl=xlimit_inf,
                         xu=xlimit_sup,
                         type_var=np.int,
                         # elementwise_evaluation=True,
                         **kwargs)
        self.data = data
        self.horizont = horizont
        self.scalar_y = scalar_y
        self.scalar_x = scalar_x
        self.zero_problem = zero_problem
        self.extract_cero = extract_cero
        self.limits = limits
        self.times = times
        self.pos_y = pos_y
        self.mask = mask
        self.mask_value = mask_value
        self.n_lags = n_lags
        self.inf_limit = inf_limit
        self.sup_limit = sup_limit
        self.type = type
        self.med = med
        self.contador = contador
        self.l_dense = l_dense
        self.batch = batch
        self.xlimit_inf = xlimit_inf
        self.xlimit_sup = xlimit_sup
        self.dropout = dropout
        self.n_var = n_var
        self.dictionary = dictionary
        self.weights = weights

    def cv_opt(self,data, fold, neurons, pacience, batch, mean_y, dictionary):
        '''
        :param fold:assumed division of the sample for cv
        :param dictionary: dictionary to fill with the options tested
        :param q:operator to differentiate when there is parallelisation and the results must be a queue
        :return: cv(rmse) and complexity of the model tested
        if mean_y is empty a variation rate will be applied
        '''

        from pathlib import Path
        import random

        h_path = Path('./best_models')
        h_path.mkdir(exist_ok=True)
        h = h_path / f'best_{random.randint(0, 1000000)}_model.h5'

        name1 = tuple(np.concatenate((neurons, np.array([pacience]))))
        try:
            a0, a1 = dictionary[name1]
            return a0, a1
        except KeyError:
            pass
        cvs = [0 for x in range(fold)]
        print(type(data))
        names = self.data.columns
        names = np.delete(names, self.pos_y)
        neurons=neurons[neurons>0]
        layers = len(neurons)
        y = self.data.iloc[:,self.pos_y]
        x =self.data.drop(self.data.columns[self.pos_y],axis=1)
        res = MLP.cv_division(x, y, fold)
        x_test = res['x_test']
        x_train = res['x_train']
        x_val = res['x_val']
        y_test = res['y_test']
        y_train = res['y_train']
        y_val = res['y_val']
        indexes = res['indexes']
        times_test = []
        tt = self.times
        for t in range(len(indexes)):
            times_test.append(tt[indexes[t][0]:indexes[t][1]])
        if self.type=='classification':
            data2 = self.data
            yy = data2.iloc[:, self.pos_y]
            yy = pd.Series(yy, dtype='category')
            n_classes = len(yy.cat.categories.to_list())
            model = MLP.mlp_classification(layers, neurons, x_train[0].shape[1], n_classes, self.mask, self.mask_value)
            ####################################################################
            # EN PROCESOO ALGÚN DíA !!!!!!!
            ##########################################################################
        else:
            if self.type=='regression':
                model = MLP.mlp_regression(layers, neurons, x_train[0].shape[1], self.mask, self.mask_value, self.dropout,len(self.pos_y))
            elif self.type=='series':
                model = MLP.mlp_series(layers, neurons, x_train[0].shape[1], self.mask, self.mask_value,
                                                      self.dropout, self.n_steps)
            # Checkpoitn callback
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pacience)
            mc = ModelCheckpoint(str(h), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            # Train the model
            for z in range(fold):
                print('Fold number', z)
                x_t = pd.DataFrame(x_train[z]).reset_index(drop=True)
                y_t = pd.DataFrame(y_train[z]).reset_index(drop=True)
                test_x = pd.DataFrame(x_test[z]).reset_index(drop=True)
                test_y = pd.DataFrame(y_test[z]).reset_index(drop=True)
                val_x = pd.DataFrame(x_val[z]).reset_index(drop=True)
                val_y = pd.DataFrame(y_val[z]).reset_index(drop=True)
                model.fit(x_t, y_t, epochs=2000, validation_data=(test_x, test_y), callbacks=[es, mc], batch_size=batch)
                y_pred = model.predict(val_x)
                y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
                y_real = val_y
                y_real = np.array(self.scalar_y.inverse_transform(y_real))
                if isinstance(self.pos_y, collections.abc.Sized):
                    for t in range(len(self.pos_y)):
                        y_pred[np.where(y_pred[:, t] < self.inf_limit[t])[0], t] = self.inf_limit[t]
                        y_pred[np.where(y_pred[:, t] > self.sup_limit[t])[0], t] = self.sup_limit[t]
                    y_real = y_real
                else:
                    y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
                    y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit
                    y_real = y_real.reshape(-1, 1)

                y_predF = y_pred.copy()
                y_predF = pd.DataFrame(y_predF)
                y_predF.index = times_test[z]
                y_realF = y_real.copy()
                y_realF = pd.DataFrame(y_realF)
                y_realF.index = times_test[z]
                if self.zero_problem == 'schedule':
                    print('*****Night-schedule fixed******')
                    res = ML.fix_values_0(times_test[z],
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

                    if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                        if mean_y.size == 0:
                            e=evals(y_pred1, y_real1).variation_rate()
                            if isinstance(self.weights, list):
                                cvs[z]=np.mean(e)
                            else:
                                cvs[z] = np.sum(e * self.weights)

                        else:
                            e = evals(y_pred1, y_real1).cv_rmse(mean_y)
                            if isinstance(self.weights, list):
                                cvs[z] = np.mean(e)
                            else:
                                cvs[z] = np.sum(e * self.weights)

                elif self.zero_problem == 'radiation':

                    print('*****Night-radiation fixed******')
                    place = np.where(names == 'radiation')[0]
                    scalar_rad = self.scalar_x['radiation']
                    res = ML.fix_values_0(scalar_rad.inverse_transform(x_val[z].iloc[:, place]),
                                               self.zero_problem, self.limits)
                    index_rad = res['indexes_out']
                    index_rad2 = np.where(y_real <= self.inf_limit)[0]
                    index_rad = np.union1d(np.array(index_rad), np.array(index_rad2))

                    if len(index_rad) > 0 and self.horizont == 0:
                        y_pred1 = np.delete(y_pred, index_rad, 0)
                        y_real1 = np.delete(y_real, index_rad, 0)
                    elif len(index_rad) > 0 and self.horizont > 0:
                        y_pred1 = np.delete(y_pred, index_rad - self.horizont, 0)
                        y_real1 = np.delete(y_real, index_rad - self.horizont, 0)
                    else:
                        y_pred1 = y_pred
                        y_real1 = y_real
                        
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

                    if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                        if mean_y.size == 0:
                            e=evals(y_pred1, y_real1).variation_rate()
                            if isinstance(self.weights, list):
                                cvs[z] = np.mean(e)
                            else:
                                cvs[z] = np.sum(e * self.weights)

                        else:
                            e = evals(y_pred1, y_real1).cv_rmse(mean_y)
                            if isinstance(self.weights, list):
                                cvs[z] = np.mean(e)
                            else:
                                cvs[z] = np.sum(e * self.weights)

                    else:
                        print('Missing values are detected when we are evaluating the predictions')
                        cvs[z] = 9999
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
                            
                    if np.sum(np.isnan(y_pred)) == 0 and np.sum(np.isnan(y_real)) == 0:
                        if mean_y.size == 0:
                            e=evals(y_pred, y_real).variation_rate()
                            if isinstance(self.weights, list):
                                cvs[z] = np.mean(e)
                            else:
                                print(e)
                                print(self.weights)
                                cvs[z] = np.sum(e * self.weights)

                        else:
                            e = evals(y_pred, y_real).cv_rmse(mean_y)
                            if isinstance(self.weights, list):
                                cvs[z] = np.mean(e)
                            else:
                                cvs[z] = np.sum(e * self.weights)

                    else:
                        print('Missing values are detected when we are evaluating the predictions')
                        cvs[z] = 9999

            complexity = MLP.complex_mlp(neurons, 2000, 8)
            dictionary[name1] = np.mean(cvs), complexity
            print(dictionary[name1] )
            res_final = {'cvs': np.mean(cvs), 'complexity': complexity}
            return res_final['cvs'], res_final['complexity']

    @staticmethod
    def bool4(x,l_dense):
        '''
        :x: neurons options
        l_dense: number of values that represent dense neurons
        :return: 0 if the constraint is fulfilled
        '''
        #
        x2 = x[range(l_dense)]
        #
        if len(x2) == 2:
            if x2[0] == 0 and x2[1] > 0:
                a_dense = np.array([1])
            else:
                a_dense = np.array([0])
        elif len(x2) == 3:
            if x2[0] == 0 and x2[1] > 0:
                a_dense = np.array([1])
                if x2[2] > 0:
                    a_dense = np.array([1, 2])
            elif x2[1] == 0 and x2[2] > 0:
                a_dense = np.array([2])
            else:
                a_dense = np.array([0])
        elif len(x2) == 4:
            if x2[0] == 0 and x2[1] > 0:
                a_dense = np.array([1])
                if x2[2] > 0:
                    a_dense = np.array([1, 2])
            elif x2[0] == 0 and x2[2] > 0:
                a_dense = np.array([2])
            elif x2[0] == 0 and x2[3] > 0:
                a_dense = np.array([3])
            elif x2[1] == 0 and x2[2] > 0:
                a_dense = np.array([2])
                if x2[3] > 0:
                    a_dense = np.array([2, 3])
            elif x2[1] == 0 and x2[3] > 0:
                a_dense = np.array([3])
            elif x2[2] == 0 and x2[3] > 0:
                a_dense = np.array([3])
            else:
                a_dense = np.array([0])
        else:
            raise NameError('Option not considered')
        #
        return a_dense
    #
    def _evaluate(self, x, out, *args, **kwargs):
        g1 = MyProblem_mlp.bool4(np.delete(x, len(x) - 1),self.l_dense)
        out["G"] = g1
        print(x)
        n_dense = x[range(self.l_dense)]*20
        n_pacience = x[len(x) - 1]*20
        f1, f2 = self.cv_opt(self.data, 3, n_dense, n_pacience, self.batch, self.med, self.dictionary)
        print(
            '\n ############################################## \n ############################# \n ########################## EvaluaciÃ³n ',
            self.contador, '\n #########################')
        self.contador[0] += 1
        out["F"] = np.column_stack([f1, f2])


class SVM(ML):
    def info(self):
        print(('Class to built SVM models. \n'))
    def __init__(self,data,horizont, scalar_y,scalar_x, zero_problem,limits,extract_cero, times, pos_y, n_lags, mask, mask_value, inf_limit,sup_limit,weights, type):
        super().__init__(data,horizont, scalar_y,scalar_x, zero_problem, limits, extract_cero, times, pos_y, n_lags, mask, mask_value, inf_limit,sup_limit)
        self.type = type
        self.weights = weights
        '''
        I want to predict the moment four steps in future
        '''

    @staticmethod
    def complex_svm(C_svm,epsilon_svm, max_C, max_epsilon):
        '''
        :param max_N: maximun neurons in the network
        :param max_H: maximum hidden layers in the network
        :return: complexity of the model
        '''

        F = 0.75 * (C_svm / max_C) + 0.25 * (epsilon_svm/max_epsilon)
        return F


    @staticmethod
    def svm_cv_division(x,y, fold):
        '''
        Division de la muestra en trozos según fold para datos normales y no recurrentes

        :param fold: division for cv_analysis
        :return: data divided into train, test and validations in addition to the indexes division for a CV analysis
        '''
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        X_test = []
        X_train = []
        Y_test = []
        Y_train = []

        step = int(x.shape[0]/fold)
        w = 0
        w2 = step
        indexes = []
        try:
            while w2 < x.shape[0]:
                X_test.append(x.iloc[range(w,w2)])
                X_train.append(x.drop(range(w,w2)))

                Y_test.append(y.iloc[range(w, w2)])
                Y_train.append(y.drop(range(w, w2)))
                indexes.append(np.array([w, w2]))
                w = w2
                w2 += step
                if w2 > x.shape[0] and w < x.shape[0]:
                    w2 = x.shape[0]
        except:
            raise NameError('Problems with the sample division in the cv classic')

        res = {'x_test': X_test, 'x_train':X_train, 'y_test':Y_test, 'y_train':Y_train,
            'indexes':indexes}

        return res

    def SVM_training(self,layers, neurons, inputs, outputs, mask, mask__value):
        'WORK IN PROGRESS'

    @staticmethod
    def SVR_training(data_train,pos_y,C,epsilon, tol, save_model, model=[]):
        now = str(datetime.now().microsecond)
        print(data_train)

        data_train = pd.DataFrame(data_train)
        x_train = data_train.drop(data_train.columns[pos_y], axis=1)
        y_train = data_train.iloc[:,pos_y]

        if isinstance(model, list):
            if len(pos_y)>1:
                model=MultiOutputRegressor(svm.LinearSVR(random_state=None, dual=True, C=C, tol=tol, epsilon=epsilon)
)
            else:
                model = svm.LinearSVR(random_state=None, dual=True, C=C, tol=tol, epsilon=epsilon)
        else:
            model = model

        history = model.fit(x_train, y_train)

        if save_model==True:
            name='mlp'+now+'.h5'
            model.save(name, save_format='h5')
        res = {'model':model,  'history':history}
        return(res)

    def predict(self, model,val,mean_y, times,plotting):
        '''
        :param model: trained model
        :param x_val: x to predict
        :param y_val: y to predict
        if mean_y is empty a variation rate will be applied as cv in result. The others relative will be nan
        :return: predictions with the errors depending of zero_problem
        '''

        y_val = val.iloc[:,self.pos_y]
        x_val = val.drop(val.columns[self.pos_y], axis=1)

        x_val=x_val.reset_index(drop=True)
        y_val=y_val.reset_index(drop=True)
        y_pred = model.predict(pd.DataFrame(x_val))
        y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
        y_real = np.array(self.scalar_y.inverse_transform(y_val))

        if len(self.pos_y)>1:
            for t in range(len(self.pos_y)):
                y_pred[np.where(y_pred[:,t] < self.inf_limit[t])[0],t] = self.inf_limit[t]
                y_pred[np.where(y_pred[:,t] > self.sup_limit[t])[0],t] = self.sup_limit[t]
            y_predF = pd.DataFrame(y_pred.copy())
            y_realF = pd.DataFrame(y_real).copy()

        else:
            y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
            y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit
            y_predF = pd.DataFrame(y_pred.copy())
            y_realF = pd.DataFrame(y_real).copy()

        y_predF.index = times
        y_realF.index = y_predF.index

        if self.zero_problem == 'schedule':
            print('*****Night-schedule fixed******')
            res = super().fix_values_0(times,  self.zero_problem, self.limits)
            index_hour = res['indexes_out']

            if len(y_pred)<=1:
                y_pred1= np.nan
                y_real1=y_real
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

            if len(y_pred1)>1:
                if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
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
                           'rmse': rmse, 'r2': r2}
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
            else:
                raise NameError('Empty prediction')
        elif self.zero_problem == 'radiation':
            print('*****Night-radiation fixed******')
            place = np.where(x_val.columns == 'radiation')[0]
            scalar_x = self.scalar_x
            scalar_rad = scalar_x['radiation']
            res = super().fix_values_0(scalar_rad.inverse_transform(x_val.iloc[:, place]),
                                          self.zero_problem, self.limits)
            index_rad = res['indexes_out']
            index_rad2 = np.where(y_real <= self.inf_limit)[0]
            index_rad = np.union1d(np.array(index_rad), np.array(index_rad2))

            if len(y_pred)<=1:
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

            if len(y_pred1)>1:
                if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
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
                           'rmse': rmse, 'r2': r2}
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
            else:
                raise NameError('Empty prediction')
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

            if len(y_pred)>1:
                if np.sum(np.isnan(y_pred)) == 0 and np.sum(np.isnan(y_real)) == 0:
                    if mean_y.size == 0:
                        e = evals(y_pred, y_real).variation_rate()
                        if isinstance(self.weights, list):
                            cv = np.mean(e)
                        else:
                            cv = np.sum(e * self.weights)
                        rmse = np.nan
                        nmbe = np.nan
                        r2 = np.nan
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
                           'rmse': rmse, 'r2': r2}
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
                    res = {'y_pred': y_predF, 'cv_rmse': cv, 'nmbe': nmbe,
                           'rmse': rmse, 'r2': r2}
            else:
                raise NameError('Empty prediction')

        if plotting==True:
            a = np.round(cv, 2)
            up = int(np.max(y_realF)) + int(np.max(y_realF) / 4)
            low = int(np.min(y_realF)) - int(np.min(y_realF) / 4)
            plt.figure()
            plt.ylim(low, up)
            plt.plot(y_realF, color='black', label='Real')
            plt.plot(y_predF, color='blue', label='Prediction')
            plt.legend()
            plt.title("CV(RMSE)={}".format(str(a)))
            plt.savefig('plot1.png')
        return res

    def cv_analysis_svm(self, fold, C, epsilon, tol, mean_y, plot, q=[], model=[]):
        '''
        :param fold: divisions in cv analysis
        :param q: a Queue to paralelyse or empty list to do not paralyse
        :param plot: True plots
        :return: predictions, real values, errors and the times needed to train
        '''
        from pathlib import Path
        import random
        names = self.data.drop(self.data.columns[self.pos_y], axis=1).columns
        print('##########################'
              '################################'
              'CROSS-VALIDATION'
              '#################################'
              '################################')
        x = pd.DataFrame(self.data.drop(self.data.columns[self.pos_y], axis=1))
        y = pd.DataFrame(self.data.iloc[:, self.pos_y])
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        res = self.svm_cv_division(x, y, fold)
        x_test = res['x_test']
        x_train = res['x_train']
        y_test = res['y_test']
        y_train = res['y_train']
        indexes = res['indexes']
        times_test = []
        tt = self.times
        for t in range(len(indexes)):
            times_test.append(tt[indexes[t][0]:indexes[t][1]])
        if self.type == 'classification':
            print('nothing')
            # data2 = self.data
            # yy = data2.iloc[:, self.pos_y]
            # yy = pd.Series(yy, dtype='category')
            # n_classes = len(yy.cat.categories.to_list())
            # model = self.__class__.mlp_classification(layers, neurons, x_train[0].shape[1], n_classes, self.mask,
            #                                          self.mask_value)
            #####################################################################
            # EN PROCESOO ALGÚN DíA !!!!!!!
            ##########################################################################
        else:
            # Train the model
            times = [0 for x in range(fold)]
            cv = [0 for x in range(fold)]
            rmse = [0 for x in range(fold)]
            nmbe = [0 for x in range(fold)]
            predictions = []
            reales = []
            for z in range(fold):
                h_path = Path('./best_models')
                h_path.mkdir(exist_ok=True)
                h = h_path / f'best_{random.randint(0, 1000000)}_model.h5'
                if isinstance(model, list):
                    res = self.SVR_training(pd.concat([pd.DataFrame(y_train[z]).reset_index(drop=True),
                                                       pd.DataFrame(x_train[z]).reset_index(drop=True)], axis=1),
                                            self.pos_y, C, epsilon, tol, False)
                    modelF = res['model']
                else:
                    model1 = model
                    res = self.SVR_training(pd.concat([pd.DataFrame(y_train[z]).reset_index(drop=True),
                                                       pd.DataFrame(x_train[z]).reset_index(drop=True)], axis=1),
                                            self.pos_y, C, epsilon, tol,
                                            False, model1)
                    modelF = res['model']

                print('Fold number', z)
                test_x = pd.DataFrame(x_test[z]).reset_index(drop=True)
                test_y = pd.DataFrame(y_test[z]).reset_index(drop=True)
                time_start = time()
                times[z] = round(time() - time_start, 3)
                y_pred = modelF.predict(test_x)

                y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
                y_real = np.array(self.scalar_y.inverse_transform(test_y))
                if isinstance(self.pos_y, collections.abc.Sized):
                    for t in range(len(self.pos_y)):
                        y_pred[np.where(y_pred[:, t] < self.inf_limit[t])[0], t] = self.inf_limit[t]
                        y_pred[np.where(y_pred[:, t] > self.sup_limit[t])[0], t] = self.sup_limit[t]
                else:
                    y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
                    y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit

                y_predF = y_pred.copy()
                y_predF = pd.DataFrame(y_predF)
                y_predF.index = times_test[z]
                y_realF = y_real.copy()
                y_realF = pd.DataFrame(y_realF)
                y_realF.index = times_test[z]
                predictions.append(y_predF)
                reales.append(y_realF)
                predictions.append(y_predF)
                reales.append(y_realF)
                if self.zero_problem == 'schedule':
                    print('*****Night-schedule fixed******')
                    res = super().fix_values_0(times_test[z],
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

                    if len(y_pred1) > 0:
                        if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                            if mean_y.size == 0:
                                e = evals(y_pred1, y_real1).variation_rate()
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e)
                                else:
                                    cv[z] = np.sum(e * self.weights)
                                rmse[z] = np.nan
                                nmbe[z] = np.nan
                            else:
                                e_cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                                e_r = evals(y_pred1, y_real1).rmse()
                                e_n = evals(y_pred1, y_real1).nmbe(mean_y)
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e_cv)
                                    rmse[z] = np.mean(e_r)
                                    nmbe[z] = np.mean(e_n)
                                else:
                                    cv[z] = np.sum(e_cv * self.weights)
                                    rmse[z] = np.sum(e_r * self.weights)
                                    nmbe[z] = np.sum(e_n * self.weights)
                        else:
                            print('Missing values are detected when we are evaluating the predictions')
                            cv[z] = 9999
                            rmse[z] = 9999
                            nmbe[z] = 9999
                    else:
                        raise NameError('Empty prediction')
                elif self.zero_problem == 'radiation':
                    print('*****Night-radiation fixed******')
                    place = np.where(names == 'radiation')[0]
                    scalar_rad = self.scalar_x['radiation']
                    res = super().fix_values_0(scalar_rad.inverse_transform(x_test[z].iloc[:, place]),
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

                    if len(y_pred1) > 0:
                        if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                            if mean_y.size == 0:
                                e = evals(y_pred1, y_real1).variation_rate()
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e)
                                else:
                                    print(e)
                                    print(self.weights)
                                    cv[z] = np.sum(e * self.weights)
                                rmse[z] = np.nan
                                nmbe[z] = np.nan
                            else:
                                e_cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                                e_r = evals(y_pred1, y_real1).rmse()
                                e_n = evals(y_pred1, y_real1).nmbe(mean_y)
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e_cv)
                                    rmse[z] = np.mean(e_r)
                                    nmbe[z] = np.mean(e_n)
                                else:
                                    cv[z] = np.sum(e_cv * self.weights)
                                    rmse[z] = np.sum(e_r * self.weights)
                                    nmbe[z] = np.sum(e_n * self.weights)
                        else:
                            print('Missing values are detected when we are evaluating the predictions')
                            cv[z] = 9999
                            rmse[z] = 9999
                            nmbe[z] = 9999
                    else:
                        raise NameError('Empty prediction')
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

                    if len(y_pred) > 0:
                        if np.sum(np.isnan(y_pred)) == 0 and np.sum(np.isnan(y_real)) == 0:
                            if mean_y.size == 0:
                                e = evals(y_pred, y_real).variation_rate()
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e)
                                else:
                                    print(e)
                                    print(self.weights)
                                    cv[z] = np.sum(e * self.weights)
                                rmse[z] = np.nan
                                nmbe[z] = np.nan
                            else:
                                e_cv = evals(y_pred, y_real).cv_rmse(mean_y)
                                e_r = evals(y_pred, y_real).rmse()
                                e_n = evals(y_pred, y_real).nmbe(mean_y)
                                if isinstance(self.weights, list):
                                    cv[z] = np.mean(e_cv)
                                    rmse[z] = np.mean(e_r)
                                    nmbe[z] = np.mean(e_n)
                                else:
                                    cv[z] = np.sum(e_cv * self.weights)
                                    rmse[z] = np.sum(e_r * self.weights)
                                    nmbe[z] = np.sum(e_n * self.weights)
                        else:
                            print('Missing values are detected when we are evaluating the predictions')
                            cv[z] = 9999
                            rmse[z] = 9999
                            nmbe[z] = 9999
                    else:
                        raise NameError('Empty prediction')
                if plot == True and len(y_realF.shape) > 1:
                    s = np.max(y_realF.iloc[:, 0]).astype(int) + 15
                    i = np.min(y_realF.iloc[:, 0]).astype(int) - 15
                    a = np.round(cv[z], 2)
                    plt.figure()
                    plt.ylim(i, s)
                    plt.plot(y_realF.iloc[:, 0], color='black', label='Real')
                    plt.plot(y_predF.iloc[:, 0], color='blue', label='Prediction')
                    plt.legend()
                    plt.title("Subsample {} - CV(RMSE)={}".format(z, str(a)))
                    a = 'Subsample-'
                    b = str(z) + '.png'
                    plot_name = a + b
                    plt.show()
                    plt.savefig(plot_name)
                elif plot == True and len(y_realF.shape) < 2:
                    s = np.max(y_realF).astype(int) + 15
                    i = np.min(y_realF).astype(int) - 15
                    a = np.round(cv[z], 2)
                    plt.figure()
                    plt.ylim(i, s)
                    plt.plot(y_realF, color='black', label='Real')
                    plt.plot(y_predF, color='blue', label='Prediction')
                    plt.legend()
                    plt.title("Subsample {} - CV(RMSE)={}".format(z, str(a)))
                    a = 'Subsample-'
                    b = str(z) + '.png'
                    plot_name = a + b
                    plt.show()
                    plt.savefig(plot_name)
            res = {'preds': predictions, 'reals': reales, 'times_test': times_test, 'cv_rmse': cv,
                   'std_cv': np.std(cv),
                   'nmbe': nmbe, 'rmse': rmse,
                   'times_comp': times}
            print(("The model with", C, " C", epsilon, "epsilon and a tol of", tol, "has: \n"
                                                                                    "The average CV(RMSE) is",
                   np.mean(cv), " \n"
                                "The average NMBE is", np.mean(nmbe), "\n"
                                                                      "The average RMSE is", np.mean(rmse), "\n"
                                                                                                            "The average time to train is",
                   np.mean(times)))
            z = Queue()
            if type(q) == type(z):
                q.put(np.array([np.mean(cv), SVM.complex_svm(C, epsilon, 10000, 100)]))
            else:
                return (res)

    def optimal_search(self, C_options,epsilon_options,tol_options, fold,mean_y, parallel, weights):
        '''
        :param fold: division in cv analyses
        :param parallel: True or false (True to linux)
        :param top: the best options yielded
        :return: the options with their results and the top options
        '''

        error = [0 for x in range(len(C_options)*len(epsilon_options)*len(tol_options))]
        complexity = [0 for x in range(len(C_options)*len(epsilon_options)*len(tol_options))]
        options = {'C':[], 'epsilon':[], 'tol':[]}
        w=0
        contador= len(C_options)*len(epsilon_options)*len(tol_options)-1
        if parallel <2:
            for t in range(len(C_options)):
                print('##################### Option ####################', w)
                C_sel = C_options[t]
                for i in range(len(epsilon_options)):
                    epsilon_sel=epsilon_options[i]
                    for u in range(len(tol_options)):
                        options['C'].append(C_sel)
                        options['epsilon'].append(epsilon_sel)
                        options['tol'].append(tol_options[u])
                        res = self.cv_analysis_svm(fold, C_sel, epsilon_sel, tol_options[u] ,mean_y,False)
                        error[w]=np.mean(res['cv_rmse'])
                        complexity[w]=SVM.complex_svm(C_sel, epsilon_sel,10000,100)
                        w +=1
        elif parallel>=2:
            processes = []
            res2 = []
            dev2 = []
            z = 0
            q = Queue()
            for t in range(len(C_options)):
                print('##################### Option ####################', w)
                C_sel = C_options[t]
                for i in range(len(epsilon_options)):
                    epsilon_sel = epsilon_options[i]
                    for u in range(len(tol_options)):
                        options['C'].append(C_sel)
                        options['epsilon'].append(epsilon_sel)
                        options['tol'].append(tol_options[u])
                        if z < parallel and w < contador:
                            multiprocessing.set_start_method('fork')
                            p = Process(target=self.cv_analysis_svm,
                                        args=(fold, C_sel, epsilon_sel, tol_options[u] ,mean_y,False, q))
                            p.start()
                            processes.append(p)
                            z1 =z+ 1
                        elif z == parallel and w < contador:
                            p.close()
                            for p in processes:
                                p.join()
                            for v in range(len(processes)):
                                res2.append(q.get()[0])
                                res2.append(q.get()[1])
                            processes = []
                            # multiprocessing.set_start_method('fork')
                            # multiprocessing.set_start_method('spawn')
                            q = Queue()
                            p = Process(target=self.cv_analysis_svm,
                                        args=(fold, C_sel, epsilon_sel, tol_options[u] ,mean_y,False, q))
                            p.start()
                            processes.append(p)
                            z1 = 1
                        elif w == contador:
                            p = Process(target=self.cv_analysis_svm,
                                        args=(fold, C_sel, epsilon_sel, tol_options[u] ,mean_y,False, q))
                            p.start()
                            processes.append(p)
                            p.close()
                            for p in processes:
                                p.join()
                                #res2.append(q.get())
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
        print(r1)

        scal_cv = MinMaxScaler(feature_range=(0, 1))
        scal_com = MinMaxScaler(feature_range=(0, 1))

        scal_cv.fit(np.array(r1).reshape(-1, 1))
        scal_com.fit(np.array(d1).reshape(-1, 1))

        cv = scal_cv.transform(np.array(r1).reshape(-1, 1))
        com = scal_com.transform(np.array(d1).reshape(-1, 1))

        r_final = np.array([cv[:, 0], com[:, 0]]).T

        I = get_decomposition("aasf", beta=5).do(r_final, weights).argmin()
        #I = get_decomposition("pbi").do(r_final, weights).argmin()

        top_result = {'error': [], 'complexity': [], 'C': [], 'epsilon': [], 'tol': []}
        top_result['error'] = r1[I]
        top_result['complexity'] = d1[I]
        top_result['C'] = options['C'][I]
        top_result['epsilon'] = options['epsilon'][I]
        top_result['tol'] = options['tol'][I]

        print(top_result['error'])
        print(top_result['complexity'])
        print(top_result['C'])
        print(top_result['epsilon'])
        print(top_result['tol'])

        np.savetxt('objectives_selected_brute.txt', np.array([top_result['error'], top_result['complexity']]))
        np.savetxt('x_selected_brute.txt',np.array([top_result['C'],top_result['epsilon'],top_result['tol'] ]))

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
        return(res)



    def nsga2_individual(self, med, contador, n_processes, C_max, epsilon_max, pop_size, tol, xlimit_inf,
                         xlimit_sup, dictionary, weights):
        '''
        :param med:
        :param contador: a operator to count the attempts
        :param n_processes: how many processes are parallelise
        :param l_dense:maximun number of layers dense
        :param batch: batch size
        :param pop_size: population size selected for NSGA2
        :param tol: tolearance selected to terminate the process
        :param xlimit_inf: array with the lower limits to the neuron  lstm , neurons dense and pacience
        :param xlimit_sup:array with the upper limits to the neuron  lstm , neurons dense and pacience
        :param dictionary: dictionary to stored the options tested
        :return: options in Pareto front, the optimal selection and the total results
        '''
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.factory import get_problem, get_visualization, get_decomposition
        from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
        from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
        from pymoo.optimize import minimize
        from pymoo.core.problem import starmap_parallelized_eval
        print('DATA is', type(self.data))
        if n_processes > 1:
            pool = multiprocessing.Pool(n_processes)
            problem = MyProblem_svm(self.data,self.horizont, self.scalar_y, self.scalar_x,self.zero_problem,self.extract_cero, self.limits, self.times, self.pos_y,self.n_lags,
                                self.mask,
                                self.mask_value, self.inf_limit, self.sup_limit,
                                self.type,
                                med, contador,len(xlimit_inf), C_max, epsilon_max, xlimit_inf, xlimit_sup,dictionary,self.weights,runner = pool.starmap,func_eval=starmap_parallelized_eval)
        else:
            problem = MyProblem_svm(self.data,self.horizont, self.scalar_y, self.scalar_x,self.zero_problem,self.extract_cero, self.limits, self.times, self.pos_y,self.n_lags,
                                self.mask,
                                self.mask_value, self.inf_limit, self.sup_limit,
                                self.type,
                                med, contador,len(xlimit_inf), C_max, epsilon_max, xlimit_inf, xlimit_sup,dictionary,self.weights)
        algorithm = NSGA2(pop_size=pop_size, repair=MyRepair_svm(),eliminate_duplicates=True,
                          sampling=get_sampling("int_random"),
                          crossover=get_crossover("int_sbx",prob=0.95),
                          mutation=get_mutation("int_pm", prob=0.4))
        termination = MultiObjectiveSpaceToleranceTermination(tol=tol,
                                                              n_last=int(pop_size / 2), nth_gen=int(pop_size / 4),
                                                              n_max_gen=None,
                                                              n_max_evals=5000)
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
        if n_processes > 1:
            pool.close()
        else:
            pass
        return (obj, struct, obj_T, struct_T, res,contador)

    def optimal_search_nsga2(self, C_max, epsilon_max, pop_size, tol, xlimit_inf, xlimit_sup, mean_y, parallel,weights):
        '''
        :param l_dense: maximun layers dense
        :param batch: batch size
        :param pop_size: population size for RVEA
        :param tol: tolerance to built the pareto front
        :param xlimit_inf: array with lower limits for neurons lstm, dense and pacience
        :param xlimit_sup: array with upper limits for neurons lstm, dense and pacience
        :param parallel: how many processes are parallelise
        if mean_y is empty a variation rate will be applied
        :return: the options selected for the pareto front, the optimal selection and the total results
        '''
        manager = multiprocessing.Manager()
        dictionary = manager.dict()
        contador = manager.list()
        contador.append(0)
        print('Start the optimization!!!!!')
        obj, x_obj, obj_total, x_obj_total, res,evaluations = self.nsga2_individual(mean_y, contador, parallel, C_max,
                                                                            epsilon_max, pop_size, tol, xlimit_inf,
                                                                            xlimit_sup, dictionary, weights)
        np.savetxt('objectives_selected.txt', obj)
        np.savetxt('x_selected.txt', x_obj)
        np.savetxt('objectives.txt', obj_total)
        np.savetxt('x.txt', x_obj_total)
        np.savetxt('evaluations.txt', evaluations)

        print('Process finished!!!')
        print('The selection is', x_obj, 'with a result of', obj)
        res = {'total_x': x_obj_total, 'total_obj': obj_total, 'opt_x': x_obj, 'opt_obj': obj, 'res': res,'evaluations':evaluations}
        return res

    def rnsga2_individual(self, med, contador, n_processes, C_max, epsilon_max, pop_size, tol, xlimit_inf,
                         xlimit_sup, dictionary, weights,epsilon):
        '''
        :param med:
        :param contador: a operator to count the attempts
        :param n_processes: how many processes are parallelise
        :param l_dense:maximun number of layers dense
        :param batch: batch size
        :param pop_size: population size selected for NSGA2
        :param tol: tolearance selected to terminate the process
        :param xlimit_inf: array with the lower limits to the neuron  lstm , neurons dense and pacience
        :param xlimit_sup:array with the upper limits to the neuron  lstm , neurons dense and pacience
        :param dictionary: dictionary to stored the options tested
        :return: options in Pareto front, the optimal selection and the total results
        '''
        from pymoo.factory import  get_decomposition
        from pymoo.factory import  get_crossover, get_mutation, get_sampling
        from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
        from pymoo.optimize import minimize
        from pymoo.core.problem import starmap_parallelized_eval
        print('DATA is', type(self.data))
        if n_processes > 1:
            pool = multiprocessing.Pool(n_processes)
            problem = MyProblem_svm(self.data,self.horizont, self.scalar_y, self.scalar_x,self.zero_problem,self.extract_cero, self.limits, self.times, self.pos_y,self.n_lags,
                                self.mask,
                                self.mask_value, self.inf_limit, self.sup_limit,
                                self.type,
                                med, contador,len(xlimit_inf), C_max, epsilon_max, xlimit_inf, xlimit_sup,dictionary,self.weights,runner = pool.starmap,func_eval=starmap_parallelized_eval)
        else:
            problem = MyProblem_svm(self.data,self.horizont, self.scalar_y, self.scalar_x,self.zero_problem,self.extract_cero, self.limits, self.times, self.pos_y,self.n_lags,
                                self.mask,
                                self.mask_value, self.inf_limit, self.sup_limit,
                                self.type,
                                med, contador,len(xlimit_inf), C_max, epsilon_max, xlimit_inf, xlimit_sup,dictionary,self.weights)

        ref_points = np.array([[0.3, 0.1], [0.1, 0.3]])

        algorithm = RNSGA2(ref_points, pop_size=pop_size, sampling=get_sampling("int_random"),
                          crossover=get_crossover("int_sbx",prob=0.95),
                          mutation=get_mutation("int_pm", prob=0.4),
                           normalization='front',
                           extreme_points_as_reference_points=False,
                           weights=weights,
                           epsilon=epsilon)

        termination = MultiObjectiveSpaceToleranceTermination(tol=tol,
                                                              n_last=int(pop_size / 2), nth_gen=int(pop_size / 4),
                                                              n_max_gen=None,
                                                              n_max_evals=5000)
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

            plt.figure(figsize=(10, 7))
            plt.scatter(r_final[:, 0], r_final[:, 1], color='black')
            plt.xlabel('Normalised CV (RMSE)', fontsize=20, labelpad=10)
            plt.ylabel('Normalised Complexity', fontsize=20, labelpad=10)
            plt.scatter(r_final[I, 0], r_final[I, 1], s=200, color='red', alpha=1, marker='o', facecolors='none',
                        label='Optimum')
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(borderpad=1, fontsize=15)
            plt.savefig('optimisation_plotR.png')
        else:
            obj_T = res.F
            struct_T = res.X
            obj = res.F
            struct = res.X
        print('The number of evaluations were:', contador)
        if n_processes > 1:
            pool.close()
        else:
            pass
        return (obj, struct, obj_T, struct_T, res,contador)

    def optimal_search_rnsga2(self, C_max, epsilon_max, pop_size, tol, xlimit_inf, xlimit_sup, mean_y, parallel,weights,epsilon=0.01):
        '''
        :param l_dense: maximun layers dense
        :param batch: batch size
        :param pop_size: population size for RVEA
        :param tol: tolerance to built the pareto front
        :param xlimit_inf: array with lower limits for neurons lstm, dense and pacience
        :param xlimit_sup: array with upper limits for neurons lstm, dense and pacience
        :param parallel: how many processes are parallelise
        if mean_y is empty a variation rate will be applied
        :return: the options selected for the pareto front, the optimal selection and the total results
        '''
        manager = multiprocessing.Manager()
        dictionary = manager.dict()
        contador = manager.list()
        contador.append(0)
        print('Start the optimization!!!!!')
        obj, x_obj, obj_total, x_obj_total, res,evaluations = self.rnsga2_individual(mean_y, contador, parallel, C_max,
                                                                            epsilon_max, pop_size, tol, xlimit_inf,
                                                                            xlimit_sup, dictionary, weights,epsilon)
        np.savetxt('objectives_selectedR.txt', obj)
        np.savetxt('x_selectedR.txt', x_obj)
        np.savetxt('objectivesR.txt', obj_total)
        np.savetxt('xR.txt', x_obj_total)
        np.savetxt('evaluationsR.txt', evaluations)

        print('Process finished!!!')
        print('The selection is', x_obj, 'with a result of', obj)
        res = {'total_x': x_obj_total, 'total_obj': obj_total, 'opt_x': x_obj, 'opt_obj': obj, 'res': res,'evaluations':evaluations}
        return res


from pymoo.core.repair import Repair
class MyRepair_svm(Repair):
    def info(self):
        print('Class defining a function to repair the possible error of the genetic algorithm. If a layer is zero the next layer cannot have positive neurons')
    def __init__(self):
        pass

    def _do(self, problem, pop, **kwargs):
        for k in range(len(pop)):
            x = pop[k].X
            #x2 = x[range(self.l_dense)]
            r_dense = MyProblem_svm.bool(x)
            if r_dense == 0:
                pass
            elif r_dense != 0:
                x[1] = x[0]-0.1

            pop[k].X = x
        return pop

from pymoo.core.problem import ElementwiseProblem
class MyProblem_svm(ElementwiseProblem):
    def info(self):
        print('Class to create a specific problem to use NSGA2 in architectures search.')
    def __init__(self,data,horizont, scalar_y,scalar_x, zero_problem,extract_cero,limits, times, pos_y, n_lags,
                 mask, mask_value, inf_limit,sup_limit, type,med, contador,
        n_var, C_max, epsilon_max, xlimit_inf, xlimit_sup, dictionary, weights, **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_constr=0,
                         xl=xlimit_inf,
                         xu=xlimit_sup,
                         type_var=np.int,
                         # elementwise_evaluation=True,
                         **kwargs)

        self.data = data
        self.horizont = horizont
        self.scalar_y = scalar_y
        self.scalar_x = scalar_x
        self.zero_problem = zero_problem
        self.extract_cero = extract_cero
        self.limits = limits
        self.times = times
        self.pos_y = pos_y
        self.mask = mask
        self.mask_value = mask_value
        self.n_lags = n_lags
        self.inf_limit = inf_limit
        self.sup_limit = sup_limit
        self.type = type
        self.med = med
        self.contador = contador
        self.C_max = C_max
        self.epsilon_max = epsilon_max
        self.xlimit_inf = xlimit_inf
        self.xlimit_sup = xlimit_sup
        self.n_var = n_var
        self.dictionary = dictionary
        self.weights = weights

    def cv_opt(self,data, fold, C_svm,epsilon_svm,tol_svm, mean_y, dictionary):
        '''
        :param fold:assumed division of the sample for cv
        :param dictionary: dictionary to fill with the options tested
        :param q:operator to differentiate when there is parallelisation and the results must be a queue
        :return: cv(rmse) and complexity of the model tested
        if mean_y is empty a variation rate will be applied
        '''

        from pathlib import Path
        import random

        h_path = Path('./best_models')
        h_path.mkdir(exist_ok=True)
        h = h_path / f'best_{random.randint(0, 1000000)}_model.h5'

        name1 = tuple(np.concatenate((np.array([C_svm]), np.array([epsilon_svm]))))
        try:
            a0, a1 = dictionary[name1]
            return a0, a1
        except KeyError:
            pass
        cvs = [0 for x in range(fold)]
        print(type(data))
        names = self.data.columns
        print(names)
        names = np.delete(names, self.pos_y)

        y = self.data.iloc[:,self.pos_y]
        x =self.data.drop(self.data.columns[self.pos_y],axis=1)

        res = SVM.svm_cv_division(x, y, fold)
        x_test = res['x_test']
        x_train = res['x_train']
        y_test = res['y_test']
        y_train = res['y_train']
        indexes = res['indexes']
        times_test = []
        tt = self.times
        for t in range(len(indexes)):
            times_test.append(tt[indexes[t][0]:indexes[t][1]])
        if self.type=='classification':
            data2 = self.data
            #yy = data2.iloc[:, self.pos_y]
            #yy = pd.Series(yy, dtype='category')
            #n_classes = len(yy.cat.categories.to_list())
            #model = SVM.SVM_training(layers, neurons, x_train[0].shape[1], n_classes, self.mask, self.mask_value)
            ####################################################################
            # EN PROCESOO ALGÚN DíA !!!!!!!
            ##########################################################################
        else:
            # Train the model
            for z in range(fold):
                print('Fold number', z)

                x_t = pd.DataFrame(x_train[z]).reset_index(drop=True)
                y_t = pd.DataFrame(y_train[z]).reset_index(drop=True)

                test_x = pd.DataFrame(x_test[z]).reset_index(drop=True)
                test_y = pd.DataFrame(y_test[z]).reset_index(drop=True)
                res = SVM.SVR_training(pd.concat([y_t, x_t], axis=1), self.pos_y,C_svm, epsilon_svm,tol_svm,False,[])
                model=res['model']
                y_pred = model.predict(test_x)
                y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
                y_real = test_y
                y_real = np.array(self.scalar_y.inverse_transform(y_real))
                if isinstance(self.pos_y, collections.abc.Sized):
                    for t in range(len(self.pos_y)):
                        y_pred[np.where(y_pred[:, t] < self.inf_limit[t])[0], t] = self.inf_limit[t]
                        y_pred[np.where(y_pred[:, t] > self.sup_limit[t])[0], t] = self.sup_limit[t]
                    y_real = y_real
                else:
                    y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
                    y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit
                    y_real = y_real.reshape(-1, 1)

                y_predF = y_pred.copy()
                y_predF = pd.DataFrame(y_predF)
                y_predF.index = times_test[z]
                y_realF = y_real.copy()
                y_realF = pd.DataFrame(y_realF)
                y_realF.index = times_test[z]
                if self.zero_problem == 'schedule':
                    print('*****Night-schedule fixed******')
                    res = ML.fix_values_0(times_test[z],
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

                    if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                        if mean_y.size == 0:
                            e=evals(y_pred1, y_real1).variation_rate()
                            if isinstance(self.weights, list):
                                cvs[z]=np.mean(e)
                            else:
                                cvs[z] = np.sum(e * self.weights)

                        else:
                            e = evals(y_pred1, y_real1).cv_rmse(mean_y)
                            if isinstance(self.weights, list):
                                cvs[z] = np.mean(e)
                            else:
                                cvs[z] = np.sum(e * self.weights)

                elif self.zero_problem == 'radiation':

                    print('*****Night-radiation fixed******')
                    place = np.where(names == 'radiation')[0]
                    scalar_rad = self.scalar_x['radiation']
                    res = ML.fix_values_0(scalar_rad.inverse_transform(x_test[z].iloc[:, place]),
                                               self.zero_problem, self.limits)
                    index_rad = res['indexes_out']
                    index_rad2 = np.where(y_real <= self.inf_limit)[0]
                    index_rad = np.union1d(np.array(index_rad), np.array(index_rad2))

                    if len(index_rad) > 0 and self.horizont == 0:
                        y_pred1 = np.delete(y_pred, index_rad, 0)
                        y_real1 = np.delete(y_real, index_rad, 0)
                    elif len(index_rad) > 0 and self.horizont > 0:
                        y_pred1 = np.delete(y_pred, index_rad - self.horizont, 0)
                        y_real1 = np.delete(y_real, index_rad - self.horizont, 0)
                    else:
                        y_pred1 = y_pred
                        y_real1 = y_real

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

                    if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                        if mean_y.size == 0:
                            e=evals(y_pred1, y_real1).variation_rate()
                            if isinstance(self.weights, list):
                                cvs[z] = np.mean(e)
                            else:
                                cvs[z] = np.sum(e * self.weights)

                        else:
                            e = evals(y_pred1, y_real1).cv_rmse(mean_y)
                            if isinstance(self.weights, list):
                                cvs[z] = np.mean(e)
                            else:
                                cvs[z] = np.sum(e * self.weights)

                    else:
                        print('Missing values are detected when we are evaluating the predictions')
                        cvs[z] = 9999
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

                    if np.sum(np.isnan(y_pred)) == 0 and np.sum(np.isnan(y_real)) == 0:
                        if mean_y.size == 0:
                            e=evals(y_pred, y_real).variation_rate()
                            if isinstance(self.weights, list):
                                cvs[z] = np.mean(e)
                            else:
                                print(e)
                                print(self.weights)
                                cvs[z] = np.sum(e * self.weights)
                        else:
                            e = evals(y_pred, y_real).cv_rmse(mean_y)
                            if isinstance(self.weights, list):
                                cvs[z] = np.mean(e)
                            else:
                                cvs[z] = np.sum(e * self.weights)

                    else:
                        print('Missing values are detected when we are evaluating the predictions')
                        cvs[z] = 9999

            complexity = SVM.complex_svm(C_svm, epsilon_svm, 10000, 100)
            dictionary[name1] = np.mean(cvs), complexity
            print(dictionary[name1] )
            res_final = {'cvs': np.mean(cvs), 'complexity': complexity}
            return res_final['cvs'], res_final['complexity']

    @staticmethod
    def bool(x):
        '''
        :x: neurons options
        l_dense: number of values that represent dense neurons
        :return: 0 if the constraint is fulfilled
        '''
        C_svm= x[0]
        epsilon_svm=x[1]
        #
        if C_svm > epsilon_svm:
           ret=0
        else:
            ret=epsilon_svm-C_svm

        return ret
    #
    def _evaluate(self, x, out, *args, **kwargs):
        g1 = MyProblem_svm.bool(x)
        out["G"] = g1
        print(x)
        C_F= x[0]*0.1
        epsilon_F =x[1]*0.1
        tol_F = x[2]*0.1
        f1, f2 = self.cv_opt(self.data, 3, C_F, epsilon_F,tol_F,  self.med, self.dictionary)
        print(
            '\n ############################################## \n ############################# \n ########################## EvaluaciÃ³n ',
            self.contador, '\n #########################')
        self.contador[0] += 1
        out["F"] = np.column_stack([f1, f2])

