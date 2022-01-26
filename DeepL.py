import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys
#sys.path.insert(1,'E:\Documents\Doctorado\Clases_python')

from errors import Eval_metrics as evals
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter
from time import time
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import RepeatVector
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.layers import TimeDistributed
import skfda
import math
import multiprocessing
from multiprocessing import Process,Manager,Queue

class DL:
    def info(self):
        print('Super class to built different deep learning models. This class has other more specific classes associated with it  ')

    def __init__(self, data, horizont,scalar_y, scalar_x,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,  inf_limit,sup_limit):
        self.data = data
        self.horizont = horizont
        self.scalar_y = scalar_y
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

    @staticmethod
    def cv_division(x,y, fold):
        '''
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
            while w2 <= x.shape[0]:
                a = x.iloc[range(w,w2)]
                X_val.append(a.iloc[range(len(a)-math.floor(len(a)/2), len(a))])
                X_test.append(a.drop(range(len(a)-math.floor(len(a)/2), len(a))))
                X_train.append(x.drop(range(w,w2)))

                a = y.iloc[range(w,w2)]
                Y_val.append(a.iloc[range(len(a)-math.floor(len(a)/2), len(a))])
                Y_train.append(y.drop(range(w,w2)))

                indexes.append(np.array([w,w2]))
                w=w2
                w2+=step
                if(w2 > x.shape[0] and w < x.shape[0]):
                    w2 = x.shape[0]
        except:
            raise NameError('Problems with the sample division in the cv classic')
        res = {'x_test': X_test, 'x_train':X_train,'X_val':X_val, 'y_test':Y_test, 'y_train':Y_train, 'y_val':Y_val,'indexes':indexes}
        return res

#    @staticmethod
#    def fix_values_0(restriction, zero_problem, limit):
#        '''
#        :param restriction: schedule hours or irradiance variable depending on the zero_problem
#        :param zero_problem: schedule or radiation
#        :param limit: limit hours or radiation limit
#        :return: the indexes where the data are not in the correct time schedule or is below the radiation limit
#        '''
#        if zero_problem == 'schedule':
#            try:
#                limit1 = limit[0]
#                limit2 = limit[1]
#                hours = restriction.hour
#                ii = np.where(hours < limit1 | hours > limit2)[0]
#                ii = ii[ii >= 0]
#            except:
#                raise NameError('Zero_problem and restriction incompatibles')
#        elif zero_problem == 'radiation':
#            try:
#                rad = restriction
#                ii = np.where(rad <= limit)[0]
#                ii = ii[ii >= 0]
#            except:
#                raise NameError('Zero_problem and restriction incompatibles')
#        else:
#            ii=[]
#            'Unknown situation with nights'
#
#        res = {'indexes_out': ii}
#        return res


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
        inf = np.where(self.data.iloc[:,self.pos_y] < self.inf_limit)[0]
        sup = np.where(self.data.iloc[:,self.pos_y] > self.sup_limit)[0]
        if len(inf)>0:
            self.data.iloc[inf, self.pos_y] = np.repeat(self.inf_limit, len(inf))
        if len(sup)>0:
            self.data.iloc[sup, self.pos_y] = np.repeat(self.sup_limit, len(sup))

    def adapt_horizont(self):
        '''
        Move the data sample to connected the y with the x based on the future selected
        '''
        if self.horizont==0:
            self.data = self.data
        else:
            X = self.data.drop(self.data.columns[self.pos_y], axis=1)
            y = self.data.iloc[:,self.pos_y]
            for t in range(self.horizont):
                y =y.drop(y.index[0], axis=0)
                X = X.drop(X.index[X.shape[0] - 1], axis=0)

            if self.pos_y == 0:
                self.data = pd.concat([y, X], axis=1)
            else:
                self.data = pd.concat([X,y], axis=1)

        print('Horizont adjusted!')



    def scalating(self, scalar_limits, groups, x, y):
        '''
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
            self.data.iloc[:, self.pos_y] = scalar_y.transform(pd.DataFrame(self.data.iloc[:, self.pos_y]))[:, 0]

            self.scalar_y = scalar_y

    def missing_values_remove(self):
        self.data = self.data.dropna()
        self.times = self.data.index

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


    def fda_outliers(self, freq):
        '''
        :param freq: amount of values in a hour (value to divide 60 and the result is the amount of data in a hour)
        :return: the variable y with missing value in the days considered as outliers
        '''
        step = int(60/freq)
        y = self.data.iloc[:, self.pos_y]
        long=len(y)
        hour=self.times.hour
        start = np.where(hour==0)[0][0]



        if np.where(hour==0)[0][len(np.where(hour==0)[0])-1] > np.where(hour==23)[0][len(np.where(hour==23)[0])-1]:
            d = np.where(hour==0)[0][len(np.where(hour==0)[0])-1]-np.where(hour==23)[0][len(np.where(hour==23)[0])-1]
            end = np.where(hour==0)[0][len(np.where(hour==0)[0])-1-d]
        elif np.where(hour==0)[0][len(np.where(hour==0)[0])-1] < np.where(hour==23)[0][len(np.where(hour==23)[0])-1]:
            if np.sum(hour[np.where(hour==0)[0][len(np.where(hour==0)[0])-1]:np.where(hour==23)[0][len(np.where(hour==23)[0])-1]] == 23) == step:
                end =np.where(hour==23)[0][len(np.where(hour==23)[0])-1]
            else:
                d = np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - 1] - np.where(hour == 23)[0][
                    len(np.where(hour == 23)[0]) - 1]
                end = np.where(hour == 0)[0][len(np.where(hour == 0)[0])-1-d]
        else:
            end=[]
            raise NameError('Problem with the limit of sample creating the functional sample')


        y1 = y.iloc[range(start+1)]
        y2 = y.iloc[range(end-1, len(y))]

        y_short = y.iloc[range(start+1,end-1)]
        if len(y_short) % (step*24)!=0:
            print(len(y_short))
            print(len(y_short)/(step*24))
            raise NameError('Sample size not it is well divided among days')

        fd_y = DL.cortes(y_short, len(y_short), int(24*step)).transpose()
        print(fd_y.shape)
        grid = []
        for t in range(int(24*step)):
            grid.append(t)

        fd_y2 = fd_y.copy()
        missing = []
        missing_p = []
        for t in range(fd_y.shape[0]):
            if np.sum(np.isnan(fd_y[t,:])) > 0:
                missing.append(t)
                missing_p.append(np.where(np.isnan(fd_y[t,:]))[0])

        if len(missing)>0:
            fd_y3 = pd.DataFrame(fd_y2.copy())
            for j in range(len(missing)):
                fd_y3.iloc[missing[j], missing_p[j]]=self.mask_value
                fd_y2[missing[j], missing_p[j]]=self.mask_value
            index2 = fd_y3.index
            print(missing)
            print(index2)
        else:
            fd_y3 = pd.DataFrame(fd_y2.copy())
            index2 = fd_y3.index


        fd = fd_y2.tolist()
        fd1 = skfda.FDataGrid(fd, grid)

        out_detector1 = skfda.exploratory.outliers.IQROutlierDetector(factor=3, depth_method=skfda.exploratory.depth.BandDepth())    #MSPlotOutlierDetector()
        out_detector2 = skfda.exploratory.outliers.LocalOutlierFactor(n_neighbors=int(fd_y2.shape[0]/5))
        oo1 = out_detector1.fit_predict(fd1)
        oo2 = out_detector2.fit_predict(fd1)
        o1 = np.where(oo1 ==-1)[0]
        o2 = np.where(oo2 ==-1)[0]
        o_final = np.intersect1d(o1,o2)

        print('El número de outliers detectado es:',len(o_final))
        if len(o_final)>0:
            out = index2[o_final]


            for t in range(len(o_final)):
                w = np.empty(fd_y.shape[1])
                w[:] = self.mask_value
                fd_y[out[t],:]= w

        Y = fd_y.flatten()

        Y = pd.concat([pd.Series(y1), pd.Series(Y), pd.Series(y2)], axis=0)
        if len(Y) != long:
            print(len(Y))
            print(long)
            raise NameError('Sample size error in the second joint')

        Y.index=self.data.index
        self.data.iloc[:,self.pos_y] = Y

        print('Data have been modified masking the outliers days!')




class LSTM_model(DL):
    def info(self):
        print('Class to built LSTM models.')


    def __init__(self, data, horizont,scalar_y, scalar_x,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,  inf_limit,sup_limit, repeat_vector,dropout, type):
        super().__init__(data, horizont,scalar_y, scalar_x,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,  inf_limit,sup_limit)
        self.repeat_vector = repeat_vector
        self.dropout = dropout
        self.type = type

    @staticmethod
    def three_dimension(data_new, n_inputs):
        rest2 = data_new.shape[0] % n_inputs
        ind_out = 0
        while rest2 != 0:
            data_new = np.delete(data_new,0, axis=0)
            rest2 = data_new.shape[0] % n_inputs
            ind_out += 1

        # restructure into windows of  data
        data_new = np.array(np.split(data_new, len(data_new) / n_inputs))
        res={'data':data_new, 'ind_out':ind_out}
        return res

    @staticmethod
    def split_dataset(data_new1, n_inputs,cut1, cut2):
       '''
        :param data_new1: data
        :param n_inputs:n lags selected to create the blocks lstm
        :param cut1: lower limit to divided into train - test
        :param cut2: upper limit to divided into train - test
        :return: data divided into train - test in three dimensions
       '''

       index=data_new1.index
       data_new1=data_new1.reset_index(drop=True)

       data_new = data_new1.copy()

       train, test = data_new.drop(range(cut1, cut2)), data_new.iloc[range(cut1, cut2)]
       index_test = index[range(cut1, cut2)]

       ###################################################################################
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
       #index_test = np.array(np.split(index_test, len(index_test) / n_inputs))

       return train1, test1, index_test


    @staticmethod
    def to_supervised(train,pos_y, n_lags, horizont, onebyone):
        '''
        :param horizont: horizont to the future selected
        :return: x (past) and y (future horizont) considering the past-future relations selected
        '''

        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))

        X, y = list(), list()

        in_start = 0
        # step over the entire history one time step at a time
        if onebyone==True:

            for _ in range(len(data)-(n_lags + horizont)):
            #for _ in range(int((len(data)-n_lags)/horizont)):
                # define the end of the input sequence
                in_end = in_start + n_lags
                if horizont ==0:
                    out_end = in_end
                else:
                    out_end = in_end+horizont

                # ensure we have enough data for this instance
                if out_end < len(data):
                    xx = np.delete(data,pos_y,1)
                    x_input = xx[in_start:in_end,:]
                    # x_input = x_input.reshape((len(x_input), 1))
                    X.append(x_input)
                    yy = data[:,pos_y].reshape(-1,1)
                    #y.append(yy.iloc[in_end:out_end])
                    if horizont==0:
                        y.append(yy[out_end])
                    else:
                        y.append(yy[in_end:out_end])
                    #se selecciona uno
                # move along one time step
                in_start += 1
                #in_start += horizont
        else:
            for _ in range(int((len(data)-(n_lags + horizont))/horizont)):
                # define the end of the input sequence
                in_end = in_start + n_lags
                if horizont ==0:
                    out_end = in_end
                else:
                    out_end = in_end+horizont

                # ensure we have enough data for this instance
                if out_end <= len(data):
                    xx = np.delete(data,pos_y,1)
                    x_input = xx[in_start:in_end,:]
                    # x_input = x_input.reshape((len(x_input), 1))
                    X.append(x_input)
                    yy = data[:,pos_y].reshape(-1,1)
                    #y.append(yy.iloc[in_end:out_end])
                    if horizont==0:
                        y.append(yy[out_end])
                    else:
                        y.append(yy[in_end:out_end])
                    #se selecciona uno
                # move along one time step
                in_start += horizont
        dd= len(data) - in_start

        return(np.array(X), np.array(y), dd)

    @staticmethod
    def built_model_classification(train_x1, train_y1, neurons_lstm, neurons_dense, mask, mask_value, repeat_vector, dropout):
        '''
        :param mask: True or False
        :param repeat_vector: True or False
        :return: the model architecture built to be trained
        '''

        print('No finished')


        layers_lstm = len(neurons_lstm)
        layers_neurons = len(neurons_dense)

        n_timesteps, n_features, n_outputs = train_x1.shape[1], train_x1.shape[2], train_y1.shape[1]  # define model

        model = Sequential()
        if mask == True:
            model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
            model.add(LSTM(n_features, activation='relu', return_sequences=True))
        else:
            model.add(LSTM(n_features, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)))
        for k in range(layers_lstm):
            if (repeat_vector == True and k == 0):
                model.add(LSTM(neurons_lstm[k], activation='relu'))
                model.add(RepeatVector(n_outputs))
            else:
                model.add(LSTM(neurons_lstm[k], activation='relu'))

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
        :param mask: True or False
        :param repeat_vector: True or False
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
        from pathlib import Path
        import random

        '''
        :param model: model architecture built
        :return: model trained
        '''
        print(train_x1.shape)
        print(train_y1.shape)
        print(test_x1.shape)
        print(test_y1.shape)

        h_path = Path('./best_models')
        h_path.mkdir(exist_ok=True)
        h = h_path / f'best_{random.randint(0, 1000000)}_model.h5'

        # Checkpoitn callback
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pacience)
        mc = ModelCheckpoint(str(h), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # Train the model
        model.fit(train_x1, train_y1, epochs=2000, validation_data=(test_x1, test_y1), batch_size=batch,
                           callbacks=[es, mc])

        return model


    @staticmethod
    def predict_model(model,n_lags, x_val,batch):
        '''
        :param model: trained model
        :param n_lags: lags to built lstm blocks
        :return: predictions in the validation sample, considering the selected moving window
        '''
        data = np.array(x_val)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        predictions = list()
        l1 = 0
        l2 = n_lags
        for i in range(len(x_val)):
            # flatten data
            input_x = data[l1:l2, :]
            input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
            # forecast the next step
            yhat = model.predict(input_x, verbose=0, batch_size=batch)
            yhat = yhat[0]
            predictions.append(yhat)
            #history.append(tt[i,:])
            l1 =l2
            l2 += n_lags


        predictions  =np.array(predictions)
        print(len(predictions))
        y_pred = predictions.reshape((predictions.shape[0] * predictions.shape[1], 1))
        res = {'y_pred': y_pred}
        return res

    @staticmethod
    def cv_division_lstm(data, horizont, fold, pos_y,n_lags):
        '''
        :return: Division to cv analysis considering that with lstm algorithm the data can not be divided into simple pieces.
        It can only be divided with a initial part and a final part
        return: train, test, and validation samples. Indexes for test samples (before division into test-validation)
        '''
        X_test = []
        X_train = []
        X_val = []
        Y_test = []
        Y_train = []
        Y_val = []
        ###################################################################################

        step = int(data.shape[0] / fold)
        w = 0
        w2 = step
        times_val = []
        ####################################################################################

        try:
           for i in range(2):
                train, test, index_val = LSTM_model.split_dataset(data, n_lags,w, w2)

                #index_val = index_test[range(index_test.shape[0]-math.ceil(index_test.shape[0]/2)),: ,1]
                val = test[range(test.shape[0]-math.ceil(test.shape[0]/2), test.shape[0]),:,:]
                test = test[range(0, math.ceil(test.shape[0] / 2)), :, :]
                x_train, y_train,dif = LSTM_model.to_supervised(train, pos_y, n_lags,horizont, True)
                index_val = index_val[range(index_val.shape[0] - val.shape[0]*val.shape[1], index_val.shape[0])]

                print(index_val.shape)
                print(val.shape)

                x_test, y_test,dif = LSTM_model.to_supervised(test, pos_y, n_lags,horizont,True)
                x_val, y_val,dif = LSTM_model.to_supervised(val, pos_y, n_lags,horizont,True)
                #index_val = index_val.reshape((index_val.shape[0] * index_val.shape[1], train.shape[2]))

                #index_val = index_val.reshape((index_val.shape[0] * index_val.shape[1], 1))
                index_val = np.delete(index_val, range(n_lags), axis=0)
                #index_val = np.delete(index_val, range(index_val.shape[0]-n_lags, index_val.shape[0]), axis=0)

                #diff = len(index_val) - (y_val.shape[0] * y_val.shape[1])
                #if diff > 0:
                #    index_val = np.delete(index_val, range(len(index_val) - diff, len(index_val)), axis=0)
                times_val.append(index_val)

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

    def cv_analysis(self, fold,rep, neurons_lstm, neurons_dense, pacience, batch,mean_y,plot, q=[]):
        '''
        :param fold: the assumed size of divisions
        :param rep: In this case, the analysis repetitions of each of the two possile division considered in lstm analysis
        :param q: queue that inform us if paralyse or not
        :param plot: True plots
        :return: Considering what zero_problem is mentioned, return thre predictions, real values, errors and computational times needed to train the models
        '''

        names = self.data.columns
        names = np.delete(names ,self.pos_y)
        layers_lstm = len(neurons_lstm)
        layers_neurons = len(neurons_dense)

        res = LSTM_model.cv_division_lstm(self.data, self.horizont, fold, self.pos_y, self.n_lags)

        x_test =np.array(res['x_test'])
        x_train=np.array(res['x_train'])
        x_val=np.array(res['x_val'])
        y_test=np.array(res['y_test'])
        y_train =np.array(res['y_train'])
        y_val =np.array(res['y_val'])
#
        times_val = res['time_val']

        print(times_val[0].shape)
        print(y_val[0].shape)


        if self.type=='regression':
            model = self.__class__.built_model_regression(x_train[0],y_train[0],neurons_lstm, neurons_dense, self.mask,self.mask_value, self.repeat_vector, self.dropout)
            # Train the model
            times = [0 for x in range(rep*2)]
            cv = [0 for x in range(rep*2)]
            rmse = [0 for x in range(rep*2)]
            nmbe = [0 for x in range(rep*2)]
            zz= 0
            predictions = []
            reales = []
            for z in range(2):
                print('Fold number', z)
                for zz2 in range(rep):
                    time_start = time()
                    model = self.__class__.train_model(model,x_train[z], y_train[z], x_test[z], y_test[z], pacience, batch)
                    times[zz] = round(time() - time_start, 3)

                    res = self.__class__.predict_model(model, self.n_lags, x_val[z], batch)
                    y_pred = res['y_pred']



                    y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
                    y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
                    y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit

                    y_real = y_val[z].reshape((y_val[z].shape[0] * y_val[z].shape[1], 1))
                    y_real = np.array(self.scalar_y.inverse_transform(y_real))

                    if plot == True:

                        if self.horizont>1:
                            y_predP = y_pred.reshape(int(y_pred.shape[0] / self.horizont), self.horizont)
                            y_realP = y_real.reshape(int(y_real.shape[0] / self.horizont), self.horizont)

                            y_predP = y_predP[:,0]
                            y_predP = y_predP.index = times_val[z]
                            y_realP = y_realP[:,0]
                            y_realP = y_realP.index = times_val[z]

                            s = np.max(y_realP).astype(int) + 12
                            i = np.min(y_realP).astype(int) - 12
                            plt.figure()
                            plt.ylim(i, s)
                            plt.plot(y_realP, color='black', label='Real')
                            plt.plot(y_predP, color='blue', label='Prediction')
                            plt.legend()
                            plt.title("Subsample {} ".format(z))
                            a = 'Subsample-'
                            b = str(z) + '.png'
                            plt.show()
                        else:
                            y_real.index=times_val[z]
                            y_pred.index=times_val[z]

                            s = np.max(y_real).astype(int) + 12
                            i = np.min(y_real).astype(int) - 12
                            plt.figure()
                            plt.ylim(i, s)
                            plt.plot(y_real, color='black', label='Real')
                            plt.plot(y_pred, color='blue', label='Prediction')
                            plt.legend()
                            plt.title("Subsample {} ".format(z))
                            a = 'Subsample-'
                            b = str(z) + '.png'
                            plt.show()


                    y_predF = y_pred.copy()
                    y_predF = pd.DataFrame(y_predF)
                    y_realF = y_real.copy()
                    y_realF = pd.DataFrame(y_realF)

                    if self.zero_problem == 'schedule':
                        print('*****Night-schedule fixed******')

                        y_pred[np.where(y_pred < self.inf_limit)[0]]=self.inf_limit
                        y_pred[np.where(y_pred > self.sup_limit)[0]]=self.sup_limit

                        res = super().fix_values_0(times_val[z],
                                                      self.zero_problem, self.limits)


                        index_hour = res['indexes_out']

                        predictions.append(y_predF)
                        reales.append(y_realF)

                        if len(index_hour) > 0 and self.horizont == 0:
                            y_pred1 = np.delete(y_pred, index_hour, 0)
                            y_real1 = np.delete(y_real, index_hour, 0)
                        elif len(index_hour) > 0 and self.horizont > 0:
                            y_pred1 = np.delete(y_pred, index_hour - 1, 0)
                            y_real1 = np.delete(y_real, index_hour - 1, 0)
                        else:
                            y_pred1 = y_pred
                            y_real1 = y_real

                        #Outliers and missing values
                        o = np.where(y_real1<self.inf_limit)[0]

                        if len(o)>0:
                            y_pred1 = np.delete(y_pred1,o,0)
                            y_real1 = np.delete(y_real1,o, 0)

                        if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                            cv[zz] = evals(y_pred1, y_real1).cv_rmse(mean_y)
                            rmse[zz] = evals(y_pred1, y_real1).rmse()
                            nmbe[zz] = evals(y_pred1, y_real1).nmbe(mean_y)
                        else:
                            print('Missing values are detected when we are evaluating the predictions')
                            cv[zz] = 9999
                            rmse[zz] = 9999
                            nmbe[zz] = 9999
                    elif self.zero_problem == 'radiation':
                        print('*****Night-radiation fixed******')
                        #y_pred = y_pred.reshape(int(y_pred.shape[0] / self.horizont), self.horizont)
                        #y_real = y_real.reshape(int(y_real.shape[0] / self.horizont), self.horizont)

                        index_rad = np.where(np.sum(y_real<=self.inf_limit*1, axis=1)>0)[0]

                        predictions.append(y_predF)
                        reales.append(y_realF)
                        #if len(index_rad) > 0 and self.horizont == 0:
                        #    y_pred1 = np.delete(y_pred, index_rad, 0)
                        #    y_real1 = np.delete(y_real, index_rad, 0)
                        #elif len(index_rad) > 0 and self.horizont > 0:
                        #    y_pred1 = np.delete(y_pred, index_rad - 1, 0)
                        #    y_real1 = np.delete(y_real, index_rad - 1, 0)
                        if len(index_rad) > 0:
                            y_pred1 = np.delete(y_pred, index_rad, 0)
                            y_real1 = np.delete(y_real, index_rad, 0)
                        else:
                            y_pred1 = y_pred
                            y_real1 = y_real


                        #y_pred1 = y_pred1.reshape(y_pred1.shape[0] * y_pred1.shape[1], 1)
                        #y_real1 = y_real1.reshape(y_real1.shape[0] * y_real1.shape[1], 1)

                        #Outliers and missing values
                        #o = np.where(y_real1 < self.inf_limit)[0]
#
                        #if len(o)>0:
                        #    y_pred1 = np.delete(y_pred1,o,0)
                        #    y_real1 = np.delete(y_real1,o, 0)

                        if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                            cv[zz] = evals(y_pred1, y_real1).cv_rmse(mean_y)
                            rmse[zz] = evals(y_pred1, y_real1).rmse()
                            nmbe[zz] = evals(y_pred1, y_real1).nmbe(mean_y)
                        else:
                            print('Missing values are detected when we are evaluating the predictions')
                            cv[zz] = 9999
                            rmse[zz] = 9999
                            nmbe[zz] = 9999
                    else:
                        y_real2 = y_real.copy()

                        predictions.append(y_predF)
                        reales.append(y_realF)

                        # Outliers and missing values
                        o = np.where(y_real2 < self.inf_limit)[0]

                        if len(o) > 0:
                            y_pred2 = np.delete(y_pred, o, 0)
                            y_real2 = np.delete(y_real, o, 0)
                        else:
                            y_pred2 = y_pred
                            y_real2 = y_real

                        if np.sum(np.isnan(y_pred2))==0 and np.sum(np.isnan(y_real2))==0:
                            cv[zz] = evals(y_pred2, y_real2).cv_rmse(mean_y)
                            rmse[zz] = evals(y_pred2, y_real2).rmse()
                            nmbe[zz] = evals(y_pred2, y_real2).nmbe(mean_y)
                        else:
                            print('Missing values are detected when we are evaluating the predictions')
                            cv[zz] = 9999
                            rmse[zz] = 9999
                            nmbe[zz] = 9999



                    zz +=1

            res_final = {'preds': predictions, 'reals':reales, 'times_val':times_val, 'cv_rmse':cv,
                 'nmbe':nmbe, 'rmse':rmse,
                 'times_comp':times}


            print(("The model with", layers_lstm, " layers lstm,",layers_neurons,'layers dense', neurons_dense, "neurons denses,", neurons_lstm,"neurons_lstm and a pacience of", pacience, "has: \n"
                                                                                                        "The average CV(RMSE) is",
                   np.nanmean(cv), " \n"
                                "The average NMBE is", np.nanmean(nmbe), "\n"
                                                                      "The average RMSE is", np.nanmean(rmse), "\n"
                                                                      "The average time to train is", np.mean(times)))

            z = Queue()
            if type(q) == type(z):
                q.put(np.array([np.mean(cv), np.std(cv)]))
            else:
                return (res_final)

    ###################################################################################################
        #FALTARÍA CLASIFICACION !!!!!!!!!!!!!!!!!
        ###################################################################################################


    def train(self, train, test, neurons_lstm, neurons_dense, pacience, batch):
        '''
        :return: the trained model and the time required to be trained

        Instance to train model outside these classes
        '''

        res = self.__class__.three_dimension(train, self.n_lags)
        train = res['data']
        res = self.__class__.three_dimension(test, self.n_lags)
        test = res['data']

        x_test, y_test,dif =self.__class__.to_supervised(test, self.pos_y, self.n_lags, self.horizont,True)
        x_train, y_train,dif = self.__class__.to_supervised(train, self.pos_y, self.n_lags, self.horizont, True)

        print(x_train.shape)
        print(x_test.shape)
        print(y_test.shape)

        if self.type=='regression':
            model = self.__class__.built_model_regression(x_train, y_train,neurons_lstm, neurons_dense, self.mask, self.mask_value, self.repeat_vector, self.dropout)
            time_start = time()
            model_trained = self.__class__.train_model(model, x_train, y_train, x_test, y_test, pacience, batch)
            times = round(time() - time_start, 3)
        else:
            model = self.__class__.built_model_classification(x_train, y_train,neurons_lstm, neurons_dense,self.mask, self.mask_value, self.repeat_vector, self.dropout)
            time_start = time()
            model_trained = self.__class__.train_model(model, x_train, y_train, x_test, y_test, pacience, batch)
            times = round(time() - time_start, 3)
        res = {'model': model_trained, 'times': times}
        return res


    def predict(self, model, val,mean_y,batch,times):
        '''
        :param model: trained model
        :return: prediction with the built metrics
        Instance to predict certain samples outside these classes
        '''

        times = np.delete(times, range(self.n_lags), 0)

        res = self.__class__.three_dimension(val, self.n_lags)

        val = res['data']
        i_out = res['ind_out']
        if i_out>0:
            times=np.delete(times, range(i_out),0)

        x_val, y_val,dif = self.__class__.to_supervised(val, self.pos_y, self.n_lags, self.horizont, False)

        print('Diferencia entre time and y:',dif)

        print(x_val.shape)
        print(y_val.shape)


        res = self.__class__.predict_model(model, self.n_lags,  x_val,batch)

        y_pred = res['y_pred']

        print(y_pred.shape)

        y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
        y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
        y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit

        y_real = y_val.reshape((y_val.shape[0] * y_val.shape[1], 1))
        y_real = np.array(self.scalar_y.inverse_transform(y_real))

        y_predF = y_pred.copy()
        y_predF = pd.DataFrame(y_predF)
        y_predF.index = times
        y_realF = pd.DataFrame(y_real.copy())
        y_realF.index = y_predF.index

        if self.zero_problem == 'schedule':
            print('*****Night-schedule fixed******')


            res = super().fix_values_0(self.times,
                                       self.zero_problem, self.limits)

            index_hour = res['indexes_out']

            if len(y_pred <= 1):
                y_pred1 = np.nan
                y_real1 = y_real
            else:

                if len(index_hour) > 0 and self.horizont == 0:
                    y_pred1 = np.delete(y_pred, index_hour, 0)
                    y_real1 = np.delete(y_real, index_hour, 0)
                elif len(index_hour) > 0 and self.horizont > 0:
                    y_pred1 = np.delete(y_pred, index_hour - 1, 0)
                    y_real1 = np.delete(y_real, index_hour - 1, 0)
                else:
                    y_pred1 = y_pred
                    y_real1 = y_real

            # Outliers and missing values
            if len(y_pred1)>0:
                o = np.where(y_real1 < self.inf_limit)[0]
                y_pred1 = np.delete(y_pred1, o, 0)
                y_real1 = np.delete(y_real1, o, 0)

            if len(y_pred1)>0:
                if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                    cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                    nmbe = evals(y_pred1, y_real1).nmbe(mean_y)
                    rmse = evals(y_pred1, y_real1).rmse()
                    r2 = evals(y_pred1, y_real1).r2()
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
            #place = np.where(x_val.columns == 'radiation')[0]
            #scalar_x = self.scalar_x
            #scalar_rad = scalar_x['radiation']

            #res = super().fix_values_0(scalar_rad.inverse_transform(x_val.iloc[:, place]),
            #                           self.zero_problem, self.limits)
            #index_rad = res['indexes_out']
            index_rad = np.where(np.sum(y_real <= self.inf_limit * 1, axis=1) > 0)[0]
#
            if len(y_pred <= 1):
                y_pred1 = np.nan
                y_real1 = y_real
            else:
                if len(index_rad) > 0:
                    y_pred1 = np.delete(y_pred, index_rad, 0)
                    y_real1 = np.delete(y_real, index_rad, 0)
                else:
                    y_pred1 = y_pred
                    y_real1 = y_real



                # Outliers and missing values
            #o = np.where(y_real < self.inf_limit)[0]
            #if len(o) > 0:
            #    y_pred1 = np.delete(y_pred1, o, 0)
            #    y_real1 = np.delete(y_real1, o, 0)

            if len(y_pred1)>0:
                if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                    cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                    nmbe = evals(y_pred1, y_real1).nmbe(mean_y)
                    rmse = evals(y_pred1, y_real1).rmse()
                    r2 = evals(y_pred1, y_real1).r2()
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
            else:
                raise NameError('Empty prediction')
        else:
            # Outliers and missing values
            y_real2 = y_real.copy()
            o = np.where(y_real2 < self.inf_limit)[0]
            y_pred = np.delete(y_pred, o, 0)
            y_real = np.delete(y_real, o, 0)
            if len(y_pred)>0:
                if np.sum(np.isnan(y_pred)) == 0 and np.sum(np.isnan(y_real)) == 0:
                    cv = evals(y_pred, y_real).cv_rmse(mean_y)
                    rmse = evals(y_pred, y_real).rmse()
                    nmbe = evals(y_pred, y_real).nmbe(mean_y)
                    r2 = evals(y_pred, y_real).r2()
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
            else:
                raise NameError('Empty prediction')

        res = {'y_pred': y_predF, 'cv_rmse': cv, 'nmbe': nmbe, 'rmse':rmse,'r2':r2}

        y_realF = pd.DataFrame(y_realF)
        y_realF.index = y_predF.index

        a = np.round(cv, 2)
        up =int(np.max(y_realF)) + int(np.max(y_realF)/4)
        low = int(np.min(y_realF)) + int(np.min(y_realF)/4)
        plt.figure()
        plt.ylim(low, up)
        plt.plot(y_predF, color='black', label='Prediction')
        plt.plot(y_realF, color='blue', label='Real')
        plt.legend()
        plt.title("CV(RMSE)={}".format(str(a)))
        plt.show()
        plt.savefig('plot1.png')

        return res

    def optimal_search(self, fold, rep, neurons_dense, neurons_lstm, paciences, batch, mean_y,parallel,top):
        '''
        :param fold: assumed division of data sample
        :param rep: repetitions of cv analysis considering the intial or the final of sample
        :param parallel: 0 no paralyse
        :param top: number of best solution selected
        :return: errors obtained with the options considered together  with the best solutions
        '''
        results = [0 for x in range(len(neurons_lstm) * len(neurons_dense) * len(paciences))]
        deviations = [0 for x in range(len(neurons_lstm) * len(neurons_dense) * len(paciences))]

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
                        res = self.cv_analysis(fold, rep, neuron_lstm, neuron_dense, paciences[i], batch, mean_y,False)
                        results[w] = np.mean(res['cv_rmse'])
                        deviations[w] = np.std(res['cv_rmse'])
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
                            #multiprocessing.set_start_method('fork')
                            #multiprocessing.set_start_method('spawn')
                            p = Process(target=self.cv_analysis,
                                        args=(fold,rep, neuron_lstm, neuron_dense, paciences[i], batch, mean_y,False, q))
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
                            #multiprocessing.set_start_method('fork')
                            #multiprocessing.set_start_method('spawn')
                            q = Queue()
                            p = Process(target=self.cv_analysis,
                                        args=(fold, rep, neuron_lstm, neuron_dense, paciences[i], batch, mean_y,False, q))
                            p.start()

                            processes.append(p)
                            z1 = 1

                        elif w==contador:
                            p = Process(target=self.cv_analysis,
                                        args=(fold, rep, neuron_lstm, neuron_dense, paciences[i], batch, mean_y,False, q))
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
            results = res2
            deviations = dev2
        else:
            raise NameError('Option not considered')

        r1 = results.copy()
        d1 = deviations.copy()
        print(r1)
        top_results = {'error': [], 'std': [], 'neurons_dense': [],'neurons_lstm':[], 'pacience': []}

        for i in range(top):
            a = np.where(r1 == np.min(r1))[0]
            print(a)
            if len(a) == 1:
                zz = a[0]
            else:
                zz = a[0][0]

            top_results['error'].append(r1[zz])
            top_results['std'].append(d1[zz])
            top_results['neurons_dense'].append(options['neurons_dense'][zz])
            top_results['neurons_lstm'].append(options['neurons_lstm'][zz])
            top_results['pacience'].append(options['pacience'][zz])

            r1.remove(np.min(r1))
            d1.remove(d1[zz])
            options['neurons_dense'].pop(zz)
            options['neurons_lstm'].pop(zz)
            options['pacience'].pop(zz)

        print('Process finished!!!')
        res = {'errors': results, 'options': options, 'best': top_results}
        return res

    def nsga2_individual(self,med, contador,n_processes,l_lstm, l_dense, batch,pop_size,tol, xlimit_inf, xlimit_sup,dictionary):
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

        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.factory import get_problem, get_visualization, get_decomposition
        from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
        from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
        from pymoo.optimize import minimize
        from pymoo.core.problem import starmap_parallelized_eval

        if n_processes>1:
            pool = multiprocessing.Pool(n_processes)
            problem = MyProblem(self.horizont, self.scalar_y, self.zero_problem, self.limits,self.times,self.pos_y,self.mask,
                                self.mask_value, self.n_lags,self.inf_limit, self.sup_limit, self.repeat_vector, self.type, self.data,
                                self.scalar_x,self.dropout,med, contador,len(xlimit_inf),l_lstm, l_dense, batch, xlimit_inf, xlimit_sup,dictionary,runner = pool.starmap,func_eval=starmap_parallelized_eval)
        else:
            problem = MyProblem(self.horizont, self.scalar_y, self.zero_problem, self.limits,self.times,self.pos_y,self.mask,
                                self.mask_value, self.n_lags,self.inf_limit, self.sup_limit, self.repeat_vector, self.type, self.data,
                                self.scalar_x, self.dropout,med, contador,len(xlimit_inf),l_lstm, l_dense, batch, xlimit_inf, xlimit_sup,dictionary)

        algorithm = NSGA2(pop_size=pop_size, repair=MyRepair(l_lstm, l_dense), eliminate_duplicates=True,
                          sampling=get_sampling("int_random"),
                          # sampling =g,
                          # crossover=0.9,
                          # mutation=0.1)
                          crossover=get_crossover("int_sbx"),
                          mutation=get_mutation("int_pm", prob=0.1))

        termination = MultiObjectiveSpaceToleranceTermination(tol=tol,
                                                              n_last=int(pop_size/2), nth_gen=int(pop_size/4), n_max_gen=None,
                                                              n_max_evals=12000)

        res = minimize(problem,
                       algorithm,
                       termination,
                       # ("n_gen", 20),
                       pf=True,
                       verbose=True,
                       seed=7)

        if res.F.shape[0] > 1:
            weights = np.array([0.75, 0.25])
            I = get_decomposition("pbi").do(res.F, weights).argmin()
            obj_T = res.F
            struct_T = res.X
            obj = res.F[I, :]
            struct = res.X[I, :]
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

        return (obj, struct,obj_T, struct_T,  res)

    def optimal_search_nsga2(self,l_lstm, l_dense, batch, pop_size, tol,xlimit_inf, xlimit_sup, mean_y,parallel):
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
        :return: the options selected for the pareto front, the optimal selection and the total results
        '''

        manager = multiprocessing.Manager()
        dictionary = manager.dict()
        contador = manager.list()
        contador.append(0)
        obj, x_obj, obj_total, x_obj_total,res = self.nsga2_individual(mean_y, contador,parallel,l_lstm, l_dense, batch,pop_size,tol, xlimit_inf, xlimit_sup,dictionary)

        np.savetxt('objectives_selected.txt', obj)
        np.savetxt('x_selected.txt', x_obj)
        np.savetxt('objectives.txt', obj_total)
        np.savetxt('x.txt', x_obj_total)

        print('Process finished!!!')
        print('The selection is', x_obj, 'with a result of', obj)
        res = {'total_x': x_obj_total, 'total_obj': obj_total, 'opt_x': x_obj, 'opt_obj':obj, 'res':res}
        return res

from pymoo.core.repair import Repair
class MyRepair(Repair):
    def info(self):
        print('Class defining a function to repair the possible error of the genetic algorithm. If a layer is zero the next layer cannot have positive neurons')

    def __init__(self,l_lstm, l_dense):
        self.l_lstm=l_lstm
        self.l_dense = l_dense

    def _do(self, problem, pop, **kwargs):
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

        return pop


from pymoo.core.problem import ElementwiseProblem
class MyProblem(ElementwiseProblem):
    def info(self):
        print('Class to create a specific problem to use NSGA2 in architectures search. Two objectives and a constraint (Repair) concerning the neurons in each layer')
#
    def __init__(self, horizont,scalar_y,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,  inf_limit,sup_limit, repeat_vector, type,data,scalar_x,dropout, med, contador,
                 n_var,l_lstm, l_dense,batch,xlimit_inf, xlimit_sup,dictionary, **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_constr=1,
                         xl=xlimit_inf,
                         xu=xlimit_sup,
                         type_var=np.int,
                         #elementwise_evaluation=True,
                         **kwargs)

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
#
#
        u = len(neurons_lstm) + len(neurons_dense)
#
        F = 0.25 * (u / max_H) + 0.75 * np.sum(np.concatenate((neurons_lstm, neurons_dense))) / max_N
#
        return F
#
    def cv_nsga(self,data,fold,rep, neurons_lstm, neurons_dense, pacience, batch, mean_y,dictionary):
        '''
        :param fold:assumed division of the sample for cv
        :param rep:repetition of the estimation in each subsample
        :param dictionary: dictionary to fill with the options tested
        :param q:operator to differentiate when there is parallelisation and the results must be a queue
        :return: cv(rmse) and complexity of the model tested
        '''
        name1 = tuple(np.concatenate((neurons_lstm, neurons_dense, np.array([pacience]))))
#
        try:
            a0, a1 = dictionary[name1]
            return a0, a1
#
        except KeyError:
            pass
        cvs = [0 for x in range(rep*2)]
#
        print(type(data))
#
#
        names = data.columns
        names = np.delete(names, self.pos_y)
        res = LSTM_model.cv_division_lstm(data, self.horizont, fold, self.pos_y, self.n_lags)
#
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
            model = LSTM_model.built_model_regression(x_train[0], y_train[0], neurons_lstm, neurons_dense,
                                                          self.mask, self.mask_value, self.repeat_vector,self.dropout)
#
#
            # Train the model
            zz = 0
            for z in range(2):
                print('Fold number', z)
                for zz2 in range(rep):
                    time_start = time()
                    model = LSTM_model.train_model(model, x_train[z], y_train[z], x_test[z], y_test[z], pacience,
                                                       batch)
#
                    res = LSTM_model.predict_model(model, self.n_lags, x_val[z],batch)
                    y_pred = res['y_pred']
#
                    y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
#
                    y_real = y_val[z].reshape((y_val[z].shape[0] * y_val[z].shape[1], 1))
                    y_real2 = y_real.copy()
                    y_real = np.array(self.scalar_y.inverse_transform(y_real))
#
                    y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
                    y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit

                    y_predF = y_pred.copy()
                    y_predF = pd.DataFrame(y_predF)
                    y_predF.index = times_val[z]
                    y_realF = y_real.copy()
                    y_realF = pd.DataFrame(y_realF)
                    y_realF.index = times_val[z]

                    if self.zero_problem == 'schedule':
                        print('*****Night-schedule fixed******')

                        res = DL.fix_values_0(times_val[z],
                                                   self.zero_problem, self.limits)

                        index_hour = res['indexes_out']


                        if len(index_hour) > 0 and self.horizont == 0:
                            y_pred1 = np.delete(y_pred, index_hour, 0)
                            y_real1 = np.delete(y_real, index_hour, 0)
                            y_real2 = np.delete(y_real2, index_hour, 0)
                        elif len(index_hour) > 0 and self.horizont > 0:
                            y_pred1 = np.delete(y_pred, index_hour - self.horizont, 0)
                            y_real1 = np.delete(y_real, index_hour - self.horizont, 0)
                            y_real2 = np.delete(y_real2, index_hour - self.horizont, 0)
                        else:
                            y_pred1 = y_pred
                            y_real1 = y_real

                        # Outliers and missing values
                        o = np.where(y_real2 < self.inf_limit)[0]

                        if len(o) > 0:
                            y_pred1 = np.delete(y_pred1, o, 0)
                            y_real1 = np.delete(y_real1, o, 0)

                        if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                            cvs[zz] = evals(y_pred1, y_real1).cv_rmse(mean_y)

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


                        if len(index_rad) > 0 and self.horizont == 0:
                            y_pred1 = np.delete(y_pred, index_rad, 0)
                            y_real1 = np.delete(y_real, index_rad, 0)
                            y_real2 = np.delete(y_real2, index_rad, 0)
                        elif len(index_rad) > 0 and self.horizont > 0:
                            y_pred1 = np.delete(y_pred, index_rad - self.horizont, 0)
                            y_real1 = np.delete(y_real, index_rad - self.horizont, 0)
                            y_real2 = np.delete(y_real2, index_rad - self.horizont, 0)
                        else:
                            y_pred1 = y_pred
                            y_real1 = y_real

                        # Outliers and missing values
                        o = np.where(y_real2 < self.inf_limit)[0]

                        if len(o) > 0:
                            y_pred1 = np.delete(y_pred1, o, 0)
                            y_real1 = np.delete(y_real1, o, 0)

                        if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                            cvs[zz] = evals(y_pred1, y_real1).cv_rmse(mean_y)

                        else:
                            print('Missing values are detected when we are evaluating the predictions')
                            cvs[zz] = 9999

                    else:


                        # Outliers and missing values
                        o = np.where(y_real2 < self.inf_limit)[0]

                        if len(o) > 0:
                            y_pred2 = np.delete(y_pred, o, 0)
                            y_real2 = np.delete(y_real, o, 0)
                        else:
                            y_pred2 = y_pred
                            y_real2 = y_real

                        if np.sum(np.isnan(y_pred2)) == 0 and np.sum(np.isnan(y_real2)) == 0:
                            cvs[zz] = evals(y_pred2, y_real2).cv_rmse(mean_y)

                        else:
                            print('Missing values are detected when we are evaluating the predictions')
                            cvs[zz] = 9999


                zz += 1
#
            complexity = MyProblem.complex(neurons_lstm,neurons_dense, 50000, 8)
            dictionary[name1] = np.mean(cvs), complexity
            res_final = {'cvs': np.mean(cvs), 'complexity': complexity}
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
#
        if len(x2)==3:
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
        if len(x1)==3:
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
#
        return a_lstm, a_dense
#
#
    def _evaluate(self, x, out, *args, **kwargs):
        g1 = MyProblem.bool4(np.delete(x, len(x)-1), self.l_lstm, self.l_dense)
        out["G"] = g1
#
        print(x)
#
        n_lstm = x[range(self.l_lstm)]*10
        n_dense = x[range(self.l_lstm, self.l_lstm + self.l_dense)]*10
        n_pacience = x[len(x)-1]*10



        f1, f2 = self.cv_nsga(self.data,10,1, n_lstm, n_dense, n_pacience, self.batch, self.med,self.dictionary)
        print(
            '\n ############################################## \n ############################# \n ########################## Evaluacion ',
            self.contador, '\n #########################')

        self.contador[0] += 1
#
        out["F"] = np.column_stack([f1, f2])
#