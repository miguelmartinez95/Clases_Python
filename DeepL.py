from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.insert(1,'E:\Documents\Doctorado\Clases_python')

from errors import Eval_metrics as evals
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from time import time
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import RepeatVector
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
        #index = []
        indexes = []
        try:
            while w2 <= x.shape[0]:
                a = x.iloc[range(w,w2)]
                X_val.append(a.iloc[range(len(a)-math.floor(len(a)/2), len(a))])
                X_test.append(a.drop(range(len(a)-math.floor(len(a)/2), len(a))))
                #X_val.append(x.iloc[range(w,w2)])
                X_train.append(x.drop(range(w,w2)))

                a = y.iloc[range(w,w2)]
                Y_val.append(a.iloc[range(len(a)-math.floor(len(a)/2), len(a))])
                Y_test.append(a.drop(range(len(a)-math.floor(len(a)/2), len(a))))
                #Y_test.append(y.iloc[range(w,w2)])
                Y_train.append(y.drop(range(w,w2)))
                #Y_val.append(y.drop(range(w,w2)))

                indexes.append(np.array([w,w2]))
                w=w2
                w2+=step
                if(w2 > x.shape[0] and w < x.shape[0]):
                    w2 = x.shape[0]
        except:
            raise NameError('Problems with the sample division in the cv classic')
        res = {'x_test': X_test, 'x_train':X_train,'X_val':X_val, 'y_test':Y_test, 'y_train':Y_train, 'y_val':Y_val,'indexes':indexes}
        return res

    @staticmethod
    def fix_values_0(restriction, zero_problem, limit):
        if zero_problem == 'schedule':
            try:
                limit1 = limit[0]
                limit2 = limit[1]
                hours = restriction.hour
                # data = pd.DataFrame(data)
                # data = data.reset_index(drop=True)
                ii = np.where(hours < limit1 | hours > limit2)[0]
                ii = ii[ii >= 0]
            except:
                raise NameError('Zero_problem and restriction incompatibles')
        elif zero_problem == 'radiation':
            try:
                rad = restriction
                # data = pd.DataFrame(data)
                # data = data.reset_index(drop=True)
                ii = np.where(rad <= limit)[0]
                ii = ii[ii >= 0]
                # data.iloc[ii] = 0
            except:
                raise NameError('Zero_problem and restriction incompatibles')
        else:
            'Unknown situation with nights'

        #   ii2 = np.where(data<limit_inf)
        #   data.iloc[ii2]=0
        res = {'indexes_out': ii}
        return (res)


    @staticmethod
    def cortes(x, D, lim):
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
    def ts(new_data, look_back, pred_col, dim, names, lag):
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
        d1 = self.data.copy()
        dim = d1.shape[1]
        selec = range(dim - var_lag, dim)
        try:
            names1 = self.data.columns[selec]
            for i in range(self.n_lags):
                self.data = self.ts(self.data, lags, selec, self.data.shape[0], names1, i + 1)

                selec = range(dim, dim + var_lag)
                dim += var_lag
            self.times = self.data.index
        except:
            raise NameError('Problems introducing time lags')

    def adjust_limits(self):
        inf = np.where(self.data.iloc[:,self.pos_y] < self.inf_limit)[0]
        sup = np.where(self.data.iloc[:,self.pos_y] > self.sup_limit)[0]
        if len(inf)>0:
            self.data.iloc[inf, self.pos_y] = np.repeat(self.inf_limit, len(inf))
        if len(sup)>0:
            self.data.iloc[sup, self.pos_y] = np.repeat(self.sup_limit, len(sup))

    def adapt_horizont(self):
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
        scalars = dict()
        names = list(groups.keys())
        if x == True and y == True:
            try:
                for i in range(len(groups)):
                    #scalars.append(MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1])))
                    scalars[names[i]] = MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1]))
                    selec = groups[names[i]]
                    d = self.data.iloc[:, selec]
                    if (len(selec) > 1):
                        #scalars[i].fit(np.concatenate(np.array(d)).reshape(-1, 1))
                        scalars[names[i]].fit(np.concatenate(np.array(d)).reshape(-1, 1))
                    else:
                        #scalars[i].fit(np.array(d).reshape(-1, 1))
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
                    #scalars.append(MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1])))
                    scalars[names[i]] = MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1]))
                    selec = groups[names[i]]
                    d = self.data.iloc[:, selec]
                    if (len(selec) > 1):
                        #scalars[i].fit(np.concatenate(np.array(d)).reshape(-1, 1))
                        scalars[names[i]].fit(np.concatenate(np.array(d)).reshape(-1, 1))
                    else:
                        #scalars[i].fit(np.array(d).reshape(-1, 1))
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

    def missing_values_interpolate(self, delete_end, delete_start, mode, limit, sup, inf, order=2):
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
        step = int(60/freq)
        y = self.data.iloc[:, self.pos_y]
        index=y.index
        hour=self.times.hour
        start = np.where(hour==0)[0][0]

        if np.where(hour==0)[0][len(np.where(hour==0)[0])-1] > np.where(hour==23)[0][len(np.where(hour==23)[0])-1]:
            end = np.where(hour==0)[0][len(np.where(hour==0)[0])-step]
        elif np.where(hour==0)[0][len(np.where(hour==0)[0])-1] < np.where(hour==23)[0][len(np.where(hour==23)[0])-1]:
            if np.sum(hour[np.where(hour==0)[0][len(np.where(hour==0)[0])-1]:np.where(hour==23)[0][len(np.where(hour==23)[0])-1]] == 23) == step:
                end =len(y)
            else:
                end = np.where(hour == 0)[0][len(np.where(hour == 0)[0])-step]
        else:
            end=[]
            raise NameError('Problem with the limit of sample creating the functional sample')


        y1 = y.iloc[range(start)]
        y2 = y.iloc[range(end, len(y))]

        y_short = y.iloc[range(start,end)]
        print(len(y_short))
        print(y_short.index)
        fd_y = DL.cortes(y_short, len(y_short), int(24*step)).transpose()
        print(fd_y.shape)
        grid = []
        for t in range(int(24*step)):
            grid.append(t)

        fd_y2 = fd_y.copy()
        missing = []
        for t in range(fd_y.shape[0]):
            if np.sum(np.isnan(fd_y[t,:])) > 0:
                missing.append(t)

        if len(missing)>0:
            fd_y3 = pd.DataFrame(fd_y2.copy())
            fd_y2 = np.delete(fd_y2, missing, 0)
            fd_y3 = fd_y3.drop(missing, 0)
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

        print(len(o_final))
        if len(o_final)>0:
            out = index2[o_final]


            for t in range(len(o_final)):
                w = np.empty(fd_y.shape[1])
                w[:] = self.mask_value
                #fd_y[out[t],:]= np.zeros(fd_y.shape[1])
                fd_y[out[t],:]= w

        Y = fd_y.flatten()

        Y = pd.concat([pd.Series(y1), pd.Series(Y), pd.Series(y2)], axis=0)
        Y.index=self.data.index
        self.data.iloc[:,self.pos_y] = Y

        print('Data have been modified masking the outliers days!')




class LSTM_model(DL):
    def info(self):
        print('Class to built LSTM models.')


    def __init__(self, data, horizont,scalar_y, scalar_x,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,  inf_limit,sup_limit, repeat_vector, type):
        super().__init__(data, horizont,scalar_y, scalar_x,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,  inf_limit,sup_limit)
        self.repeat_vector = repeat_vector
        self.type = type



    @staticmethod
    def split_dataset(data_new1, n_inputs,cut1, cut2):
       index=data_new1.index
       data_new1=data_new1.reset_index(drop=True)

       data_new = data_new1.copy()
       #train, test = data[0:-cut], data[-cut:]
       train, test = data_new.drop(range(cut1, cut2)), data_new.iloc[range(cut1, cut2)]
       index_test = index[cut1:cut2]

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
 #      print(test.shape)
 #      train= train.drop(train.index[0], axis=0)
 #      test= test.drop(test.index[0], axis=0)
#
 #      train=train.reset_index(drop=True)
 #      test=test.reset_index(drop=True)
#
       # restructure into windows of weekly data
       train1 = np.array(np.split(train, len(train) / n_inputs))
       test1 = np.array(np.split(test, len(test) / n_inputs))
       index_test = np.array(np.split(index_test, len(index_test) / n_inputs))
     #  a=np.zeros((len(train1), 4, 6))
     #  print(a.shape)
     #  print(train1.shape)
     #  print(test1.shape)
#
     #  for i in range(a.shape[0]):
     #      print(i)
     #      a[i,:,:]=train1[i]
     #  print(a.shape)
      # train1 = torch.tensor(np.split(np.array(train), np.array([int(len(train) / n_inputs)])))
      # test1 = torch.tensor(np.split(np.array(test), np.array([int(len(test) / n_inputs)])))
       #train1 = np.array(train).reshape(int(train.shape[0]/n_inputs),n_inputs,train.shape[1] )
       #test1 = np.array(np.split(test, len(test) / n_inputs))
       return(train1, test1, index_test)


    @staticmethod
    def to_supervised(train,pos_y, n_lags, horizont):
        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        X, y = list(), list()

        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_lags
            #out_end = in_end + n_out
            if horizont ==0:
                out_end = in_end-1
            else:
                out_end = (in_end-1)+horizont

            # ensure we have enough data for this instance
            #if out_end <= len(data):
            if out_end < len(data):
                xx = np.delete(data,pos_y,1)
                x_input = xx[in_start:in_end]
                # x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                yy = data[:,pos_y].reshape(-1,1)
                #y.append(yy.iloc[in_end:out_end])
                y.append(yy[in_end])
                #se selecciona uno
            # move along one time step
            in_start += 1

        return(np.array(X), np.array(y))

    @staticmethod
    def built_model_classification(train_x1, train_y1, n_input, neurons_lstm, neurons_dense, mask, mask_value, repeat_vector):
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
            # model.add(Dense(int(nn / 2), activation='relu'))

        model.add(Dense(n_outputs), kernel_initializer='normal', activation='softmax')
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        model.summary()

        # res = {'model':model}
        return (model)


    @staticmethod
    def built_model_regression(train_x1, train_y1, n_input, neurons_lstm, neurons_dense, mask,mask_value, repeat_vector):
        layers_lstm = len(neurons_lstm)
        layers_neurons = len(neurons_dense)

        n_timesteps, n_features, n_outputs = train_x1.shape[1], train_x1.shape[2], train_y1.shape[1]

        model = Sequential()
        if mask==True:
            model.add(Masking(mask_value = mask_value,input_shape=(n_timesteps, n_features)))
            model.add(LSTM(n_features, activation='relu', return_sequences=True))
        else:
            model.add(LSTM(n_features, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)))
        for k in range(layers_lstm):
            if (repeat_vector==True and k==0):
                model.add(LSTM(neurons_lstm[k], activation='relu'))
                model.add(RepeatVector(n_outputs))
            else:
                model.add(LSTM(neurons_lstm[k], activation='relu'))

        for z in range(layers_neurons):
            if neurons_dense[z]==0:
                pass
            else:
                model.add(Dense(neurons_dense[z], activation='relu'))

        model.add(Dense(n_outputs,kernel_initializer='normal', activation='linear'))
        model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
        model.summary()

        return(model)

    @staticmethod
    def train_model(model,train_x1, train_y1, test_x1, test_y1, pacience, batch):

        # Checkpoitn callback
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pacience)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # Train the model
        model.fit(train_x1, train_y1, epochs=2000, validation_data=(test_x1, test_y1), batch_size=batch,
                           callbacks=[es, mc])
        # fit network
        return (model)


    @staticmethod
    def predict_model(model,n_lags, x_train,x_val, y_val,mean_y, scalar_y):
        #history=[x for x in x_train]
        #history=[x for x in x_val]
        data = np.array(x_val)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        #tt = x_val
        #print(x_val.shape)
        predictions = list()
        l1 = 0
        l2 = n_lags
        for i in range(len(x_val)):
            # flatten data
            #data = np.array(history)
            #data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
            # retrieve last observations for input data
            #input_x = data[-n_lags:, :]
            #input_x = data[-n_lags:, :]
            input_x = data[l1:l2, :]
            input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
            # forecast the next step
            yhat = model.predict(input_x, verbose=0)
            yhat = yhat[0]
            predictions.append(yhat)
            #history.append(tt[i,:])
            l1 = l2
            l2 += n_lags


        predictions  =np.array(predictions)
        print(predictions.shape)
        if len(predictions.shape)>2:
            y_pred = predictions[:,:,0]
        else:
            y_pred = predictions

        #y_pred = np.array(scalar_y.inverse_transform(predictions))
        #real = np.delete(y_val, range(self.n_lags))
        #y_pred = np.delete(y_pred, range(len(y_pred)-self.n_lags,len(y_pred)))
        #cv = evals(y_pred, y_val).cv_rmse(mean_y)
        #nmbe = evals(y_pred, y_val).nmbe(mean_y)
        #r2 = evals(y_pred, y_val).r2()

        #res = {'y_pred': y_pred, 'cv_rmse': cv, 'nmbe': nmbe, 'r2': r2}
        res = {'y_pred': y_pred}
        return(res)


    def adapt_sample(self,x, dates):
        if len(dates)>1:
            i1 = np.where(x.index==dates[0])[0][0]
            i2 = np.where(x.index==dates[1])[0][0]+1
            #dat_test = self.data.iloc[i1:i2]
            #data_train = self.data.drop(range(i1,i2))
            data_train, data_test, index_test = LSTM_model.split_dataset(x, self.n_lags, i1, i2)
       #     if len(index_test)>0:
       #         raise NameError('Problems with data dimension in relation with LSTM')
            x_train, y_train = LSTM_model.to_supervised(data_train, self.pos_y, self.n_lags, self.horizont)
            x_test, y_test = LSTM_model.to_supervised(data_test, self.pos_y, self.n_lags, self.horizont)


            res={'x_train':x_train, 'y_train':y_train, 'x_test':x_test, 'y_test':y_test}
        elif len(dates)==1:
            i1 = len(x)-1- dates
            i2 = len(x)-1
            # dat_test = self.data.iloc[i1:i2]
            # data_train = self.data.drop(range(i1,i2))
            data_train, data_test, index_test = LSTM_model.split_dataset(x, self.n_lags, i1, i2)
            #     if len(index_test)>0:
            #         raise NameError('Problems with data dimension in relation with LSTM')
            x_train, y_train = LSTM_model.to_supervised(data_train, self.pos_y, self.n_lags, self.horizont)
            x_test, y_test = LSTM_model.to_supervised(data_test, self.pos_y, self.n_lags, self.horizont)

            res = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

        else:
            index = x.index
            data_new1 = x.reset_index(drop=True)

            data_new = data_new1.copy()
            # train, test = data[0:-cut], data[-cut:]
            index_val = index
            ###################################################################################
            rest1 = data_new.shape[0] % self.n_lags
            ind_out1 = 0
            while rest1 != 0:
                data_new = data_new.drop(data_new.index[0], axis=0)
                rest1 = data_new.shape[0] % self.n_lags
                ind_out1 += 1

            ###################################################################################
            if ind_out1 > 0:
                index_val = np.delete(index_val, range(ind_out1), axis=0)
            data_new = np.array(np.split(data_new, len(data_new) / self.n_lags))
            index_val = np.array(np.split(index_val, len(index_val) / self.n_lags))

            x, y = LSTM_model.to_supervised(data_new, self.pos_y, self.n_lags, self.horizont)

            index_val = index_val.reshape((index_val.shape[0] * index_val.shape[1], 1))
            index_val = np.delete(index_val, range(self.n_lags), axis=0)


            res = {'x_val': x, 'y_val': y, 'index': index_val}
        return(res)

    @staticmethod
    def cv_division_lstm(data, horizont, fold, pos_y,n_lags, times):
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

        ####################################################################################


        #indexes = []
        times_test = []
        ####################################################################################

        try:
           for i in range(2):

                #train, test = LSTM_model.split_dataset(data, n_lags,w, w2)
                train, test, index_test = LSTM_model.split_dataset(data, n_lags,w, w2)
                #tt = times[w:w2]
                #print(len(tt))
                #if ind_out>0:
                #    tt = np.delete(tt, range(ind_out),axis=0)

                index_test = index_test[range(len(index_test)-math.ceil(len(index_test)/2), len(index_test)-1),:]
                val = test[range(test.shape[0]-math.ceil(test.shape[0]/2), test.shape[0]-1),:,:]
                test = test[range(test.shape[0]-math.floor(test.shape[0]/2), test.shape[0]),:,:]


                x_train, y_train = LSTM_model.to_supervised(train, pos_y, n_lags,horizont)
                x_test, y_test = LSTM_model.to_supervised(test, pos_y, n_lags,horizont)
                x_val, y_val = LSTM_model.to_supervised(val, pos_y, n_lags,horizont)
                index_test = index_test.reshape((index_test.shape[0] * index_test.shape[1], 1))
                index_test = np.delete(index_test, range(n_lags), axis=0)
                times_test.append(index_test[:,0])

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
        res = {'x_test': X_test, 'x_train': X_train, 'x_val':X_val, 'y_test': Y_test, 'y_train': Y_train, 'y_val':Y_val,  'time_test':times_test}
        return(res)



    def cv_analysis(self, fold,mc, neurons_lstm, neurons_dense, pacience, batch,mean_y, q=[]):
        scalar_y =self.scalar_y

        names = self.data.columns
        names = np.delete(names ,self.pos_y)
        layers_lstm = len(neurons_lstm)
        layers_neurons = len(neurons_dense)

        res = LSTM_model.cv_division_lstm(self.data, self.horizont, fold, self.pos_y, self.n_lags, self.times)

        x_test =np.array(res['x_test'])
        x_train=np.array(res['x_train'])
        x_val=np.array(res['x_val'])
        y_test=np.array(res['y_test'])
        y_train =np.array(res['y_train'])
        y_val =np.array(res['y_val'])
#
        times_test = res['time_test']


        if self.type=='regression':
            model = self.__class__.built_model_regression(x_train[0],y_train[0], self.n_lags,neurons_lstm, neurons_dense, self.mask,self.mask_value, self.repeat_vector)
            # Train the model
            times = [0 for x in range(mc*2)]
            cv = [0 for x in range(mc*2)]
            rmse = [0 for x in range(mc*2)]
            nmbe = [0 for x in range(mc*2)]
            zz= 0
            predictions = []
            reales = []
            for z in range(2):
                print('Fold number', z)
                for zz2 in range(mc):
                    time_start = time()
                    model = self.__class__.train_model(model,x_train[z], y_train[z], x_test[z], y_test[z], pacience, batch)
                    times[zz] = round(time() - time_start, 3)

                    #res = self.__class__.predict_model(model, self.n_lags, x_train[z], x_test[z], y_test[z], mean_y, scalar_y)
                    res = self.__class__.predict_model(model, self.n_lags, x_train[z], x_val[z], y_val[z], mean_y, scalar_y)
                    y_pred = res['y_pred']

                    y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))

                    y_real = y_val[z]
                    y_real2 = y_val[z].copy()
                    y_real = np.array(self.scalar_y.inverse_transform(y_real))


                    if self.zero_problem == 'schedule':
                        print('*****Night-schedule fixed******')

                        y_pred[np.where(y_pred < self.inf_limit)[0]]=self.inf_limit
                        y_pred[np.where(y_pred > self.sup_limit)[0]]=self.sup_limit

                        res = super().fix_values_0(times_test[z],
                                                      self.zero_problem, self.limits)


                        y_pred = res['data']
                        index_hour = res['indexes_out']


                        y_predF = y_pred.copy()
                        y_predF = pd.DataFrame(y_predF)
                        y_predF.index = times_test[z]
                        y_realF = y_real.copy()
                        y_realF = pd.DataFrame(y_realF)
                        y_realF.index = times_test[z]


                        predictions.append(y_predF)
                        reales.append(y_realF)

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

                        # l1 = np.where(self.times == self.schedule[0])[0]
                       # l2 = np.where(self.times == self.schedule[1])[0]
                       # pred_list = []
                       # real_list = []
                       # for t in range(len(l1)):
                       #     pred_list.append(np.array(y_pred[l1[t]:l2[t]]))
                       #     real_list.append(np.array(y_real[l1[t]:l2[t]]))

                        #y_pred2 =np.array(list(chain.from_iterable(pred_list)))
                        #y_real2 =np.array(list(chain.from_iterable(real_list)))

                        #Outliers and missing values
                        o = np.where(y_real2<self.inf_limit)[0]

                        if len(o)>0:
                            y_pred1 = np.delete(y_pred1,o,0)
                            y_real1 = np.delete(y_real1,o, 0)

                        cv[zz] = evals(y_pred1, y_real1).cv_rmse(mean_y)
                        rmse[zz] = evals(y_pred1, y_real1).rmse()
                        nmbe[zz] = evals(y_pred1, y_real1).nmbe(mean_y)

                    elif self.zero_problem == 'radiation':
                        print('*****Night-radiation fixed******')
                        place = np.where(names == 'radiation')[0]
                        scalar_rad = self.scalar_x['radiation']

                        y_pred[np.where(y_pred < self.inf_limit)[0]]=self.inf_limit
                        y_pred[np.where(y_pred > self.sup_limit)[0]]=self.sup_limit


                        res = super().fix_values_0(scalar_rad.inverse_transform(x_val[z][:,self.n_lags-1,place]),
                                                      self.zero_problem, self.limits)

                        index_rad = res['indexes_out']

                        y_predF = y_pred.copy()
                        y_predF = pd.DataFrame(y_predF)
                        y_predF.index = times_test[z]
                        y_realF = y_real.copy()
                        y_realF = pd.DataFrame(y_realF)
                        y_realF.index = times_test[z]

                        predictions.append(y_predF)
                        reales.append(y_realF)
                        if len(index_rad) > 0 and self.horizont == 0:
                            y_pred1 = np.delete(y_pred, index_rad, 0)
                            y_real1 = np.delete(y_real, index_rad, 0)
                            y_real2 = np.delete(y_real2, index_rad, 0)
                        elif len(index_rad) > 0 and self.horizont > 0:
                            y_pred1 = np.delete(y_pred, index_rad-self.horizont, 0)
                            y_real1 = np.delete(y_real, index_rad-self.horizont, 0)
                            y_real2 = np.delete(y_real2, index_rad-self.horizont, 0)
                        else:
                            y_pred1 = y_pred
                            y_real1 = y_real

                        #Outliers and missing values
                        o = np.where(y_real2 < self.inf_limit)[0]

                        if len(o)>0:
                            y_pred1 = np.delete(y_pred1,o,0)
                            y_real1 = np.delete(y_real1,o, 0)

                        cv[z] = evals(y_pred1, y_real1).cv_rmse(mean_y)
                        rmse[z] = evals(y_pred1, y_real1).rmse()
                        nmbe[z] = evals(y_pred1, y_real1).nmbe(mean_y)
                    else:
                        y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
                        y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit

                        # Outliers and missing values
                        o = np.where(y_real2 < self.inf_limit)[0]

                        if len(o) > 0:
                            y_pred2 = np.delete(y_pred, o, 0)
                            y_real2 = np.delete(y_real, o, 0)
                        else:
                            y_pred2 = y_pred
                            y_real2 = y_real

                        y_predF = y_pred.copy()
                        y_predF = pd.DataFrame(y_predF)
                        y_predF.index = times_test[z]
                        y_realF = y_real.copy()
                        y_realF = pd.DataFrame(y_realF)
                        y_realF.index = times_test[z]

                        predictions.append(y_predF)
                        reales.append(y_realF)

                        cv[zz] = evals(y_pred2, y_real2).cv_rmse(mean_y)
                        rmse[zz] = evals(y_pred2, y_real2).rmse()
                        nmbe[zz] = evals(y_pred2, y_real2).nmbe(mean_y)

                    zz +=1


            res_final = {'preds': predictions, 'reals':reales, 'times_test':times_test, 'cv_rmse':cv,
                 'nmbe':nmbe, 'rmse':rmse,
                 'times_comp':times}            #res_final['preds']= predictions
            #res_final['reals']=reales
            #res_final['times_train']=times
            #res_final['cv_rmse']=cv
            #res_final['nmbe']=nmbe
            #res_final['rmse']=rmse

            print(("The model with", layers_lstm, " layers lstm,",layers_neurons,'layers dense', neurons_dense, "neurons denses,", neurons_lstm,"neurons_lstm and a pacience of", pacience, "has: \n"
                                                                                                        "The average CV(RMSE) is",
                   np.mean(cv), " \n"
                                "The average NMBE is", np.mean(nmbe), "\n"
                                                                      "The average RMSE is", np.mean(rmse), "\n"
                                                                      "The average time to train is", np.mean(times)))

            z = Queue()
            if type(q)==type(z):
                q.put(np.mean(cv))
            else:
                return (res_final)


        ###################################################################################################
        #FALTARÍA CLASIFICACION !!!!!!!!!!!!!!!!!
        ###################################################################################################


    def train(self, x_train, y_train, x_test, y_test, neurons_lstm, neurons_dense, pacience, batch):
        if self.type=='regression':
            model = self.__class__.built_model_regression(x_train, y_train, self.n_lags,neurons_lstm, neurons_dense)
            time_start = time()
            model_trained = self.__class__.train_model(model, x_train, y_train, x_test, y_test, pacience, batch)
            times = round(time() - time_start, 3)
        else:
            model = self.__class__.built_model_classification(x_train, y_train, self.n_lags,neurons_lstm, neurons_dense)
            time_start = time()
            model_trained = self.__class__.train_model(model, x_train, y_train, x_test, y_test, pacience, batch)
            times = round(time() - time_start, 3)
        res = {'model': model_trained, 'times': times}
        return (res)


    def predict(self, model, x_train,x_val, y_val,mean_y):
        scalar_y = self.scalar_y

        res = self.__class__.predict_model(model, self.n_lags, x_train, x_val, y_val,mean_y, scalar_y)

        y_pred = res['y_pred']

        y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))

        y_real = y_val
        y_real2 = y_val.copy()
        y_real = np.array(self.scalar_y.inverse_transform(y_real))

        if self.zero_problem == 'schedule':
            print('*****Night-schedule fixed******')
            y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
            y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit

            res = super().fix_values_0(y_pred, self.times,
                                       self.zero_problem, self.limits)

            index_hour = res['indexes_out']

            y_predF = y_pred.copy()
            y_predF = pd.DataFrame(y_predF)
            y_predF.index = self.times

            if len(y_pred<=1) and len(index_hour)>0:
                y_pred1= np.nan
                y_real1=y_real
            elif len(y_pred<=1) and len(index_hour)==0:
                y_pred1= y_real
                y_real1=y_real

            else:

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


            # l1 = np.where(self.times == self.schedule[0])[0]
            # l2 = np.where(self.times == self.schedule[1])[0]
            # pred_list = []
            # real_list = []
            # for t in range(len(l1)):
            #     pred_list.append(np.array(y_pred[l1[t]:l2[t]]))
            #     real_list.append(np.array(y_real[l1[t]:l2[t]]))

            # y_pred2 =np.array(list(chain.from_iterable(pred_list)))
            # y_real2 =np.array(list(chain.from_iterable(real_list)))

            # Outliers and missing values
            if len(y_pred1)>0:
                o = np.where(y_real2 < self.inf_limit)[0]
                # o2 = np.where(y_real2==self.sup_limit)[0]
                # o = np.union1d(o1,o2)
                y_pred1 = np.delete(y_pred1, o, 0)
                y_real1 = np.delete(y_real1, o, 0)

            if len(y_pred1)>0:
                cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                nmbe = evals(y_pred1, y_real1).nmbe(mean_y)
                rmse = evals(y_pred1, y_real1).rmse()
                r2 = evals(y_pred1, y_real1).r2()
            else:
                raise NameError('Empty prediction')

        elif self.zero_problem == 'radiation':
            print('*****Night-radiation fixed******')
            place = np.where(x_val.columns == 'radiation')[0]
            scalar_x = self.scalar_x
            scalar_rad = scalar_x['radiation']
            y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
            y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit

            res = super().fix_values_0(y_pred, scalar_rad.inverse_transform(x_val.iloc[:, place]),
                                       self.zero_problem, self.limits)
            #y_pred = res['data']
            index_rad = res['indexes_out']

            y_predF = y_pred.copy()
            y_predF = pd.DataFrame(y_predF)
            y_predF.index = self.times

            # Radiation under the limit
            if len(y_pred <= 1) and len(index_rad) > 0:
                y_pred1 = np.nan
                y_real1 = y_real
            elif len(y_pred <= 1) and len(index_rad) == 0:
                y_pred1 = y_real
                y_real1 = y_real

            else:

                if len(index_rad) > 0 and self.horizont == 0:
                    y_pred1 = np.delete(y_pred, index_rad, 0)
                    y_real1 = np.delete(y_real, index_rad, 0)
                    y_real2 = np.delete(y_real2, index_rad, 0)
                else:
                    y_pred1 = y_pred
                    y_real1 = y_real

            if len(y_pred1)>0:
                # Outliers and missing values
                o = np.where(y_real2 < self.inf_limit)[0]
                # o2 = np.where(y_real2 == self.sup_limit)[0]
                # o = np.union1d(o1, o2)
                y_pred1 = np.delete(y_pred1, o, 0)
                y_real1 = np.delete(y_real1, o, 0)


            if len(y_pred1)>0:
                cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                nmbe = evals(y_pred1, y_real1).nmbe(mean_y)
                rmse = evals(y_pred1, y_real1).rmse()
                r2 = evals(y_pred1, y_real1).r2()
            else:
                raise NameError('Empty prediction')
        else:
            y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
            y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit

            y_predF = y_pred.copy()
            y_predF = pd.DataFrame(y_predF)
            y_predF.index = self.times

            # Outliers and missing values
            o = np.where(y_real2 < self.inf_limit)[0]
                # o2 = np.where(y_real2==self.sup_limit)[0]
                # o = np.union1d(o1,o2)
            y_pred = np.delete(y_pred, o, 0)
            y_real = np.delete(y_real, o, 0)
            if len(y_pred)>0:
                cv = evals(y_pred, y_real).cv_rmse(mean_y)
                rmse = evals(y_pred, y_real).rmse()
                nmbe = evals(y_pred, y_real).nmbe(mean_y)
                r2 = evals(y_pred, y_real).r2()
            else:
                raise NameError('Empty prediction')
            # ).nmbe(mean_y)


        res = {'y_pred': y_predF, 'cv_rmse': cv, 'nmbe': nmbe, 'rmse':rmse,'r2':r2}

        return (res)


#    def optimal_search(self, fold,mc, neurons_dense, neurons_lstm, paciences, batch, mean_y):
#        results = [0 for x in range(len(neurons_lstm) * len(neurons_dense) * len(paciences))]
#        options = {'neurons_dense': [],'neurons_lstm':[], 'pacience': []}
#        w = 0
#        for t in range(len(neurons_dense)):
#            print('##################### Option ####################', w)
#            neuron_dense = neurons_dense[t]
#            for j in range(len(neurons_lstm)):
#                neuron_lstm=neurons_lstm[j]
#                for i in range(len(paciences)):
#                    options['neurons_dense'].append(neuron_dense)
#                    options['neurons_lstm'].append(neuron_lstm)
#                    options['pacience'].append(paciences[i])
#                    res = self.cv_analysis(fold,mc, neuron_lstm , neuron_dense,paciences[i],batch,mean_y)
#                    results[w]=np.mean(res['cv_rmse'])
#                    w +=1
#
#        r1 = results.copy()
#
#        top = []
#        for i in range(10):
#            a = np.where(r1 == np.min(r1))[0]
#            print(a)
#            if len(a) == 1:
#                zz = a[0]
#            else:
#                zz = a[0][0]
#
#            list1 = list()
#            list1.append(r1[zz])
#            list1.append(options['neurons_dense'][zz])
#            list1.append(options['neurons_lstm'][zz])
#            list1.append(options['pacience'][zz])
#            r1.remove(np.min(r1))
#            top.append(list1)
#
#        res = {'errors': results, 'options': options, '10_best': top}
#        return(res)
#    @staticmethod
#    def wait_process_done(f, wait_time=1):
#    # Monitor the status of another process
#        if not f.is_alive():
#            a=0
#            time.sleep(wait_time)
#            print('foo is done.')
#        else:
#            a=1
#            print('Done')
#        return(a)


    def optimal_search(self, fold, mc, neurons_dense, neurons_lstm, paciences, batch, mean_y,parallel,top):
        results = [0 for x in range(len(neurons_lstm) * len(neurons_dense) * len(paciences))]
        deviations = [0 for x in range(len(neurons_lstm) * len(neurons_dense) * len(paciences))]

        #results = [0 for x in range(len(neurons) * len(paciences))]
        options = {'neurons_dense': [], 'neurons_lstm': [], 'pacience': []}
        w = 0
        contador=len(neurons_lstm) * len(neurons_dense) * len(paciences)-1
        if parallel==0:
            for t in range(len(neurons_dense)):
                print('##################### Option ####################', w)
                neuron_dense = neurons_dense[t]
                for j in range(len(neurons_lstm)):
                    neuron_lstm = neurons_lstm[j]
                    for i in range(len(paciences)):
                        options['neurons_dense'].append(neuron_dense)
                        options['neurons_lstm'].append(neuron_lstm)
                        options['pacience'].append(paciences[i])
                        res = self.cv_analysis(fold, mc, neuron_lstm, neuron_dense, paciences[i], batch, mean_y,dict())
                        results[w] = np.mean(res['cv_rmse'])
                        deviations[w] = np.std(res['cv_rmse'])
                        w += 1
        elif parallel>=2:
            processes = []
            res2 = []
            dev2 = []
            z = 0
            #with Manager() as manager:
            #    res_final = manager.dict()
            ##res_final = dict()
            #res_final['preds'] = []
            #res_final['reals'] = []
            #res_final['times_train'] = []
            #res_final['cv_rmse'] = []
            #res_final['nmbe'] = []
            #res_final['rmse'] = []
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
                            multiprocessing.set_start_method('fork')
                            p = Process(target=self.cv_analysis,
                                        args=(fold,mc, neuron_lstm, neuron_dense, paciences[i], batch, mean_y, q))
                            p.start()

                            processes.append(p)
                            z1 =z+ 1
                        if z == parallel and w < contador:
                            for p in processes:
                                p.join()
                                #res2.append(dict(res_final))
                            for v in range(len(processes)):
                                res2.append(q.get()[0])
                                res2.append(q.get()[1])
                            #for i in range(parallel):
                            #    r = res2[i]
                            #    results.append(np.mean(r['cv_rmse']))

                         #   a=0
                         #   while a==0:
                         #       a=LSTM_model.wait_process_done(p)

                            #with Manager() as manager:
                            #    res_final = manager.dict()
                            ##res_final=dict()
                            #res_final['preds'] = []
                            #res_final['reals'] = []
                            #res_final['times_train'] = []
                            #res_final['cv_rmse'] = []
                            #res_final['nmbe'] = []
                            #res_final['rmse'] = []
                            processes=[]
                            multiprocessing.set_start_method('fork')
                            q = Queue()
                            p = Process(target=self.cv_analysis,
                                        args=(fold, mc, neuron_lstm, neuron_dense, paciences[i], batch, mean_y, q))
                            p.start()

                            processes.append(p)
                            z1 = 1

                        elif w==contador:
                            p = Process(target=self.cv_analysis,
                                        args=(fold, mc, neuron_lstm, neuron_dense, paciences[i], batch, mean_y, q))
                            p.start()

                            processes.append(p)

                            for p in processes:
                                p.join()

                                #wait_process_done(p)
                                #res2.append(dict(res_final))
                            for v in range(len(processes)):
                                res2.append(q.get()[0])
                                dev2.append(q.get()[1])
                            #a=0
                            #while a==0:
                            #    a=LSTM_model.wait_process_done(p)
                            #processes=[]
                            #p = Process(target=self.cv_analysis,
                            #            args=(fold, mc, neuron_lstm, neuron_dense, paciences[i], 64, mean_y, res_final))
                            #p.start()
                            #processes.append(p)
                            #z = 1
                            #for i in range(parallel):
                            #    r = res2[i]
                            #    results.append(np.mean(r['cv_rmse']))
                        z=z1

                        w += 1
            results = res2
            deviations = dev2
        else:
            raise NameError('Option not considered')

        r1 = results.copy()
        d1 = deviations.copy()
        print(r1)
        # indx = [0 for x in range(10)]
        top_results = []
        top_results = {'error': [], 'std': [], 'neurons_dense': [],'neurons_lstm':[], 'pacience': []}

        for i in range(top):
            a = np.where(r1 == np.min(r1))[0]
            print(a)
            if len(a) == 1:
                zz = a[0]
            else:
                zz = a[0][0]

            # list1=[]
            # list1.append(r1[zz])
            top_results['error'].append(r1[zz])
            top_results['std'].append(d1[zz])
            top_results['neurons_dense'].append(options['neurons_dense'][zz])
            top_results['neurons_lstm'].append(options['neurons_lstm'][zz])
            top_results['pacience'].append(options['pacience'][zz])

            # list1.append()
            # list1.append(options['neurons'][zz])
            # list1.append(options['pacience'][zz])
            r1.remove(np.min(r1))
            d1.remove(d1[zz])
            options['neurons_dense'].pop(zz)
            options['neurons_lstm'].pop(zz)
            options['pacience'].pop(zz)
            # top_results.append(list1)
        print('Process finished!!!')

#       r1 = results.copy()
#       print(results)
#       top_results = []
#       for i in range(top):
#           a = np.where(r1 == np.min(r1))[0]
#           print(a)
#           if len(a) == 1:
#               zz = a[0]
#           else:
#               zz = a[0][0]

#           list1 = list()
#           list1.append(r1[zz])
#           list1.append(options['neurons_dense'][zz])
#           list1.append(options['neurons_lstm'][zz])
#           list1.append(options['pacience'][zz])
#           r1.remove(np.min(r1))
#           top_results.append(list1)
#       print('Process finished!!!')
        res = {'errors': results, 'options': options, 'best': top_results}
        return(res)
