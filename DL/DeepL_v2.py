from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math
import collections

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
        Class for defining the bases and some function for preprocessing for deepl learning models
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

        '''
        data: dataframe with the data
        horizont: how many steps are considered to the future
        scalar x: scalar object empty to be fill with the trained scalar for x
        scalar y: scalar object empty to be fill with the trained scalar for y
        zero_problem: relation with the night and day schedule. Can be nothing, schedule or radiation
        limits: threshold to evaluate the outputs: f.e. radiation > 0.001
        times: dates object
        pos_y: number of colunm or columns where the output data are
        n_lags: time lags considered to train the model
        mask: masking true or false
        mask_value: if mask is true the value for masking
        inf_limit: lower limit wihitn the output of the model must be
        sup_limit:upper limit wihitn the output of the model must be
        nanes: columns labels
        extract_zero: True if we want to discard the moment when real values are 0
        '''

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

        #Operator defining how long will be the slices of data
        step = int(x.shape[0]/fold)
        w = 0
        w2 = step
        indexes = []
        try:
            while w2 < x.shape[0]:
                #Dividing the sample into val and train (and val into test and val)
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
                if restriction.shape[1]==1:
                    restriction=pd.Series(restriction)
                else:
                    r=np.concatenate(restriction)
                    restriction=pd.Series(r)

                #If third value is weekend the weekend are eliminated for subsequent evaluation
                if limit[2]=='weekend':
                    wday = restriction.dt.weekday
                    ii1 = np.where(wday>4)[0]

                    hours = restriction.dt.hour
                    ii = np.where((hours < limit1) | (hours > limit2))[0]
                    ii = np.union1d(ii1, ii)

                else:
                    hours = pd.Series(restriction).dt.hour
                    ii = np.where((hours < limit1) | (hours > limit2))[0]


            except:
                raise NameError('Zero_problem and restriction incompatibles')
        elif zero_problem == 'radiation':
            #If radiation is the one, only the values where the radiation is less than the limit are considered
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
            #If we are very close to the final, we delete the final empty column (and measure the gap)
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
    def cortes_onebyone(x, D, lim):
        '''
        :param x:
        :param D: length of data
        :param lim: dimension of the curves
        :return: data divided in curves of specific length but one by one time step
        The same as cortes but we move one step at a time
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
        #We select the data without the required rows in the beginning
        t = t.iloc[look_back:, :]
        t.set_index('id', inplace=True)
        pred_value = new_data.copy()
        #We select the data without the required rows at the end
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
        Introduction of lags moving the sample (***Data for be lagged at the end of the columns)

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
        if isinstance(self.pos_y, collections.abc.Sized):
            for t in range(len(self.pos_y)):
                inf = np.where(self.data.iloc[:, self.pos_y[t]] < self.inf_limit[t])[0]
                sup = np.where(self.data.iloc[:, self.pos_y[t]] > self.sup_limit[t])[0]
                if len(inf) > 0:
                    self.data.iloc[inf, self.pos_y[t]] = np.repeat(self.inf_limit[t], len(inf))
                if len(sup) > 0:
                    self.data.iloc[sup, self.pos_y[t]] = np.repeat(self.sup_limit[t], len(sup))
        else:
            inf = np.where(self.data.iloc[:,self.pos_y] < self.inf_limit[0])[0]
            sup = np.where(self.data.iloc[:,self.pos_y] > self.sup_limit[0])[0]
            if len(inf)>0:
                self.data.iloc[inf, self.pos_y] = np.repeat(self.inf_limit[0], len(inf))
            if len(sup)>0:
                self.data.iloc[sup, self.pos_y] = np.repeat(self.sup_limit[0], len(sup))

    def adapt_horizont(self, onebyone):
        '''
        Move the data sample to connected the y with the x based on the future selected and the possible steps
        After introduce_lags
        n_steps=step in the future for predicting
        onebyone: although we want to predict 5 steps in the future (we move the sampke one by one)
        '''
        if self.horizont == 0:
            self.data = self.data
        else: # if we have horizont>0, make a jump to the future
            X = self.data.drop(self.data.columns[self.pos_y], axis=1)
            y = self.data.iloc[:, self.pos_y]
            for t in range(self.horizont):
                y = y.drop(y.index[0], axis=0)
                X = X.drop(X.index[X.shape[0] - 1], axis=0)

            X = X.reset_index(drop=True)
            X.index = y.index

        if self.type == 'series': #if we are working with series the y will have several columns (future time steps)
            X = self.data.drop(self.data.columns[self.pos_y], axis=1)
            y = self.data.iloc[:, self.pos_y]
            #We create the matrix y with the first step and then columns with the data moved to math the future steps
            y1 = y.copy().drop(y.index[len(y)-1-self.n_steps:len(y)], axis=0)
            ys = np.zeros((y1.shape[0], self.n_steps))
            for t in range(self.n_steps-1):
                a=y.copy().drop(y.index[0:(t+1)])
                if self.n_steps - 2 == 1 and (t-1)==0:
                    a = a
                elif self.n_steps - 2 == 1 and (t-1)<0:
                    a=a.drop(a.index[len(a)-1+t],axis=0)
                else:
                    a=a.drop(a.index[len(a)-1-(self.n_steps-2-1+t):len(y)],axis=0)
                ys[:,t+1]=a
            ys[:,0]=y1
            y=ys.copy()

            #Merge inputs and outputs
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
