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
from keras.layers import Masking
from datetime import datetime
from time import time
import skfda
import math
import multiprocessing
from multiprocessing import Process,Manager,Queue

class ML:
    def info(self):
        print(('Super class to built different machine learning models. This class has other more specific classes associated with it  \n'
              'Positions_y is required to be 0 or len(data) \n'
               'Zero problem is related with the night-day issue \n'
               'Horizont is focused on the future \n'
               'Limit is the radiation limit and schedule is the working hours'
              ))

    def __init__(self, data,horizont, scalar_y,scalar_x, zero_problem,limits, times, pos_y, n_lags, mask, mask_value, inf_limit,sup_limit ):
        self.data = data
        self.horizont = horizont
        self.scalar_y = scalar_y
        self.scalar_x = scalar_x
        self.zero_problem = zero_problem
        self.times = times
        self.limits = limits
        self.pos_y = pos_y
        self.n_lags = n_lags
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

                X_val.append(a.iloc[range(len(a)-math.ceil(len(a)/2), len(a)-1)])
                X_test.append(a.drop(a.index[range(len(a)-math.floor(len(a)/2), len(a))]))
                X_train.append(x.drop(range(w,w2)))


                a = y.iloc[range(w,w2)]
                Y_val.append(a.iloc[range(len(a)-math.ceil(len(a)/2), len(a)-1)])
                Y_test.append(a.drop(a.index[range(len(a)-math.floor(len(a)/2), len(a))]))
                Y_train.append(y.drop(range(w,w2)))
                indexes.append(np.array([w2-math.ceil(len(a)/2)+1,w2]))
                w=w2
                w2+=step
                if(w2 > x.shape[0] and w < x.shape[0]):
                    w2 = x.shape[0]
        except:
            raise NameError('Problems with the sample division in the cv classic')
        res = {'x_test': X_test, 'x_train':X_train,'x_val':X_val, 'y_test':Y_test, 'y_train':Y_train, 'y_val':Y_val,
            'indexes':indexes}
        return(res)

    @staticmethod
    def fix_values_0(restriction, zero_problem, limit):
        '''
        :param restriction: schedule hours or irradiance variable depending on the zero_problem
        :param zero_problem: schedule or radiation
        :param limit: limit hours or radiation limit
        :return: the indexes where the data are not in the correct time schedule or is below the radiation limit
        '''
        if zero_problem == 'schedule':
            try:
                limit1 = limit[0]
                limit2 = limit[1]
                hours = restriction.hour
                ii = np.where(hours < limit1 | hours > limit2)[0]
                ii= ii[ii>=0]
            except:
                raise NameError('Zero_problem and restriction incompatibles')
        elif zero_problem == 'radiation':
            try:
                rad = restriction
                ii = np.where(rad <= limit)[0]
                ii = ii[ii >= 0]
                # data.iloc[ii] = 0
            except:
                raise NameError('Zero_problem and restriction incompatibles')
        else:
            ii=[]
            'Unknown situation with nights'

        res = {'indexes_out': ii}
        return (res)

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
        while i <= D:
            if D - i < lim:
                Y = np.delete(Y, s - 1, 1)
                break
            else:
                Y[:, s] = x[i:(i + lim)]
                i += lim
                s += 1
                if i == D:
                    break

        return (Y)


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
            names_lag[i]=names[i] + '.'+ str(lag)
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

    def scalating(self, scalar_limits,groups,x,y):
        '''
        :param scalar_limits: limit of the scalar data
        :param groups: groups defining the different variable groups to be scaled together
        :return: data scaled depending if x, y or both are scaled
        '''
        scalars = dict()
        names = list(groups.keys())
        if x==True and y==True:
            try:
                for i in range(len(groups)):
                    scalars[names[i]] = MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1]))
                    selec =groups[names[i]]
                    d = self.data.iloc[:,selec]
                    if(len(selec)>1):
                        scalars[names[i]].fit(np.concatenate(np.array(d)).reshape(-1,1))
                    else:
                        scalars[names[i]].fit(np.array(d).reshape(-1, 1))
                    for z in range(len(selec)):
                        self.data.iloc[:,selec[z]]= scalars[names[i]].transform(pd.DataFrame(d.iloc[:,z]))[:,0]
            except:
                raise NameError('Problems with the scalar by groups of variables')
            scalar_y = MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1]))
            scalar_y.fit(pd.DataFrame(self.data.iloc[:,self.pos_y]))
            self.data.iloc[:,self.pos_y] = scalar_y.transform(pd.DataFrame(self.data.iloc[:,self.pos_y]))[:,0]

            self.scalar_y=scalar_y
            self.scalar_x = scalars
        elif x==True and y==False:
            try:
                for i in range(len(groups)):
                    scalars[names[i]] = MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1]))
                    selec  =groups[names[i]]
                    d = self.data.iloc[:,selec]
                    if (len(selec) > 1):
                        scalars[names[i]].fit(np.concatenate(np.array(d)).reshape(-1, 1))
                    else:
                        scalars[names[i]].fit(np.array(d).reshape(-1, 1))
                    for z in range(len(selec)):
                        self.data.iloc[:,selec[z]]= scalars[names[i]].transform(pd.DataFrame(d.iloc[:,z]))[:,0]

                self.scalar_x = scalars
            except:
                raise NameError('Problems with the scalar by groups of variables')
        elif y==True and x==False:
            scalar_y = MinMaxScaler(feature_range=(scalar_limits[0], scalar_limits[1]))
            scalar_y.fit(pd.DataFrame(self.data.iloc[:, self.pos_y]))
            self.data.iloc[:, self.pos_y] = scalar_y.transform(pd.DataFrame(self.data.iloc[:, self.pos_y]))[:,0]

            self.scalar_y = scalar_y


    def missing_values_remove(self):
        self.data = self.data.dropna()

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

    def fda_outliers(self, freq):
        '''
        :param freq: amount of values in a hour
        :return: the variable y with missing value in the days considered as outliers
        '''
        step = int(60 / freq)
        y = self.data.iloc[:, self.pos_y]
        hour = self.times.hour
        start = np.where(hour == 0)[0][0]

        if np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - 1] > np.where(hour == 23)[0][
            len(np.where(hour == 23)[0]) - 1]:
            end = np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - step]
        elif np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - 1] < np.where(hour == 23)[0][
            len(np.where(hour == 23)[0]) - 1]:
            if np.sum(hour[np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - 1]:np.where(hour == 23)[0][
                len(np.where(hour == 23)[0]) - 1]] == 23) == step:
                end = len(y)
            else:
                end = np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - step]
        else:
            end = []
            raise NameError('Problem with the limit of sample creating the functional sample')

        y1 = y.iloc[range(start)]
        y2 = y.iloc[range(end, len(y))]

        y_short = y.iloc[range(start,end)]


        fd_y = ML.cortes(y_short, len(y_short), int(24 * step)).transpose()

        grid = []
        for t in range(int(24 * step)):
            grid.append(t)

        fd_y2 = fd_y.copy()
        missing = []
        for t in range(fd_y.shape[0]):
            if np.sum(np.isnan(fd_y[t, :])) > 0:
                missing.append(t)

        if len(missing) > 0:
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

        out_detector1 = skfda.exploratory.outliers.IQROutlierDetector(factor=4,
                                                                      depth_method=skfda.exploratory.depth.BandDepth())  # MSPlotOutlierDetector()
        out_detector2 = skfda.exploratory.outliers.LocalOutlierFactor(n_neighbors=int(fd_y2.shape[0] / 6))
        oo1 = out_detector1.fit_predict(fd1)
        oo2 = out_detector2.fit_predict(fd1)
        o1 = np.where(oo1 == -1)[0]
        o2 = np.where(oo2 == -1)[0]
        o_final = np.intersect1d(o1, o2)

        print(len(o_final))
        # diff = 0
        if len(o_final) > 0:
            out = index2[o_final]

            for t in range(len(o_final)):
                w = np.empty(fd_y.shape[1])
                w[:] = np.nan
                fd_y[out[t], :] = w

        Y = fd_y.flatten()

        Y = pd.concat([pd.Series(y1), pd.Series(Y), pd.Series(y2)], axis=0)

        print(Y.shape)
        print(self.data.shape)

        Y.index = self.data.index
        self.data.iloc[:, self.pos_y] = Y

        print('Data have been modified converting the outliers days in missing values!')


class MLP(ML):


    def info(self):
        print(('Class to built MLP models. \n'
              'All the parameters comes from the ML class except the activation functions'))

    def __init__(self,data,horizont, scalar_y,scalar_x, zero_problem,limits, times, pos_y, n_lags, mask, mask_value, inf_limit,sup_limit, type):
        super().__init__(data,horizont, scalar_y,scalar_x, zero_problem,limits, times, pos_y, n_lags, mask, mask_value, inf_limit,sup_limit)
        self.type = type


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
    def mlp_regression(layers, neurons,  inputs,mask, mask_value):
        '''
        :param inputs:amount of inputs
        :param mask:True or false
        :return: the MLP architecture
        '''
        try:
            ANN_model = Sequential()
            if mask==True:
                ANN_model.add(Masking(mask_value=mask_value, input_shape=(inputs)))
                ANN_model.add(Dense(inputs,kernel_initializer='normal', input_dim=inputs,
                                activation='relu'))
            else:
                ANN_model.add(Dense(inputs, kernel_initializer='normal', input_dim=inputs,
                                activation='relu'))
            for i in range(layers):
                ANN_model.add(Dense(neurons[i], kernel_initializer='normal', activation='relu'))

            # The Output Layer :
            ANN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
            # Compile the network :
            ANN_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
            # ANN_model.summary()
            return(ANN_model)
        except:
            raise NameError('Problems building the MLP')


    def cv_analysis(self,fold, neurons, pacience, batch, mean_y, q=[]):
        '''
        :param fold: divisions in cv analysis
        :param q: a Queue to paralelyse or empty list to do not paralyse
        :return: predictions, real values, errors and the times needed to train
        '''
        scalar_y =self.scalar_y
        names = self.data.columns


        print('##########################'
              '################################'
              'CROSS-VALIDATION'
              '#############################3'
              '################################')


        layers = len(neurons)
        x =pd.DataFrame(self.data.drop(self.data.columns[self.pos_y],axis=1))
        y =pd.DataFrame(self.data.iloc[:,self.pos_y])
        x=x.reset_index(drop=True)
        y=y.reset_index(drop=True)

        ####################################3
        if self.zero_problem=='radiation':
            place = np.where(x.columns == 'radiation')[0]
            scalar_rad = self.scalar_x['radiation']
            res = super().fix_values_0(scalar_rad.inverse_transform(x.iloc[:, place]), self.zero_problem,
                                       self.limits)

            index_rad = res['indexes_out']
            if len(index_rad)>0 and self.horizont==0:
                x=x.drop(x.index[index_rad], axis=0)
                y=y.drop(y.index[index_rad], axis=0)
            elif len(index_rad)>0 and self.horizont>0:
                x=x.drop(x.index[index_rad-self.horizont], axis=0)
                y=y.drop(y.index[index_rad-self.horizont], axis=0)
            else:
                pass
            print('*****Night-radiation fixed******')
        elif self.zero_problem=='schedule':
            res = super().fix_values_0(self.times, self.zero_problem, self.limits)

            index_hour = res['indexes_out']
            if len(index_hour)>0 and self.horizont==0:
                x=x.drop(x.index[index_hour], axis=0)
                y=y.drop(y.index[index_hour], axis=0)
            elif len(index_hour)>0 and self.horizont>0:
                x=x.drop(x.index[index_hour-self.horizont], axis=0)
                y=y.drop(y.index[index_hour-self.horizont], axis=0)
            else:
                pass
            print('*****Night-schedule fixed******')
        else:
            pass
        ######################################3

        res = super().cv_division(x, y, fold)

        x_test =res['x_test']
        x_train=res['x_train']
        x_val=res['x_val']
        y_test=res['y_test']
        y_train = res['y_train']
        y_val = res['y_val']
        indexes = res['indexes']

        times_test = []
        tt = self.times
        for t in range(len(indexes)):
            times_test.append(tt[indexes[t][0]:indexes[t][1]])


        if self.type=='regression':
            model= self.__class__.mlp_regression(layers, neurons, x_train[0].shape[1],self.mask, self.mask_value)

            # Checkpoitn callback
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pacience)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            # Train the model
            times=[0 for x in range(fold)]
            cv=[0 for x in range(fold)]
            rmse=[0 for x in range(fold)]
            nmbe = [0 for x in range(fold)]
            predictions=[]
            reales = []
            for z in range(fold):
                print('Fold number', z)
                x_t = pd.DataFrame(x_train[z]).reset_index(drop=True)
                y_t = pd.DataFrame(y_train[z]).reset_index(drop=True)
                test_x = pd.DataFrame(x_test[z]).reset_index(drop=True)
                test_y = pd.DataFrame(y_test[z]).reset_index(drop=True)
                val_x = pd.DataFrame(x_val[z]).reset_index(drop=True)
                val_y = pd.DataFrame(y_val[z]).reset_index(drop=True)
                time_start = time()
                model.fit(x_t, y_t, epochs=2000, validation_data=(test_x, test_y), callbacks=[es, mc],batch_size=batch )
                times[z] = round(time() - time_start, 3)
                y_pred = model.predict(val_x)
                y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
                y_real = np.array(self.scalar_y.inverse_transform(val_y))
                y_real2 = np.array(val_y.copy())

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

                if self.mask==True:
                    o = np.where(y_real2 < self.inf_limit)[0]
                    if len(o)>0:
                        y_pred = np.delete(y_pred, o, 0)
                        y_real = np.delete(y_real, o, 0)
                cv[z] = evals(y_pred, y_real).cv_rmse(mean_y)
                rmse[z] = evals(y_pred, y_real).rmse()
                nmbe[z] = evals(y_pred, y_real).nmbe(mean_y)

            res={'preds': predictions, 'reals':reales, 'times_test':times_test, 'cv_rmse':cv,
                 'nmbe':nmbe, 'rmse':rmse,
                 'times_comp':times}
            print(("The model with", layers," layers", neurons, "neurons and a pacience of",pacience,"has: \n"
            "The average CV(RMSE) is", np.mean(cv), " \n"
            "The average NMBE is", np.mean(nmbe), "\n"
            "The average RMSE is", np.mean(rmse), "\n"
            "The average time to train is", np.mean(times)))

            z = Queue()
            if type(q)==type(z):
                q.put(np.array([np.mean(cv), np.std(cv)]))
            else:
                return (res)


        else:
            #data2 = self.data
            #yy = data2.iloc[:,self.pos_y]
            #yy = pd.Series(yy, dtype='category')
            #n_classes = len(yy.cat.categories.to_list())
            #model = self.__class__.mlp_classification(layers, neurons,x_train[0].shape[1], n_classes)

            ####################################################################

            #EN PROCESOO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


            ##########################################################################

    def optimal_search(self, neurons, paciences,batch, fold,mean_y, parallel, top):
        '''
        :param fold: division in cv analyses
        :param parallel: True or false (True to linux)
        :param top: the best options yielded
        :return: the options with their results and the top options
        '''
        results = [0 for x in range(len(neurons) * len(paciences))]
        deviations = [0 for x in range(len(neurons) * len(paciences))]
        options = {'neurons':[], 'pacience':[]}
        w=0
        contador= len(neurons) * len(paciences)-1
        if parallel == 0:
            for t in range(len(neurons)):
                print('##################### Option ####################', w)
                neuron = neurons[t]

                for i in range(len(paciences)):
                    options['neurons'].append(neuron)
                    options['pacience'].append(paciences[i])
                    res = self.cv_analysis( fold, neuron , paciences[i],batch,mean_y)
                    results[w]=np.mean(res['cv_rmse'])
                    deviations[w]=np.std(res['cv_rmse'])
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
                                    args=(fold, neuron, paciences[i], batch, mean_y, q))
                        p.start()
                        processes.append(p)
                        z1 =z+ 1
                    elif z == parallel and w < contador:
                        for p in processes:
                            p.join()
                        for v in range(len(processes)):
                            res2.append(q.get()[0])
                            res2.append(q.get()[1])

                        processes = []
                        multiprocessing.set_start_method('fork')
                        #multiprocessing.set_start_method('spawn', force=True)
                        q = Queue()
                        p = Process(target=self.cv_analysis,
                                    args=(fold, neuron, paciences[i], batch, mean_y, q))
                        p.start()
                        processes.append(p)
                        z1 = 1
                    elif w >= contador:
                        p = Process(target=self.cv_analysis,
                                    args=(fold, neuron, paciences[i], batch, mean_y, q))
                        p.start()
                        processes.append(p)
                        for p in processes:
                            p.join()
                            #res2.append(q.get())
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
        top_results = {'error':[], 'std':[], 'neurons':[], 'pacience':[]}

        for i in range(top):
                a = np.where(r1==np.min(r1))[0]
                print(a)
                if len(a)==1:
                    zz = a[0]
                else:
                    zz= a[0][0]

                top_results['error'].append(r1[zz])
                top_results['std'].append(d1[zz])
                top_results['neurons'].append(options['neurons'][zz])
                top_results['pacience'].append(options['pacience'][zz])

                r1.remove(np.min(r1))
                d1.remove(d1[zz])
                options['neurons'].pop(zz)
                options['pacience'].pop(zz)

        print('Process finished!!!')


        res = {'errors': results,'options':options, 'best': top_results}
        return(res)




    def train(self, neurons, pacience, batch,x_train, x_test, y_train, y_test):
        '''
        :param x_train: x to train
        :param x_test: x to early stopping
        :param y_train: y to train
        :param y_test: y to early stopping
        :return: trained model and the time needed to train
        '''
        layers = len(neurons)
        model = self.__class__.mlp_regression(layers, neurons,self.data.shape[1]-1, self.mask, self.mask_value)

        # Checkpoint callback
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pacience)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        time_start = time()
        model.fit(x_train, y_train, epochs=2000, validation_data=(x_test, y_test),
                  callbacks=[es, mc],batch_size=batch)
        times = round(time() - time_start, 3)


        res = {'model':model, 'times':times}
        return(res)

    def predict(self, model,x_val, y_val,mean_y):
        '''
        :param model: trained model
        :param x_val: x to predict
        :param y_val: y to predict
        :return: predictions with the errors depending of zero_problem
        '''
        x_val=x_val.reset_index(drop=True)
        y_val=y_val.reset_index(drop=True)
        scalar_y = self.scalar_y

        y_pred = model.predict(pd.DataFrame(x_val))
        y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
        y_real = y_val
        y_real2 = np.array(y_val.copy())
        y_real = np.array(self.scalar_y.inverse_transform(y_real))

        if self.zero_problem == 'schedule':
            print('*****Night-schedule fixed******')
            y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
            y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit

            res = super().fix_values_0( self.times,  self.zero_problem, self.limits)
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
                if self.mask == True:
                    o = np.where(y_real2 < self.inf_limit)[0]
                    if len(o)>0:
                        y_pred1 = np.delete(y_pred1, o, 0)
                        y_real1 = np.delete(y_real1, o, 0)
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
                if self.mask == True and len(y_pred1)>0:
                    o = np.where(y_real2 < self.inf_limit)[0]
                    if len(o)>0:
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
            y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
            y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit

            place = np.where(x_val.columns == 'radiation')[0]
            scalar_x = self.scalar_x
            scalar_rad = scalar_x['radiation']

            res = super().fix_values_0(scalar_rad.inverse_transform(x_val.iloc[:, place]),
                                          self.zero_problem, self.limits)
            index_rad=res['indexes_out']

            index_rad = res['indexes_out']
            y_predF = y_pred.copy()
            y_predF = pd.DataFrame(y_predF)
            y_predF.index = self.times

            if len(y_pred<=1) and len(index_rad)>0:
                y_pred1= np.nan
                y_real1=y_real
            elif len(y_pred<=1) and len(index_rad)==0:
                y_pred1= y_real
                y_real1=y_real
                if self.mask == True:
                    o = np.where(y_real2 < self.inf_limit)[0]
                    if len(o)>0:
                        y_pred1 = np.delete(y_pred1, o, 0)
                        y_real1 = np.delete(y_real1, o, 0)
            else:

                if len(index_rad) > 0 and self.horizont == 0:
                    y_pred1 = np.delete(y_pred, index_rad, 0)
                    y_real1 = np.delete(y_real, index_rad, 0)
                    y_real2 = np.delete(y_real2, index_rad, 0)
                else:
                    y_pred1 = y_pred
                    y_real1 = y_real

                if self.mask == True and len(y_pred1)>0:
                    o = np.where(y_real2 < self.inf_limit)[0]
                    if len(o)>0:
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

            if self.mask == True:
                o = np.where(y_real2 < self.inf_limit)[0]
                if len(o)>0:
                    y_pred = np.delete(y_pred, o, 0)
                    y_real = np.delete(y_real, o, 0)
            if len(y_pred)>0:
                cv = evals(y_pred, y_real).cv_rmse(mean_y)
                rmse = evals(y_pred, y_real).rmse()
                nmbe = evals(y_pred, y_real).nmbe(mean_y)
                r2 = evals(y_pred, y_real).r2()
            else:
                raise NameError('Empty prediction')

        res = {'y_pred': y_predF,  'cv_rmse': cv, 'nmbe': nmbe, 'rmse':rmse,'r2':r2}
        return(res)


class XGB(ML):
    def info(self):
        print(('Class to built XGB models. \n'))

    def __init__(self,data,scalar_y,scalar_x,zero_problem,limit,schedule,times):
        super().__init__(data,scalar_y, scalar_x,zero_problem,limit,schedule, times, pos_y)



class SVM(ML):
    def info(self):
        print(('Class to built SVM models. \n'))

    def __init__(self,data,scalar_y,scalar_x,zero_problem,limit,schedule,times):
        super().__init__(data,scalar_y, scalar_x,zero_problem,limit,schedule, times, pos_y)




class RF(ML):
    def info(self):
        print(('Class to built RF models. \n'))

    def __init__(self,data,scalar_y,scalar_x,zero_problem,limit,schedule,times):
        super().__init__(data,scalar_y, scalar_x,zero_problem,limit,schedule, times, pos_y)




#from itertools import permutations
#
#n = np.array([10,100,1000,50,500])
#l=[]
#perm = permutations(n)
#
#a = list(list(perm),list(np.array[10,100]))