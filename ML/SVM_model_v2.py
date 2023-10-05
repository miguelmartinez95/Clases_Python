from sklearn.preprocessing import MinMaxScaler
from errors import Eval_metrics as evals
import pandas as pd
import numpy as np
from time import time
import collections
import multiprocessing
from multiprocessing import Process,Queue
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from ML_v2 import ML
from datetime import datetime
from MyProblem_svm import MyProblem_svm
from sklearn.multioutput import MultiOutputRegressor
from sklearn import svm
from MyRepair_svm import MyRepair_svm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.aasf import AASF
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.optimize import minimize
from pymoo.core.problem import starmap_parallelized_eval

class SVM(ML):
    def info(self):
        print(('Class to built SVM models. \n'))
    def __init__(self,data,horizont, scalar_y,scalar_x, zero_problem,limits,extract_cero, times, pos_y, n_lags, n_steps,mask, mask_value, inf_limit,sup_limit,weights, type):
        super().__init__(data,horizont, scalar_y,scalar_x, zero_problem,limits, extract_cero, times, pos_y, n_lags, n_steps,mask, mask_value, inf_limit,sup_limit,type)
        self.type = type
        self.weights = weights

        '''
        horizont: distance to the present: I want to predict the moment four steps in future
        scalar_y, scalar_x: empty lists to save the object fit to the data
        zero_problem: schedule, radiation o else. Adjust the result to the constraints
        limits: limits based on the zero problems (hours, radiation limits, etc)
        extract_zero: Logic, if we want to consider or not the moment when real data is 0 (True are deleted)
        times: dates
        pos_y: column or columns where the y is located (np.array([]))
        n_lags: times that the variables must be lagged
        mask: logic if we want to mask the missing values
        mask_value: specific value for the masking
        inf_limit: lower accepted limits for the estimated values
        sup_limits: upper accepted limits for the estimated values
        weights: weights based on the error in mutivariable case (some error must be more weighted)
        type: regression or classification
        '''

    @staticmethod
    def complex_svm(C_svm,epsilon_svm, max_C, max_epsilon):
        '''
        :param C_svm: value of C
        :param epsilon_svm: epsilon value
        :param max_N: maximun neurons in the network
        :param max_H: maximum hidden layers in the network
        :return: complexity of the model
        '''

        F = 0.75 * (C_svm / max_C) + 0.25 * (epsilon_svm/max_epsilon)
        return F


    @staticmethod
    def svm_cv_division(x,y, pos_y, fold,values):
        '''
        Division de la muestra en trozos según fold para datos normales y no recurrentes
        :param values specific values to divide the sample. specific values of a variable to search division
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        :param fold: division for cv_analysis
        :return: data divided into train, test and validations in addition to the indexes division for a CV analysis
        '''
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        X_test = []
        X_train = []
        Y_test = []
        Y_train = []

        if any(pos_y)==0:
            data=pd.concat([y,x], axis=1)
        else:
            data = pd.concat([x,y], axis=1)

        if values:
            indexes = []
            place = values[2]
            var = data.iloc[:, place]
            for t in range(values[0]):
                if len(place) == 1:
                    w = np.where(var == values[1][t])[0][0]
                    w2 = np.where(var == values[1][t])[0][len(np.where(var == values[1][t])[0]) - 1]
                elif len(place) == 2:
                    w = np.where((var.iloc[:, 0] == values[1][:,0][t]) & (var.iloc[:, 1] == values[1][:,1][t]))[0][0]
                    w2 = np.where((var.iloc[:, 0] == values[1][:,0][t]) & (var.iloc[:, 1] == values[1][:,1][t]))[0][
                        len(np.where((var.iloc[:, 0] == values[1][:,0][t]) & (var.iloc[:, 1] == values[1][:,1][t]))[0]) - 1]
                elif len(place) == 3:
                    w = np.where((var.iloc[:, 0] == values[1][:,0][t]) & (var.iloc[:, 1] == values[1][:,1][t]) & (
                                var.iloc[:, 2] == values[1][:,2][t]))[0][0]
                    w2 = np.where((var.iloc[:, 0] == values[1][:,0][t]) & (var.iloc[:, 1] == values[1][:,1][t]) & (
                                var.iloc[:, 2] == values[1][:,2][t]))[0][
                        len(np.where((var.iloc[:, 0] == values[1][:,0][t]) & (var.iloc[:, 1] == values[1][:,1][t]) & (
                                var.iloc[:, 2] == values[1][:,2][t]))[0]) - 1]
                else:
                    raise (NameError('Not considered'))

                # Divide based on the limits according the values (index_val simply index for validation set)
                # Dividing the sample into val and train (and val into test and val)
                X_test.append(x.iloc[range(w, w2)])
                X_train.append(x.drop(range(w, w2)))

                Y_test.append(y.iloc[range(w, w2)])
                Y_train.append(y.drop(range(w, w2)))
                indexes.append(np.array([w, w2]))
                print('cv_division done')
        else:

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

        '''
        :param data_train: data for training
        :param pos_y: column index of target values
        :param C: C values
        :param epsilon: epsilon value
        :param tol: tolerance value (parameter SVR)
        :param save_model: True or False
        :param model: previously model trained
        :return: model and history of training
        '''
        now = str(datetime.now().microsecond)

        data_train = pd.DataFrame(data_train)
        x_train = data_train.drop(data_train.columns[pos_y], axis=1)
        y_train = data_train.iloc[:,pos_y]

        if isinstance(model, list):
            if len(pos_y)>1:
                model=MultiOutputRegressor(svm.LinearSVR(random_state=None, dual=False, loss='squared_epsilon_insensitive',C=C, tol=tol, epsilon=epsilon)
)
            else:
                model = svm.LinearSVR(random_state=None, dual=False,loss='squared_epsilon_insensitive', C=C, tol=tol, epsilon=epsilon)
        else:
            model = model

        history = model.fit(x_train, y_train)

        if save_model==True:
            name='svr'+now+'.h5'
            model.save(name, save_format='h5')

        res = {'model':model,  'history':history}
        return(res)

    def predict(self, model,train,val,mean_y, times,times_train,plotting):
        '''
        :param model: trained model
        :param x_val: x to predict
        :param y_val: y to predict
        if mean_y is empty a variation rate will be applied as cv in result. The others relative will be nan
        :return: predictions with the errors depending of zero_problem
        '''

        y_train = train.iloc[:,self.pos_y]
        x_train = train.drop(train.columns[self.pos_y], axis=1)
        y_val = val.iloc[:,self.pos_y]
        x_val = val.drop(val.columns[self.pos_y], axis=1)

        x_val=x_val.reset_index(drop=True)
        y_val=y_val.reset_index(drop=True)
        x_train=x_train.reset_index(drop=True)
        y_train=y_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_pred = model.predict(pd.DataFrame(x_val))
        y_pred_t = model.predict(pd.DataFrame(x_train))

        y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
        y_real = np.array(self.scalar_y.inverse_transform(y_val))
        y_real_t = np.array(self.scalar_y.inverse_transform(y_train))

        if len(self.pos_y)>1:
            for t in range(len(self.pos_y)):
                y_pred[np.where(y_pred[:,t] < self.inf_limit[t])[0],t] = self.inf_limit[t]
                y_pred[np.where(y_pred[:,t] > self.sup_limit[t])[0],t] = self.sup_limit[t]
                y_pred_t[np.where(y_pred_t[:,t] < self.inf_limit[t])[0],t] = self.inf_limit[t]
                y_pred_t[np.where(y_pred_t[:,t] > self.sup_limit[t])[0],t] = self.sup_limit[t]
            y_predF = pd.DataFrame(y_pred.copy())
            y_realF = pd.DataFrame(y_real).copy()

        else:
            y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
            y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit
            y_pred_t[np.where(y_pred_t < self.inf_limit)[0]] = self.inf_limit
            y_pred_t[np.where(y_pred_t > self.sup_limit)[0]] = self.sup_limit
            y_predF = pd.DataFrame(y_pred.copy())
            y_realF = pd.DataFrame(y_real).copy()

        y_predF.index = times
        y_realF.index = y_predF.index

        if self.zero_problem == 'schedule':
            print('*****Night-schedule fixed******')
            # Indexes out due to the zero_problem
            res = super().fix_values_0(times,  self.zero_problem, self.limits)
            res2 = super().fix_values_0(times_train,  self.zero_problem, self.limits)
            index_hour = res['indexes_out']
            index_hourt = res2['indexes_out']

            if len(y_pred)<=1:
                y_pred1= np.nan
                y_predt1=np.nan
                y_real1=y_real
                y_realt1=y_real_t
            else:
                if len(index_hour) > 0 and self.horizont == 0:
                    y_pred1 = np.delete(y_pred, index_hour, 0)
                    y_predt1 = np.delete(y_pred_t, index_hourt, 0)
                    y_real1 = np.delete(y_real, index_hour, 0)
                    y_realt1 = np.delete(y_real_t, index_hourt, 0)
                elif len(index_hour) > 0 and self.horizont > 0:
                    y_pred1 = np.delete(y_pred, index_hour - self.horizont, 0)
                    y_predt1 = np.delete(y_pred_t, index_hourt - self.horizont, 0)
                    y_real1 = np.delete(y_real, index_hour - self.horizont, 0)
                    y_realt1 = np.delete(y_real_t, index_hourt - self.horizont, 0)
                else:
                    y_pred1 = y_pred
                    y_predt1 = y_pred_t
                    y_real1 = y_real
                    y_realt1 = y_real_t

                # Indexes where the real values are 0
                if self.extract_cero == True and len(y_pred1) > 0:
                    if mean_y.size == 0:
                        o = np.where(y_real1 == 0)[0]
                        o2 = np.where(y_realt1 == 0)[0]
                        if len(o) > 0:
                            y_pred1 = np.delete(y_pred1, o, 0)
                            y_real1 = np.delete(y_real1, o, 0)
                        if len(o2) > 0:
                            y_predt1 = np.delete(y_predt1, o2, 0)
                            y_realt1 = np.delete(y_realt1, o2, 0)
                    else:
                        o = list()
                        o2 = list()
                        for t in range(len(mean_y)):
                            o.append(np.where(y_real1[:, t] == 0)[0])
                            o2.append(np.where(y_realt1[:, t] == 0)[0])

                        oT = np.unique(np.concatenate(o))
                        oT2 = np.unique(np.concatenate(o2))
                        y_pred1 = np.delete(y_pred1, oT, 0)
                        y_real1 = np.delete(y_real1, oT, 0)
                        y_predt1 = np.delete(y_predt1, oT2, 0)
                        y_realt1 = np.delete(y_realt1, oT2, 0)

            # Errors calculation based on mean values, weights...
            if len(y_pred1)>1:
                if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                    if mean_y.size == 0:
                        e = evals(y_pred1, y_real1).variation_rate()
                        et = evals(y_predt1, y_realt1).variation_rate()
                        if isinstance(self.weights, list):
                            cv = np.mean(e)
                            cvt = np.mean(et)
                        else:
                            cv = np.sum(e * self.weights)
                            cvt = np.sum(et * self.weights)
                        rmse = np.nan
                        nmbe = np.nan
                    else:
                        e_cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                        e_cvt = evals(y_predt1, y_realt1).cv_rmse(mean_y)
                        e_r = evals(y_pred1, y_real1).rmse()
                        e_rt = evals(y_predt1, y_realt1).rmse()
                        e_n = evals(y_pred1, y_real1).nmbe(mean_y)
                        e_nt = evals(y_predt1, y_realt1).nmbe(mean_y)
                        r2 = evals(y_pred1, y_real1).r2()

                        if isinstance(self.weights, list):
                            cv = np.mean(e_cv)
                            cvt = np.mean(e_cvt)
                            rmse = np.mean(e_r)
                            rmset = np.mean(e_rt)
                            nmbe = np.mean(e_n)
                            nmbet = np.mean(e_nt)
                        else:
                            cv = np.sum(e_cv * self.weights)
                            cvt = np.sum(e_cvt * self.weights)
                            rmse = np.sum(e_r * self.weights)
                            rmset = np.sum(e_rt * self.weights)
                            nmbe = np.sum(e_n * self.weights)
                            nmbet = np.sum(e_nt * self.weights)

                    print('ERROR TRAINING CV RMSE, RMSE,NMBE:', [cvt,rmset, nmbet])
                    print('ERROR TEST CV RMSE, RMSE,NMBE:', [cv,rmse, nmbe])
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
            # Indexes out due to the zero_problem
            place = np.where(x_val.columns == 'radiation')[0]
            scalar_x = self.scalar_x
            scalar_rad = scalar_x['radiation']
            res = super().fix_values_0(scalar_rad.inverse_transform(x_val.iloc[:, place]),
                                          self.zero_problem, self.limits)
            res2 = super().fix_values_0(scalar_rad.inverse_transform(x_train.iloc[:, place]),
                                          self.zero_problem, self.limits)
            index_rad = res['indexes_out']
            index_radt = res2['indexes_out']
            index_rad2 = np.where(y_real <= self.inf_limit)[0]
            index_rad2t = np.where(y_real_t <= self.inf_limit)[0]
            index_rad = np.union1d(np.array(index_rad), np.array(index_rad2))
            index_radt = np.union1d(np.array(index_radt), np.array(index_rad2t))

            if len(y_pred)<=1:
                y_pred1= np.nan
                y_real1=y_real
                y_realt1=y_real_t
            else:
                if len(index_rad) > 0 and self.horizont == 0:
                    y_pred1 = np.delete(y_pred, index_rad, 0)
                    y_predt1 = np.delete(y_pred_t, index_radt, 0)
                    y_real1 = np.delete(y_real, index_rad, 0)
                    y_realt1 = np.delete(y_real_t, index_radt, 0)
                elif len(index_rad) > 0 and self.horizont > 0:
                    y_pred1 = np.delete(y_pred, np.array(index_rad) - self.horizont, 0)
                    y_predt1 = np.delete(y_pred_t, np.array(index_radt) - self.horizont, 0)
                    y_real1 = np.delete(y_real, np.array(index_rad) - self.horizont, 0)
                    y_realt1 = np.delete(y_real_t, np.array(index_radt) - self.horizont, 0)
                else:
                    y_pred1 = y_pred
                    y_predt1 = y_pred_t
                    y_real1 = y_real
                    y_realt1 = y_real_t

                # Indexes where the real values are 0
                if self.extract_cero == True and len(y_pred1) > 0:
                    if mean_y.size == 0:
                        o = np.where(y_real1 == 0)[0]
                        o2 = np.where(y_realt1 == 0)[0]
                        if len(o) > 0:
                            y_pred1 = np.delete(y_pred1, o, 0)
                            y_real1 = np.delete(y_real1, o, 0)
                        if len(o2) > 0:
                            y_predt1 = np.delete(y_predt1, o2, 0)
                            y_realt1 = np.delete(y_realt1, o2, 0)
                    else:
                        o = list()
                        o2 = list()
                        for t in range(len(mean_y)):
                            o.append(np.where(y_real1[:, t] == 0)[0])
                            o2.append(np.where(y_realt1[:, t] == 0)[0])

                        oT = np.unique(np.concatenate(o))
                        oT2 = np.unique(np.concatenate(o2))
                        y_pred1 = np.delete(y_pred1, oT, 0)
                        y_real1 = np.delete(y_real1, oT, 0)
                        y_predt1 = np.delete(y_predt1, oT2, 0)
                        y_realt1 = np.delete(y_realt1, oT2, 0)

            # Errors calculation based on mean values, weights..
            if len(y_pred1)>1:
                if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                    if mean_y.size == 0:
                        e = evals(y_pred1, y_real1).variation_rate()
                        et = evals(y_predt1, y_realt1).variation_rate()
                        if isinstance(self.weights, list):
                            cv = np.mean(e)
                            cvt = np.mean(et)
                        else:
                            cv = np.sum(e * self.weights)
                            cvt = np.sum(et * self.weights)
                        rmse = np.nan
                        nmbe = np.nan
                    else:
                        e_cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                        e_cvt = evals(y_predt1, y_realt1).cv_rmse(mean_y)
                        e_r = evals(y_pred1, y_real1).rmse()
                        e_rt = evals(y_predt1, y_realt1).rmse()
                        e_n = evals(y_pred1, y_real1).nmbe(mean_y)
                        e_nt = evals(y_predt1, y_realt1).nmbe(mean_y)
                        r2 = evals(y_pred1, y_real1).r2()

                        if isinstance(self.weights, list):
                            cv = np.mean(e_cv)
                            cvt = np.mean(e_cvt)
                            rmse = np.mean(e_r)
                            rmset = np.mean(e_rt)
                            nmbe = np.mean(e_n)
                            nmbet = np.mean(e_nt)
                        else:
                            cv = np.sum(e_cv * self.weights)
                            cvt = np.sum(e_cvt * self.weights)
                            rmse = np.sum(e_r * self.weights)
                            rmset = np.sum(e_rt * self.weights)
                            nmbe = np.sum(e_n * self.weights)
                            nmbet = np.sum(e_nt * self.weights)
                    print('ERROR TRAINING CV RMSE, RMSE,NMBE:', [cvt,rmset, nmbet])
                    print('ERROR TEST CV RMSE, RMSE,NMBE:', [cv,rmse, nmbe])
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
            # Indexes where the real values are 0
            if self.extract_cero == True and len(y_pred) > 0:
                if mean_y.size == 0:
                    o = np.where(y_real == 0)[0]
                    o2 = np.where(y_real_t == 0)[0]
                    if len(o) > 0:
                        y_pred = np.delete(y_pred, o, 0)
                        y_real = np.delete(y_real, o, 0)
                    if len(o2) > 0:
                        y_pred_t = np.delete(y_pred_t, o2, 0)
                        y_real_t = np.delete(y_real_t, o2, 0)
                else:
                    o = list()
                    o2 = list()
                    for t in range(len(mean_y)):
                        o.append(np.where(y_real[:, t] == 0)[0])
                        o2.append(np.where(y_real_t[:, t] == 0)[0])

                    oT = np.unique(np.concatenate(o))
                    oT2 = np.unique(np.concatenate(o2))
                    y_pred = np.delete(y_pred, oT, 0)
                    y_pred_t = np.delete(y_pred_t, oT2, 0)
                    y_real = np.delete(y_real, oT, 0)
                    y_real_t = np.delete(y_real_t, oT2, 0)

            # Errors calculation based on mean values, weights...
            if len(y_pred)>1:
                if np.sum(np.isnan(y_pred)) == 0 and np.sum(np.isnan(y_real)) == 0:
                    if mean_y.size == 0:
                        e = evals(y_pred, y_real).variation_rate()
                        et = evals(y_pred_t, y_real_t).variation_rate()
                        if isinstance(self.weights, list):
                            cv = np.mean(e)
                            cvt = np.mean(et)
                        else:
                            cv = np.sum(e * self.weights)
                            cvt = np.sum(et * self.weights)
                        rmse = np.nan
                        nmbe = np.nan
                        r2 = np.nan
                    else:
                        e_cv = evals(y_pred, y_real).cv_rmse(mean_y)
                        e_cvt = evals(y_pred_t, y_real_t).cv_rmse(mean_y)
                        e_r = evals(y_pred, y_real).rmse()
                        e_rt = evals(y_pred_t, y_real_t).rmse()
                        e_n = evals(y_pred, y_real).nmbe(mean_y)
                        e_nt = evals(y_pred_t, y_real_t).nmbe(mean_y)
                        r2 = evals(y_pred, y_real).r2()
                        if isinstance(self.weights, list):
                            cv = np.mean(e_cv)
                            cvt = np.mean(e_cvt)
                            rmse = np.mean(e_r)
                            rmset = np.mean(e_rt)
                            nmbe = np.mean(e_n)
                            nmbet = np.mean(e_nt)
                        else:
                            cv = np.sum(e_cv * self.weights)
                            cvt = np.sum(e_cvt * self.weights)
                            rmse = np.sum(e_r * self.weights)
                            rmset = np.sum(e_rt * self.weights)
                            nmbe = np.sum(e_n * self.weights)
                            nmbet = np.sum(e_nt * self.weights)

                    print('ERROR TRAINING CV RMSE, RMSE,NMBE:', [cvt,rmset, nmbet])
                    print('ERROR TEST CV RMSE, RMSE,NMBE:', [cv,rmse, nmbe])
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

        if plotting == True and len(y_realF.shape) > 1:
            if y_realF.shape[1]>1:
                for t in range(y_realF.shape[1]):
                    a = np.round(cv, 2)
                    up = int(np.max(y_realF.iloc[:, t])) + int(np.max(y_realF.iloc[:, t]) / 4)
                    low = int(np.min(y_realF.iloc[:, t])) - int(np.min(y_realF.iloc[:, t]) / 4)
                    plt.figure()
                    plt.ylim(low, up)
                    plt.plot(y_realF.iloc[:, t], color='black', label='Real')
                    plt.plot(y_predF.iloc[:, t], color='blue', label='Prediction')
                    plt.legend()
                    plt.title("CV(RMSE)={}".format(str(a)))
                    a = 'Var-'
                    b = str(t) + '.png'
                    plot_name = a + b
                    plt.savefig(plot_name)
            else:
                a = np.round(cv, 2)
                up = int(np.max(y_realF.iloc[:, 0])) + int(np.max(y_realF.iloc[:, 0]) / 4)
                low = int(np.min(y_realF.iloc[:, 0])) - int(np.min(y_realF.iloc[:, 0]) / 4)
                plt.figure()
                plt.ylim(low, up)
                plt.plot(y_realF.iloc[:, 0], color='black', label='Real')
                plt.plot(y_predF.iloc[:, 0], color='blue', label='Prediction')
                plt.legend()
                plt.title("CV(RMSE)={}".format(str(a)))
                plt.savefig('plot1.png')
        elif plotting == True and len(y_realF.shape) < 2:
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

    def cv_analysis_svm(self, fold, values, C, epsilon, tol, mean_y, plot, q=[], model=[]):
        '''
        :param fold: divisions in cv analysis
        :param values specific values to divide the sample. specific values of a variable to search division
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        :param C: C value
        :param epsilon: epsilon value
        :param tol: tolerance value (SVR parameter)
        :param mean_y: mean of y values for error calculations
        :param dropout: percentage of dropout considered
        :param plot: True plots
        :param limite: error threshold for stop training
        :param q: a Queue to paralelyse or empty list to do not paralyse
        :param model: if model is not empty a pretrained model is considered
        :return: predictions, real values, errors and the times needed to train        '''
        from pathlib import Path
        import random
        names = self.data.drop(self.data.columns[self.pos_y], axis=1).columns
        print('##########################'
              '################################'
              'CROSS-VALIDATION'
              '#################################'
              '################################')

        #Separate y and X
        x = pd.DataFrame(self.data.drop(self.data.columns[self.pos_y], axis=1))
        y = pd.DataFrame(self.data.iloc[:, self.pos_y])
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

        #Division of the data for a CV analysis
        res = self.svm_cv_division(x, y,self.pos_y, fold,values)
        x_test = res['x_test']
        x_train = res['x_train']
        y_test = res['y_test']
        y_train = res['y_train']

        #Trying to save all the dates in each slice considered
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
                    # Create the model
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

                #Test sample
                test_x = pd.DataFrame(x_test[z]).reset_index(drop=True)
                test_y = pd.DataFrame(y_test[z]).reset_index(drop=True)
                time_start = time()
                times[z] = round(time() - time_start, 3)

                #Predicting
                y_pred = modelF.predict(test_x)
                y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
                y_real = np.array(self.scalar_y.inverse_transform(test_y))

                #Check the limits
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

                #Error calculation based on the limits (schedule, radiation)
                if self.zero_problem == 'schedule':
                    print('*****Night-schedule fixed******')
                    #Indexes out due to the zero_problem
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

                    #Indexes where the real values are 0
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

                    #Errors calculation based on mean values, weights...
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
                    # Indexes out due to the zero_problem
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

                    # Indexes where the real values are 0
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

                    # Errors calculation based on mean values, weights...
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
                    # Indexes where the real values are 0
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
                    # Errors calculation based on mean values, weights...
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

                #Plotting the results for each slice (depending of output variables)
                if plot == True and len(y_realF.shape) > 1:
                    if y_realF.shape[1] > 1:
                        for t in range(y_realF.shape[1]):
                            s = np.max(y_realF.iloc[:, t]).astype(int) + 2
                            i = np.min(y_realF.iloc[:, t]).astype(int) - 2
                            a = np.round(cv[z], 2)
                            plt.figure()
                            plt.ylim(i, s)
                            plt.plot(y_realF.iloc[:, t], color='black', label='Real')
                            plt.plot(y_predF.iloc[:, t], color='blue', label='Prediction')
                            plt.legend()
                            plt.title("Subsample {} - CV(RMSE)={}".format(z, str(a)))
                            a = 'Subsample-'
                            b = str(z) + '.png'
                            plot_name = a + b
                            plot_name = plot_name + '- Var' + str(t)
                            plt.show()
                            plt.savefig(plot_name)
                    else:
                        s = np.max(y_realF.iloc[:, 0]).astype(int) + 2
                        i = np.min(y_realF.iloc[:, 0]).astype(int) - 2
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
                    s = np.max(y_realF).astype(int) + 2
                    i = np.min(y_realF).astype(int) - 2
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

    def optimal_search(self, C_options,epsilon_options,tol_options, fold,values,mean_y, parallel, weights):
        '''
        :param C_options:possible values for C
        :param epsilon_options: possible values for epsilon
        :param tol_options:possible values for tolerance (SVR parameter)
        :param fold: division in cv analyses
        :param values specific values to divide the sample. specific values of a variable to search division
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        :param mean_y: mean of y values for error calculations
        :param parallel: True or false (True to linux)
        :param weights: weights for the two objective functions (*AL REVES)
        :return: the options with their results and the top options
        '''

        error = [0 for x in range(len(C_options)*len(epsilon_options)*len(tol_options))]
        complexity = [0 for x in range(len(C_options)*len(epsilon_options)*len(tol_options))]
        options = {'C':[], 'epsilon':[], 'tol':[]}
        w=0
        contador= len(C_options)*len(epsilon_options)*len(tol_options)-1

        #Based on the cv_analysis and if there is paralelisation colection of results
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

        #Scalating the results: errors and complexity
        scal_cv = MinMaxScaler(feature_range=(0, 1))
        scal_com = MinMaxScaler(feature_range=(0, 1))

        scal_cv.fit(np.array(r1).reshape(-1, 1))
        scal_com.fit(np.array(d1).reshape(-1, 1))

        cv = scal_cv.transform(np.array(r1).reshape(-1, 1))
        com = scal_com.transform(np.array(d1).reshape(-1, 1))

        r_final = np.array([cv[:, 0], com[:, 0]]).T

        #Trying to get the best results consdiering the two metrics
        #AASF.do(r_final, weights).argmin()
        PBI.do(r_final, weights).argmin()

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

        #Plotting the results of the two metrics and the best selected
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
                         xlimit_sup, dictionary, values,weights):
        '''
        :param model: object of ML or DL (class)
        :param med: vector of means
        :param contador: a operator to count the attempts
        :param n_processes: how many processes are parallelised
        :param l_dense:maximun number of layers dense
        :param batch: batch size
        :param pop_size: population size selected for NSGA2
        :param tol: tolearance selected to terminate the process (NSGa2 parameter)
        :param xlimit_inf: array with the lower limits to the neuron  lstm , neurons dense and pacience
        :param xlimit_sup:array with the upper limits to the neuron  lstm , neurons dense and pacience
        :param dictionary: dictionary to stored the options tested
        :param values specific values to divide the sample. specific values of a variable to search division
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        :param weights: weights for the two objective functions (*AL REVES)
        :return: options in Pareto front, the optimal selection and the total results
        '''

        #Creation of the problem
        if n_processes > 1:
            pool = multiprocessing.Pool(n_processes)
            problem = MyProblem_svm(self.data,self.horizont, self.scalar_y, self.scalar_x,self.zero_problem,self.extract_cero, self.limits, self.times, self.pos_y,self.n_lags,
                                self.mask,
                                self.mask_value, self.inf_limit, self.sup_limit,
                                self.type,
                                med, contador,len(xlimit_inf), C_max, epsilon_max, xlimit_inf, xlimit_sup,dictionary,values,self.weights,runner = pool.starmap,func_eval=starmap_parallelized_eval)
        else:
            problem = MyProblem_svm(self.data,self.horizont, self.scalar_y, self.scalar_x,self.zero_problem,self.extract_cero, self.limits, self.times, self.pos_y,self.n_lags,
                                self.mask,
                                self.mask_value, self.inf_limit, self.sup_limit,
                                self.type,
                                med, contador,len(xlimit_inf), C_max, epsilon_max, xlimit_inf, xlimit_sup,dictionary,values,self.weights)

        #Algorithm for optimisation
        algorithm = NSGA2(pop_size=pop_size, repair=MyRepair_svm(),eliminate_duplicates=True,
                          sampling=get_sampling("int_random"),
                          crossover=get_crossover("int_sbx",prob=0.95),
                          mutation=get_mutation("int_pm", prob=0.4))

        #Termination of the algorithm based on tolerance
        termination = MultiObjectiveSpaceToleranceTermination(tol=tol,
                                                              n_last=int(pop_size / 2), nth_gen=int(pop_size / 4),
                                                              n_max_gen=None,
                                                              n_max_evals=5000)
        #Result of optimisation
        res = minimize(problem,
                       algorithm,
                       termination,
                       # ("n_gen", 20),
                       pf=True,
                       verbose=True,
                       seed=7)

        #Selection of the optimum
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

            # AASF.do(r_final, weights).argmin()
            PBI.do(r_final, weights).argmin()

            obj_T = res.F
            struct_T = rx
            obj = res.F[I, :]
            struct = rx[I, :]

            #Plot of the pareto front with the optimum
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

    def optimal_search_nsga2(self, C_max, epsilon_max, pop_size, tol, xlimit_inf, xlimit_sup, mean_y, parallel,values,weights):
        '''
        :param C_max: maximum value for C_max
        :param epsilon_max: maximum value for epsilon
        :param pop_size: population size for RVEA
        :param tol: tolerance to built the pareto front (NSGA2 parameter)
        :param xlimit_inf: array with lower limits for neurons lstm, dense and pacience
        :param xlimit_sup: array with upper limits for neurons lstm, dense and pacience
        :param mean_y: vector of means
        if mean_y is empty a variation rate will be applied
        :param parallel: how many processes are parallelise
        :param values specific values to divide the sample. specific values of a variable to search division
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        :param weights: weights for the two objective functions (*AL REVES)
        :return: the options selected for the pareto front, the optimal selection and the total results
        '''

        #Multiprocessing for possible paralelisation and fill the dictionary and the contador
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

    def rnsga2_individual(self, model,med, contador, n_processes, C_max, epsilon_max, pop_size, tol, xlimit_inf,
                         xlimit_sup, dictionary,values,ref_points, weights,epsilon):
        '''
        :param model: object of ML or DL (class)
        :param med: vector of means
        :param contador: a operator to count the attempts
        :param n_processes: how many processes are parallelise
        :param C_max:maximun value for C
        :param epsilon_max: maximum value for epsilon (hyperarameter of SVR)
        :param pop_size: population size selected for NSGA2
        :param tol: tolearance selected to terminate the process (RNSGA2 parameter)
        :param xlimit_inf: array with the lower limits to the neuron  lstm , neurons dense and pacience
        :param xlimit_sup:array with the upper limits to the neuron  lstm , neurons dense and pacience
        :param dictionary: dictionary to stored the options tested
        :param values specific values to divide the sample. specific values of a variable to search division
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        :param weights: weights for the two objective functions (*AL REVES)
        :param ref_points:reference points for algorithm initialisation i.e np.array([[0.3, 0.1], [0.1, 0.3]])
        :param epsilon: parameter for RNSGA
        :return: options in Pareto front, the optimal selection and the total results
        '''

        # Creation of the problem
        if n_processes > 1:
            pool = multiprocessing.Pool(n_processes)
            problem = MyProblem_svm(model,self.data,self.horizont, self.scalar_y, self.scalar_x,self.zero_problem,self.extract_cero, self.limits, self.times, self.pos_y,self.n_lags,
                                self.mask,
                                self.mask_value, self.inf_limit, self.sup_limit,
                                self.type,
                                med, contador,len(xlimit_inf), C_max, epsilon_max, xlimit_inf, xlimit_sup,dictionary,values,self.weights,runner = pool.starmap,func_eval=starmap_parallelized_eval)
        else:
            problem = MyProblem_svm(model,self.data,self.horizont, self.scalar_y, self.scalar_x,self.zero_problem,self.extract_cero, self.limits, self.times, self.pos_y,self.n_lags,
                                self.mask,
                                self.mask_value, self.inf_limit, self.sup_limit,
                                self.type,
                                med, contador,len(xlimit_inf), C_max, epsilon_max, xlimit_inf, xlimit_sup,dictionary,values,self.weights)

        # Algorithm for optimisation
        algorithm = RNSGA2(ref_points, pop_size=pop_size, sampling=get_sampling("int_random"),
                          crossover=get_crossover("int_sbx",prob=0.95),
                          mutation=get_mutation("int_pm", prob=0.4),
                           normalization='front',
                           extreme_points_as_reference_points=False,
                           weights=weights,
                           epsilon=epsilon)

        # Termination of the algorithm based on tolerance
        termination = MultiObjectiveSpaceToleranceTermination(tol=tol,
                                                              n_last=int(pop_size / 2), nth_gen=int(pop_size / 4),
                                                              n_max_gen=None,
                                                              n_max_evals=5000)
        # Result of optimisation
        res = minimize(problem,
                       algorithm,
                       termination,
                       # ("n_gen", 20),
                       pf=True,
                       verbose=True,
                       seed=7)

        # Selection of the optimum
        if res.F.shape[0] > 1:
            rx=res.X
            scal_cv = MinMaxScaler(feature_range=(0, 1))
            scal_com = MinMaxScaler(feature_range=(0, 1))

            scal_cv.fit(res.F[:,0].reshape(-1,1))
            scal_com.fit(res.F[:,1].reshape(-1,1))

            cv=scal_cv.transform(res.F[:,0].reshape(-1,1))
            com=scal_com.transform(res.F[:,1].reshape(-1,1))

            r_final = np.array([cv[:,0], com[:,0]]).T

            # AASF.do(r_final, weights).argmin()
            PBI.do(r_final, weights).argmin()

            obj_T = res.F
            struct_T = rx
            obj = res.F[I, :]
            struct = rx[I, :]

            # Plot of the pareto front with the optimum
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

    def optimal_search_rnsga2(self, model, C_max, epsilon_max, pop_size, tol, xlimit_inf, xlimit_sup, mean_y, parallel,values,weights,ref_points=np.array([[0.3, 0.1], [0.1, 0.3]]),epsilon=0.01):
        '''
        :param model: object of ML or DL (class)
        :param C_max:maximun value for C
        :param epsilon_max: maximum value for epsilon (hyperarameter of SVR)
        :param pop_size: population size for RVEA
        :param tol: tolerance to built the pareto front (RNSGA2 parameter)
        :param xlimit_inf: array with lower limits for neurons lstm, dense and pacience
        :param xlimit_sup: array with upper limits for neurons lstm, dense and pacience
        :param mean_y: vector of means
        :param parallel: how many processes are parallelise
        :param values specific values to divide the sample. specific values of a variable to search division
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        :param weights: weights for the two objective functions (*AL REVES)
        :param ref_points:reference points for algorithm initialisation i.e np.array([[0.3, 0.1], [0.1, 0.3]])
        :param epsilon: parameter of RNSGA
        if mean_y is empty a variation rate will be applied
        :return: the options selected for the pareto front, the optimal selection and the total results        '''

        # Multiprocessing for possible paralelisation and fill the dictionary and the contador
        manager = multiprocessing.Manager()
        dictionary = manager.dict()
        contador = manager.list()
        contador.append(0)

        print('Start the optimization!!!!!')
        obj, x_obj, obj_total, x_obj_total, res,evaluations = self.rnsga2_individual(model, mean_y, contador, parallel, C_max,
                                                                            epsilon_max, pop_size, tol, xlimit_inf,
                                                                            xlimit_sup, dictionary, values, weights,ref_points,epsilon)
        np.savetxt('objectives_selectedR.txt', obj)
        np.savetxt('x_selectedR.txt', x_obj)
        np.savetxt('objectivesR.txt', obj_total)
        np.savetxt('xR.txt', x_obj_total)
        np.savetxt('evaluationsR.txt', evaluations)

        print('Process finished!!!')
        print('The selection is', x_obj, 'with a result of', obj)
        res = {'total_x': x_obj_total, 'total_obj': obj_total, 'opt_x': x_obj, 'opt_obj': obj, 'res': res,'evaluations':evaluations}
        return res
