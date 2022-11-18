from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pandas as pd
import collections
from errors import Eval_metrics as evals
from ML_v2 import ML

class MyProblem_svm(ElementwiseProblem):
    def info(self):
        print('Class to create a specific problem to use NSGA2 in architectures search.')
    def __init__(self,model,data,horizont, scalar_y,scalar_x, zero_problem,extract_cero,limits, times, pos_y, n_lags,
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
        self.model =model
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

        res = self.model.svm_cv_division(x, y, fold)
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
                res = self.model.SVR_training(pd.concat([y_t, x_t], axis=1), self.pos_y,C_svm, epsilon_svm,tol_svm,False,[])
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

            complexity = self.model.complex_svm(C_svm, epsilon_svm, 10000, 100)
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

