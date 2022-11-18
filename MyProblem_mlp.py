from pymoo.core.problem import ElementwiseProblem
import numpy as np
import collections
import pandas as pd
from errors import Eval_metrics as evals
from ML_v2 import ML
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

class MyProblem_mlp(ElementwiseProblem):
    def info(self):
        print('Class to create a specific problem to use NSGA2 in architectures search.')

    def __init__(self,model, horizont, scalar_y, zero_problem, extract_cero, limits, times, pos_y, mask, mask_value, n_lags,
                 inf_limit,
                 sup_limit, type, data, scalar_x, med, contador,
                 n_var, l_dense, batch, xlimit_inf, xlimit_sup, dropout, dictionary, weights, **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_constr=1,
                         xl=xlimit_inf,
                         xu=xlimit_sup,
                         type_var=np.int,
                         # elementwise_evaluation=True,
                         **kwargs)
        self.model=model
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

    def cv_opt(self, data, fold, neurons, pacience, batch, mean_y, dictionary):
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
        neurons = neurons[neurons > 0]
        layers = len(neurons)
        y = self.data.iloc[:, self.pos_y]
        x = self.data.drop(self.data.columns[self.pos_y], axis=1)
        res = self.model.cv_division(x, y, fold)
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
            model = self.model.mlp_classification(layers, neurons, x_train[0].shape[1], n_classes, self.mask, self.mask_value)
            ####################################################################
            # EN PROCESOO ALGÚN DíA !!!!!!!
            ##########################################################################
        else:
            if self.type == 'regression':
                model = self.model.mlp_regression(layers, neurons, x_train[0].shape[1], self.mask, self.mask_value,
                                           self.dropout, len(self.pos_y))
            elif self.type == 'series':
                model = self.model.mlp_series(layers, neurons, x_train[0].shape[1], self.mask, self.mask_value,
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
                            e = evals(y_pred1, y_real1).variation_rate()
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
                            e = evals(y_pred1, y_real1).variation_rate()
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
                            e = evals(y_pred, y_real).variation_rate()
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
            print(dictionary[name1])
            res_final = {'cvs': np.mean(cvs), 'complexity': complexity}
            return res_final['cvs'], res_final['complexity']

    @staticmethod
    def bool4(x, l_dense):
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
        g1 = MyProblem_mlp.bool4(np.delete(x, len(x) - 1), self.l_dense)
        out["G"] = g1
        print(x)
        n_dense = x[range(self.l_dense)] * 20
        n_pacience = x[len(x) - 1] * 20
        f1, f2 = self.cv_opt(self.data, 3, n_dense, n_pacience, self.batch, self.med, self.dictionary)
        print(
            '\n ############################################## \n ############################# \n ########################## EvaluaciÃ³n ',
            self.contador, '\n #########################')
        self.contador[0] += 1
        out["F"] = np.column_stack([f1, f2])
