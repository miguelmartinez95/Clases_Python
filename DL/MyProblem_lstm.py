from pymoo.core.problem import ElementwiseProblem
import numpy as np
from time import time
import collections
import pandas as pd
from errors import Eval_metrics as evals
from DeepL_v2 import DL

class MyProblem_lstm(ElementwiseProblem):
    def info(self):
        print('Class to create a specific problem to use NSGA2 in architectures search. Two objectives and a constraint (Repair) concerning the neurons in each layer')


    def __init__(self,model,names,extract_cero, horizont,scalar_y,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,n_steps,  inf_limit,sup_limit, repeat_vector, type,data,scalar_x,dropout, weights, med, contador,
                 n_var,l_lstm, l_dense,batch,xlimit_inf, xlimit_sup,dictionary,onebyone,values,optimizer,learning_rate,activation, **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_constr=2,
                         xl=xlimit_inf,
                         xu=xlimit_sup,
                         type_var=np.int,
                         #elementwise_evaluation=True,
                         **kwargs)
        self.model=model
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
        self.optimizer=optimizer
        self.learning_rate=learning_rate
        self.activation=activation

        '''
        model: object of a class ML
        horizont: distance to the present: I want to predict the moment four steps in future
        scalar_y, scalar_x: scalars fitted
        zero_problem: schedule, radiation o else. Adjust the result to the constraints
        limits: limits based on the zero problems (hours, radiation limits, etc)
        times: dates
        pos_y: column or columns where the y is located
        mask: logic if we want to mask the missing values
        mask_value: specific value for the masking
        n_lags: times that the variables must be lagged
        n_steps: continuos time steps to predicti in the future
        inf_limit: lower accepted limits for the estimated values
        sup_limits: upper accepted limits for the estimated values
        repeat_vector:True or False (specific layer). Repeat the inputs n times (batch, 12) -- (batch, n, 12). n would be the timesteps considered as inertia
        dropout: percentage for NN
        type: regression or classification
        med: vector of means
        contador: operator to count the iterations
        l_lstm = maximum number for lstm layers
        l_dense = maximum number for dense layers
        batch: batch size in training
        xlimit_inf = lower limits for the variables optimised
        xlimit_sup = upper limits for the variable optimised
        n_var: number of inputs
        dictionary: where the different option tested are kept
        onebyone: [0] if we want to move the sample one by one [1] (True)although the horizont is 0 we want to move th sample lags by lags
        values specific values to divide the sample. specific values of a variable to search division
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        weights: weights based on the error in mutivariable case (some error must be more weighted)
        optimizer: one selectec for training
        learning_rate: one selectec for training
        activation: one selectec for training
        '''

    def cv_opt(self, data, fold, rep, neurons_lstm, neurons_dense, pacience, batch, mean_y):
        '''
        :param fold:assumed division of the sample for cv
        :param rep:repetition of the estimation in each subsample
        :param neurons_lstm: list of the different options of structures
        :param neurons_dense: list of the different options of structures
        :param pacience: limits for epochs without improvement in training
        :param batch: batchsize for training
        :param mean_y: vector of means
        :return: cv(rmse) and complexity of the model tested
        if mean_y is empty a variation rate will be applied
        '''
        name1 = tuple(np.concatenate((neurons_lstm, neurons_dense, pacience)))
        try:
            a0, a1 = self.dictionary[name1]
            return a0, a1
        except KeyError:
            pass

        if self.values:
            cvs=[0 for x in range(self.values[0])]
        else:
            cvs = [0 for x in range(rep * 2)]

        #Division of the data between inputs and outputs and the slices of the CV
        names = self.names
        names = np.delete(names, self.pos_y)
        res = self.model.cv_division_lstm(data, self.horizont, fold, self.pos_y, self.n_lags, self.n_steps,
                                          self.onebyone, self.values)
        x_test = np.array(res['x_test'])
        x_train = np.array(res['x_train'])
        x_val = np.array(res['x_val'])
        y_test = np.array(res['y_test'])
        y_train = np.array(res['y_train'])
        y_val = np.array(res['y_val'])

        times_val = res['time_val']

        if self.type == 'regression':
            # Train the model
            zz = 0
            if self.values:
                stop = self.values[0]
            else:
                stop = len(x_train)
            #Stop is the number of division in the CV
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
                model = self.model.built_model_regression(x_train[z], ytrain, neurons_lstm,
                                                          neurons_dense,
                                                          self.mask, self.mask_value, self.repeat_vector,
                                                          self.dropout,self.optimizer,self.learning_rate,self.activation)
                if self.n_steps>1:
                    batch=1
                model, history = self.model.train_model(model, x_train[z], ytrain, x_test[z], ytest, pacience,
                                                        batch)
                print('The training spent ', time() - time_start)
                if isinstance(self.pos_y, collections.abc.Sized):
                    outputs = len(self.pos_y)
                else:
                    outputs = 1

                #Conduct the predictions
                res = self.model.predict_model(model, self.n_lags, x_val[z], batch, outputs)
                y_pred = res['y_pred']
                print('Y_val SHAPE in CV_OPT', yval[z].shape)
                print('X_val SHAPE in CV_OPT', x_val[z].shape)

                #Inverse scalating and check the limits of predictions
                y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
                y_real = np.array(self.scalar_y.inverse_transform(yval))

                if isinstance(self.pos_y, collections.abc.Sized):
                    for t in range(len(self.pos_y)):
                        y_pred[np.where(y_pred[:, t] < self.inf_limit[t])[0], t] = np.repeat(self.inf_limit[t],len(np.where(y_pred[:, t] < self.inf_limit[t])[0]))
                        y_pred[np.where(y_pred[:, t] > self.sup_limit[t])[0], t] = np.repeat(self.sup_limit[t],len(np.where(y_pred[:, t] > self.sup_limit[t])[0]))
                    y_real = y_real
                else:
                    y_pred[np.where(y_pred < self.inf_limit)[0]] = np.repeat(self.inf_limit,len(np.where(y_pred < self.inf_limit)[0]))
                    y_pred[np.where(y_pred > self.sup_limit)[0]] = np.repeat(self.sup_limit,len(np.where(y_pred < self.sup_limit)[0]))
                    y_real = y_real.reshape(-1, 1)

                print('Y_pred SHAPE in CV_OPT ', y_pred.shape)
                y_predF = y_pred.copy()
                y_predF = pd.DataFrame(y_predF)
                y_predF.index = times_val[z]
                y_realF = y_real.copy()
                y_realF = pd.DataFrame(y_realF)
                y_realF.index = times_val[z]

                if self.zero_problem == 'schedule':
                    print('*****Night-schedule fixed******')
                    # Indexes out due to the zero_problem
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

                    #Errors calculation based on mean values, weights...
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
                    # Indexes out due to the zero_problem
                    place = np.where(names == 'radiation')[0]
                    scalar_rad = self.scalar_x['radiation']
                    res = DL.fix_values_0(scalar_rad.inverse_transform(x_val[z][:, self.n_lags - 1, place]),
                                          self.zero_problem, self.limits)
                    index_rad = res['indexes_out']

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
                    #NO ZERO PROBLEM
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

            #Calculation of the complexity
            complexity = self.model.complex(neurons_lstm, neurons_dense, 2000, 12)
            self.dictionary[name1] = np.mean(cvs), complexity
            res_final = {'cvs': np.mean(cvs), 'complexity': complexity}
            print(self.dictionary[name1])
            print(res_final)
            return res_final['cvs'], res_final['complexity']

    #
    @staticmethod
    def bool4(x, l_lstm, l_dense):
        '''
        :x: neurons options
        l_lstm: number of values that represent lstm neurons
        l_dense: number of values that represent dense neurons
        :return: 0 if the constraint is fulfilled or the places where the constraint is not fulfill
        It can not be a layer without neurons previous to another layer with neurons

        '''
#
        x1 = x[range(l_lstm)]
        x2 = x[range(l_lstm, l_lstm+l_dense)]

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
        '''
        :param x: option considered
        :param out: dictionary where kept the results
        :return: the results according the constrains (G) and the results fo the objective functions (F), which are gotten from cv_opt function

        '''
        g1,g2 = MyProblem_lstm.bool4(np.delete(x, len(x)-1), self.l_lstm, self.l_dense)
        out["G"] =np.column_stack([g1, g2])
#
        print('##########################################  X=',x,'##########################################')

        # Modify the vector defined for values more real
        n_lstm = x[range(self.l_lstm)]*20 #lstm neurons
        n_dense = x[range(self.l_lstm, self.l_lstm + self.l_dense)]*20 #dense neurons
        n_pacience = np.array([x[len(x)-1]])*20 #patience options

        #if not tuple(np.concatenate((n_lstm, n_dense, n_pacience))) in self.dictionary.keys():
        #    self.contador[0] += 1

        print(
            '\n ############################################## \n ############################# \n ########################## EVALUATION ',
            self.contador, '\n ######################### \n #####################################')

        f1, f2 = self.cv_opt(self.data,2,1, n_lstm, n_dense, n_pacience, self.batch, self.med)

        print('F1:',f1)
        print('F2:',f2)

        out["F"] = np.column_stack([f1, f2])
