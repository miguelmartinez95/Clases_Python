import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
print('importa bien')
from errors import Eval_metrics as evals
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import RepeatVector
from keras.layers import Dropout
import time
import math
import multiprocessing
from multiprocessing import Process,Queue
import collections
from pathlib import Path
import random
from datetime import datetime
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.aasf import AASF
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from DeepL_v2 import DL
from MyRepair_lstm import MyRepair_lstm
from time import time

#Class for stop the training based on a threshold
class MyThresholdCallback(keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs["val_mse"]
        if val_loss <= self.threshold:
            self.model.stop_training = True


'''
Conexion con GPUs
'''


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)


class LSTM_model(DL):
    def info(self):
        print('Class to built LSTM models.')
        '''
        Subclass of DL 
        '''

    def __init__(self, data, horizont,scalar_y, scalar_x,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,n_steps, inf_limit,sup_limit,names,extract_cero, repeat_vector,dropout,weights, type,
                 optimizer='adam',learning_rate=0.001,activation='relu'):
        super().__init__(data, horizont,scalar_y, scalar_x,zero_problem, limits,times, pos_y, mask,mask_value,n_lags,n_steps, inf_limit,sup_limit,names,extract_cero, type)
        self.repeat_vector = repeat_vector
        self.dropout = dropout
        self.weights=weights
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        '''
        horizont: distance to the present: I want to predict the moment four steps in future
        scalar_y, scalar_x: empty lists to save the object fit to the data
        zero_problem: schedule, radiation o else. Adjust the result to the constraints
        limits: limits based on the zero problems (hours, radiation limits, etc) [0][1] limits and [2] weekend o nothing
        times: dates
        pos_y: column or columns where the y is located
        mask: logic if we want to mask the missing values
        mask_value: specific value for the masking
        n_lags: times that the variables must be lagged
        n_steps= time steps into future to predict (if >1 bathsize must be 1 and only one variable can be considered) how much estimations want to do in line
        inf_limit: lower accepted limits for the estimated values
        sup_limits: upper accepted limits for the estimated values
        names: column labels 
        extract_zero: Logic, if we want to consider or not the moment when real data is 0 (True are deleted)
        repeat_vector:True or False (specific layer). Repeat the inputs n times (batch, 12) -- (batch, n, 12). n would be the timesteps considered as inertia
        dropout: between 0 and 1. regularization technique where randomly selected neurons are ignored during training. They are “dropped out” 
        randomly. This means that their contribution to the activation of downstream neurons is temporally removed.
        type: regression or classification
        weights: weights for the outputs. mainly for multivriate output
        optimizer: used in training
        learning_rate: used in training
        activation: activiation function used in training
        '''

    @staticmethod
    def complex(neurons_lstm, neurons_dense, max_N, max_H):
        '''
        :param neurons_lstm: vector with the LSTM neurons
        :param neurons_dense: vector with the Dense neurons
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
        :return: data converted in three dimension based on the lags and the variables (slice length, amount of slices, variables)
        '''

        data_new =np.array(data_new)
        rest2 = data_new.shape[0] % n_inputs
        #We delete the surplus values for a clean division
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
        Also return the index modified if some index was left over
       '''

       index=data_new1.index
       data_new1=data_new1.reset_index(drop=True)
       data_new = data_new1.copy()

       train, test = data_new.drop(range(cut1, cut2)), data_new.iloc[range(cut1, cut2)]
       index_test = index[range(cut1, cut2)]
       index_train = np.delete(index, range(cut1, cut2))

       ###################################################################################
       #Evalaute that the datasets match with a number of timesteps sets
       rest1 = train.shape[0] % n_inputs
       ind_out1 = 0
       while rest1 != 0:
           train = train.drop(train.index[0], axis=0)
           index_train = np.delete(index_train, 0, axis=0)
           rest1 = train.shape[0] % n_inputs
           ind_out1 += 1

       rest2 = test.shape[0] % n_inputs
       ind_out = 0
       while rest2 != 0:
           test = test.drop(test.index[0], axis=0)
           index_test = np.delete(index_test,0 , axis=0)
           rest2 = test.shape[0] % n_inputs
           ind_out+=1

       ###################################################################################
       # restructure into windows of  data
       #train1 = np.array(np.split(train, len(train) / n_inputs))
       #test1 = np.array(np.split(test, len(test) / n_inputs))

       return train, test, index_train, index_test


    @staticmethod
    def to_supervised(train,pos_y, n_lags,n_steps, horizont, onebyone):
        '''
        Relate x and y based on lags and future horizont

        :param train: dataset for transforming
        :param pos_y: colums for the output data
        :param n_lags: number of lags considered
        :param n_steps= time steps into future to predict (if >1 bathsize must be 1 and only one variable can be considered) how much estimations want to do in line
        :param horizont: horizont to the future selected
        :param onebyone: how divide the data: [0] True 1 by 1, [1] True lags by lags, [1] False steps by steps        :return: x (past) and y (future horizont) considering the past-future relations selected
        '''

        #data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        data=pd.DataFrame(train)
        X, y, timesF= list(), list(),list()

        in_start = 0
        # step over the entire history one time step at a time

        #Forming the data set (X, y) according the lags and the steps considered
        if onebyone[0]==True: # 1 a 1
            for _ in range(len(data)-(n_lags + horizont+(n_steps-1))+1): #we go up to the final of dataset minus the lags, the horizont and the steps (steps-1 because wqe start at 1 default)
                # define the end of the input sequence
                in_end = in_start + n_lags
                out_end=in_end+horizont
                xx = data.drop(data.columns[pos_y], axis=1)
                yy = data.iloc[:,pos_y]
                # ensure we have enough data for this instance
                if (out_end-1+(n_steps-1))<= len(data):
                    x_input = xx.iloc[in_start:in_end,:]
                    X.append(x_input)

                    #choose the outputs with the dates
                    if n_steps==1:
                        y.append(yy.iloc[out_end-1])
                        timesF.append(data.index[out_end-1])
                    else:
                        y.append(yy.iloc[range(out_end-1,out_end-1+(n_steps))])
                        timesF.append(data.index[range(out_end-1,out_end-1+(n_steps))])

                # move along one time step
                in_start += 1

            #check if we go to the correct value (0)
            dd=len(data)-in_start-(n_lags + horizont+(n_steps-1))+1
        else:
            if onebyone[1]==True: #lag a lag
                while in_start <= data.shape[0] -(n_lags + horizont+(n_steps-1)):#we go up to the final of dataset minus the lags, the horizont and the steps (steps-1 because wqe start at 1 default)
                    # define the end of the input sequence
                    in_end = in_start + n_lags
                    out_end = in_end + horizont + (n_steps - 1)

                    xx = data.drop(data.columns[pos_y], axis=1)
                    yy = data.iloc[:, pos_y]
                    # ensure we have enough data for this instance
                    if out_end <= len(data):
                        x_input = xx.iloc[in_start:in_end, :]
                        X.append(x_input)
                        if n_steps == 1:
                            y.append(yy.iloc[out_end - 1])
                            timesF.append(data.index[out_end - 1])
                        else:
                            y.append(yy.iloc[range(out_end - 1, out_end - 1 + (n_steps))])
                            timesF.append(data.index[range(out_end - 1, out_end - 1 + (n_steps))])

                    # move along one time step
                    in_start += n_lags

                # check if we go to the correct value (0)
                dd = len(data) - (in_start-n_lags) - (n_lags + horizont + (n_steps - 1))

            else:
                while in_start <= data.shape[0] -(n_lags + horizont+(n_steps-1)+1):#we go up to the final of dataset minus the lags, the horizont and the steps (steps-1 because wqe start at 1 default)
                    # define the end of the input sequence
                    in_end = in_start + n_lags
                    out_end = in_end + horizont + (n_steps - 1)

                    xx = data.drop(data.columns[pos_y], axis=1)
                    yy = data.iloc[:, pos_y]
                    # ensure we have enough data for this instance
                    if out_end <= len(data):
                        x_input = xx.iloc[in_start:in_end, :]
                        X.append(x_input)
                        if n_steps == 1:
                            y.append(yy.iloc[out_end - 1])
                            timesF.append(data.index[out_end - 1])
                        else:
                            y.append(yy.iloc[range(out_end - 1, out_end - 1 + (n_steps - 1))])
                            timesF.append(data.index[range(out_end - 1, out_end - 1 + (n_steps))])

                    # move along one time step
                    in_start += n_steps

                # check if we go to the correct value (0)
                dd = len(data) - in_start - (n_lags + horizont + (n_steps - 1))+1

        print('Data supervised')
        print('X shape: ', np.array(X).shape)

        return(np.array(X), np.array(y),pd.to_datetime(pd.Series(timesF)), dd)

    @staticmethod
    def built_model_classification(train_x1, train_y1, neurons_lstm, neurons_dense, mask, mask_value, repeat_vector, dropout, optimizer,learning_rate,activation):
        '''

        :param trains: datasets
        :param neurons_lstm: array with the LSTM neurons that define the LSTM layers
        :param neurons_dense: array with the dense neurons that define the dense layers
        :param mask: True or False
        :param mask_value: specific value for masking
        :param repeat_vector: True or False (specific layer). Repeat the inputs n times (batch, 12) -- (batch, n, 12). n would be the timesteps considered as inertia
        :param dropout: between 0 and 1
        :param optimizer: for training
        :param learning_rate: for training
        :param activation: activation function for training
        :return: the model architecture built to be trained
        '''
        from keras import backend as K

        if any(neurons_lstm==0):
            neurons_lstm = neurons_lstm[neurons_lstm>0]
        if any(neurons_dense==0):
            neurons_dense = neurons_dense[neurons_dense>0]

        #Define the operator based on the shape of X and y
        layers_lstm = len(neurons_lstm)
        layers_neurons = len(neurons_dense)
        if len(train_x1.shape)<3 and len(train_y1.shape)<2:
            n_timesteps, n_features, n_outputs = train_x1.shape[1], 1,1
        elif len(train_x1.shape)<3:
            n_timesteps, n_features, n_outputs = train_x1.shape[1], 1, train_y1.shape[1]
        elif len(train_y1.shape)<2:
            n_timesteps, n_features, n_outputs = train_x1.shape[1],train_x1.shape[2], 1
        else:
            n_timesteps, n_features, n_outputs = train_x1.shape[1], train_x1.shape[2], train_y1.shape[1]

        model = Sequential()

        #Building the structure of LSTM layers (masking, dropout, repeat_vector)
        if layers_lstm<2:
            if mask == True:
                model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                model.add(LSTM(neurons_lstm[0], activation=activation))
            else:
                model.add(LSTM(neurons_lstm[0], input_shape=(n_timesteps, n_features),
                               activation=activation))
        else:
            for k in range(layers_lstm):
                if dropout>0 and repeat_vector==False:
                    if k == 0:
                        if mask == True:
                            model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                            model.add(LSTM(neurons_lstm[k], activation=activation,return_sequences=True,))
                            model.add(Dropout(dropout))
                        else:
                            model.add(
                                LSTM(neurons_lstm[k], input_shape=(n_timesteps, n_features), activation=activation))
                            model.add(Dropout(dropout))
                    elif k==layers_lstm-1:
                        model.add(LSTM(neurons_lstm[k],activation=activation))
                    else:
                        model.add(LSTM(neurons_lstm[k], return_sequences=True, activation=activation))
                        model.add(Dropout(dropout))

                elif dropout==0 and repeat_vector==True:
                    if k==0:
                        if mask==True:
                            model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                            model.add(LSTM(neurons_lstm[k],activation=activation))
                            model.add(RepeatVector(n_timesteps))
                        else:
                            model.add(LSTM(neurons_lstm[k], input_shape=(n_timesteps, n_features),activation=activation))
                            model.add(RepeatVector(n_timesteps))
                    elif k==layers_lstm-1:
                        model.add(LSTM(neurons_lstm[k],activation=activation))
                    else:
                        model.add(LSTM(neurons_lstm[k],return_sequences=True,activation=activation))
                elif dropout == 0 and repeat_vector == False:
                    if k == 0:
                        if mask == True:
                            model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                            model.add(LSTM(neurons_lstm[k], activation=activation,return_sequences=True,))
                        else:
                            model.add(
                                LSTM(neurons_lstm[k], input_shape=(n_timesteps, n_features), activation=activation))
                    elif k == layers_lstm - 1:
                        model.add(LSTM(neurons_lstm[k], activation=activation))
                    else:
                        model.add(LSTM(neurons_lstm[k], return_sequences=True, activation=activation))
                else:
                    raise (NameError('Dropout and Repeat vector together not considered'))

        #Building the structure of Dense layers (dropout)
        if layers_neurons>0:
            if dropout>0:
                for z in range(layers_neurons):
                    if neurons_dense[z]==0:
                        pass
                    else:
                        model.add(Dense(neurons_dense[z], activation=activation))
                        model.add(Dropout(dropout))
            else:
                for z in range(layers_neurons):
                    if neurons_dense[z]==0:
                        pass
                    else:
                        model.add(Dense(neurons_dense[z], activation=activation))

        #Last layer neurons equal to the number of outputs
        model.add(Dense(n_outputs,kernel_initializer='normal', activation='linear'))

        model.compile(loss='mse', optimizer=optimizer,metrics=['mse'])
        K.set_value(model.optimizer.learning_rate, learning_rate)
        model.summary()

        model.add(Dense(n_outputs), kernel_initializer='normal', activation='softmax')
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        K.set_value(model.optimizer.learning_rate, learning_rate)
        model.summary()

        return model


    @staticmethod
    def built_model_regression(train_x1, train_y1, neurons_lstm, neurons_dense, mask,mask_value, repeat_vector,dropout, optimizer, learning_rate, activation):
        '''

        :param trains: datasets
        :param neurons_lstm: array with the LSTM neurons that define the LSTM layers
        :param neurons_dense: array with the dense neurons that define the dense layers
        :param mask: True or False
        :param mask_value: specific value for masking
        :param repeat_vector: True or False (specific layer). Repeat the inputs n times (batch, 12) -- (batch, n, 12). n would be the timesteps considered as inertia
        :param dropout: between 0 and 1
        :param optimizer: for training
        :param learning_rate: for training
        :param activation: activation function for training
        :return: the model architecture built to be trained
        '''
        from keras import backend as K

        if any(neurons_lstm==0):
            neurons_lstm = neurons_lstm[neurons_lstm>0]
        if any(neurons_dense==0):
            neurons_dense = neurons_dense[neurons_dense>0]

        #Define the operator based on the shape of X and y
        layers_lstm = len(neurons_lstm)
        layers_neurons = len(neurons_dense)
        if len(train_x1.shape)<3 and len(train_y1.shape)<2:
            n_timesteps, n_features, n_outputs = train_x1.shape[1], 1,1
        elif len(train_x1.shape)<3:
            n_timesteps, n_features, n_outputs = train_x1.shape[1], 1, train_y1.shape[1]
        elif len(train_y1.shape)<2:
            n_timesteps, n_features, n_outputs = train_x1.shape[1],train_x1.shape[2], 1
        else:
            n_timesteps, n_features, n_outputs = train_x1.shape[1], train_x1.shape[2], train_y1.shape[1]

        model = Sequential()

        #Building the structure of LSTM layers (masking, dropout, repeat_vector)
        if layers_lstm<2:
            if mask == True:
                model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                model.add(LSTM(neurons_lstm[0], activation=activation))
            else:
                model.add(LSTM(neurons_lstm[0], input_shape=(n_timesteps, n_features),
                               activation=activation))
        else:
            for k in range(layers_lstm):
                if dropout>0 and repeat_vector==False:
                    if k == 0:
                        if mask == True:
                            model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                            model.add(LSTM(neurons_lstm[k], activation=activation,return_sequences=True,))
                            model.add(Dropout(dropout))
                        else:
                            model.add(
                                LSTM(neurons_lstm[k], input_shape=(n_timesteps, n_features), activation=activation))
                            model.add(Dropout(dropout))
                    elif k==layers_lstm-1:
                        model.add(LSTM(neurons_lstm[k],activation=activation))
                    else:
                        model.add(LSTM(neurons_lstm[k], return_sequences=True, activation=activation))
                        model.add(Dropout(dropout))

                elif dropout==0 and repeat_vector==True:
                    if k==0:
                        if mask==True:
                            model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                            model.add(LSTM(neurons_lstm[k],activation=activation))
                            model.add(RepeatVector(n_timesteps))
                        else:
                            model.add(LSTM(neurons_lstm[k], input_shape=(n_timesteps, n_features),activation=activation))
                            model.add(RepeatVector(n_timesteps))
                    elif k==layers_lstm-1:
                        model.add(LSTM(neurons_lstm[k],activation=activation))
                    else:
                        model.add(LSTM(neurons_lstm[k],return_sequences=True,activation=activation))
                elif dropout == 0 and repeat_vector == False:
                    if k == 0:
                        if mask == True:
                            model.add(Masking(mask_value=mask_value, input_shape=(n_timesteps, n_features)))
                            model.add(LSTM(neurons_lstm[k], activation=activation,return_sequences=True,))
                        else:
                            model.add(
                                LSTM(neurons_lstm[k], input_shape=(n_timesteps, n_features), activation=activation))
                    elif k == layers_lstm - 1:
                        model.add(LSTM(neurons_lstm[k], activation=activation))
                    else:
                        model.add(LSTM(neurons_lstm[k], return_sequences=True, activation=activation))
                else:
                    raise (NameError('Dropout and Repeat vector together not considered'))

        #Building the structure of Dense layers (dropout)
        if layers_neurons>0:
            if dropout>0:
                for z in range(layers_neurons):
                    if neurons_dense[z]==0:
                        pass
                    else:
                        model.add(Dense(neurons_dense[z], activation=activation))
                        model.add(Dropout(dropout))
            else:
                for z in range(layers_neurons):
                    if neurons_dense[z]==0:
                        pass
                    else:
                        model.add(Dense(neurons_dense[z], activation=activation))

        #Last layer neurons equal to the number of outputs
        model.add(Dense(n_outputs,kernel_initializer='normal', activation='linear'))

        model.compile(loss='mse', optimizer=optimizer,metrics=['mse'])
        K.set_value(model.optimizer.learning_rate, learning_rate)
        model.summary()

        return model

    @staticmethod
    def train_model(model,train_x1, train_y1, test_x1, test_y1, pacience, batch, loss_plot, metric_plot, limite):
        '''
        :param model: model architecture built
        :param train: train data set
        :param test: test data set
        :param pacience: stopping criterion
        :param bath: batchsize
        :param loss_plot: plotting the evolution of loss function
        :param metric_plot: plotting the evolution of a metric function. The seconc will be the title for the plot
        :param limite: error threshold for stop training
        :return: model trained based on pacience
        '''

        print('SHAPE TRAIN:', train_x1.shape)
        print('SHAPE TRAIN_Y:',train_y1.shape)
        print('SHAPE TEST:', test_x1.shape)
        print('SHAPE TEST_Y:', test_y1.shape)

        #Create path for save the models
        h_path = Path('./best_models')
        h_path.mkdir(exist_ok=True)
        h = h_path / f'best_{random.randint(0, 1000000)}_model.h5'

        # Checkpoitn callback
        if limite==False:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pacience)
            mc = ModelCheckpoint(str(h), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            # Train the model
            history = model.fit(train_x1, train_y1, epochs=2000, validation_data=(test_x1, test_y1), batch_size=batch,
                           callbacks=[es, mc])
        else:
            my_callback= MyThresholdCallback(threshold=limite)
            history = model.fit(train_x1, train_y1, epochs=200, validation_data=(test_x1, test_y1), batch_size=batch,
                           callbacks=[my_callback])


        #Plots of training evolution
        if loss_plot==True:
            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Val'], loc='upper left')
            plt.show()
        if not metric_plot[0]==False:
            plt.figure(figsize=(12,9))
            val_name = 'val_'+ str(metric_plot[0])
            plt.plot(history.history[str(metric_plot[0])],linewidth=2)
            plt.plot(history.history[val_name],linewidth=2)
            plt.title('')
            plt.ylabel(str(metric_plot[0]), fontsize=47,labelpad=13)
            plt.xlabel('Epoch',fontsize=38,labelpad=16)
            plt.xticks(fontsize=35)
            plt.yticks(fontsize=35)
            leg=plt.legend(['Train', 'Val'], loc='upper right', fontsize=36)
            for line in leg.get_lines():
                line.set_linewidth(5.0)
            plt.title(str(metric_plot[1]), fontsize=50)
            plt.show()
            plt.tight_layout()

        return model, history


    @staticmethod
    def predict_model(model,x_val,batch, n_outputs):
        '''
        :param model: trained model
        :param n_lags: lags to built lstm block
        :param x_val: data for predictions
        :param n_outputs: how many variables want to estimate
        :return: predictions in the validation sample, considering the selected moving window (x_val: samples, time steps, features)
        '''

        data = np.array(x_val)
        #data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        predictions = list()
        #l1 = 0
        #l2 = n_lags
        for i in range(data.shape[0]):
            # flatten data
            #input_x = data[l1:l2, :]
            input_x = data[i, :,:]
            #input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
            input_x = input_x.reshape((1, data.shape[1], data.shape[2]))
            # forecast the next step
            yhat = model.predict(input_x, verbose=0, batch_size=batch)
            if n_outputs>1:
                yhat=yhat
            else:
                yhat = yhat[0]
            predictions.append(yhat)
            #history.append(tt[i,:])
            #l1 =l2
            #l2 += n_lags

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

        :param horizont: steps in the future to estimate (two hours afterwards)
        :param fold: how many divisions (optimum 3)
        :param n_lags: number of lags
        :param estimations simutaneously to the future
        :param onebyone: how divide the data: [0] True 1 by 1, [1] True lags by lags, [1] False steps by steps
        The validation sample is extracted from the first test sample
        :param values:
        If values the division is previously defined
        It can only be divided with a initial part and a final part
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        :return: Division to cv analysis considering that with lstm algorithm the data can not be divided into simple pieces.
        return: train, test, and validation samples. Indexes for test samples (before division into test-validation)
        '''

        X_test = []
        X_train = []
        X_val = []
        Y_test = []
        Y_train = []
        Y_val = []

        #If values the division will be based on a list of operators or a variable
        if values:
            times_val = []
            place = values[2]
            var= data.iloc[:,place]
            for t in range(values[0]):
                if len(place) == 1:
                    w = np.where(var == values[1][t])[0][0]
                    w2 = np.where(var == values[1][t])[0][len(np.where(var == values[1][t])[0]) - 1]
                elif len(place) == 2:
                    w = np.where((var.iloc[:, 0] == values[1][:, 0][t]) & (var.iloc[:, 1] == values[1][:, 1][t]))[0][0]
                    w2 = np.where((var.iloc[:, 0] == values[1][:, 0][t]) & (var.iloc[:, 1] == values[1][:, 1][t]))[0][
                        len(np.where((var.iloc[:, 0] == values[1][:, 0][t]) & (var.iloc[:, 1] == values[1][:, 1][t]))[
                                0]) - 1]
                elif len(place) == 3:
                    w = np.where((var.iloc[:, 0] == values[1][:, 0][t]) & (var.iloc[:, 1] == values[1][:, 1][t]) & (
                            var.iloc[:, 2] == values[1][:, 2][t]))[0][0]
                    w2 = np.where((var.iloc[:, 0] == values[1][:, 0][t]) & (var.iloc[:, 1] == values[1][:, 1][t]) & (
                            var.iloc[:, 2] == values[1][:, 2][t]))[0][
                        len(np.where((var.iloc[:, 0] == values[1][:, 0][t]) & (var.iloc[:, 1] == values[1][:, 1][t]) & (
                                var.iloc[:, 2] == values[1][:, 2][t]))[0]) - 1]
                else:
                    raise(NameError('Not considered'))

                #Divide based on the limits according the values (index_val simply index for validation set)
                train, val, index_train, index_val = LSTM_model.split_dataset(data, n_lags, w, w2)
                val.index=index_val
                train.index = index_train

                #Once we have train and val, we define test
                #index_val = index_val[range(index_val.shape[0] - math.ceil(index_val.shape[0] / 2))]
                st = int(train.shape[0]/3)
                #test= train[range(train.shape[0]-st, train.shape[0]),:,:] #the last rows (one third) for testing
                test= train.iloc[range(train.shape[0]-st, train.shape[0]),:] #the last rows (one third) for testing
                #train=np.delete(train, list(range(train.shape[0]-st, train.shape[0])), 0) # the rest for training
                train=train.drop(train.index[range(train.shape[0]-st, train.shape[0])], axis=0) # the rest for training

                #We build the three dimension dataset with their indexes
                x_train, y_train, ind_train, dif = LSTM_model.to_supervised(train, pos_y, n_lags,n_steps, horizont, onebyone)
                x_test, y_test, ind_test, dif = LSTM_model.to_supervised(test, pos_y, n_lags,n_steps, horizont, onebyone)
                x_val, y_val, ind_val, dif = LSTM_model.to_supervised(val, pos_y, n_lags,n_steps, horizont, onebyone)

                ##Define the index
                #if onebyone[0] == True:
                #    if horizont == 0:
                #        index_val = np.delete(index_val, range(n_lags - 1), axis=0)
                #    else:
                #        index_val = np.delete(index_val, range(n_lags), axis=0)
                #else:
                #    index_val = ind_val

                #We form the list with the appropiate division based on lags and according the different values
                times_val.append(ind_val)
                X_test.append(x_test)
                X_train.append(x_train)
                X_val.append(x_val)
                Y_test.append(y_test)
                Y_train.append(y_train)
                Y_val.append(y_val)
                print('cv_division done')

        else:
            #If not values the cv division is classic but only taking into account the beginning and the end of the sample (simplification)
            ###################################################################################
            step = int(data.shape[0] / fold)
            w = 0
            w2 = step
            times_val = []
            ####################################################################################

            try:
               for i in range(2): #division in two: test with the beginning and test with the final
                    train, test, index_train,index_val = LSTM_model.split_dataset(data, n_lags,w, w2)
                    test.index = index_val
                    train.index = index_train
                    #r = LSTM_model.three_dimension(index_val, n_lags)
                    #index_val=r['data']

                    #index_val = index_val[range(test.shape[0]-math.ceil(test.shape[0]/2), test.shape[0]),:]
                    #val = test[range(test.shape[0]-math.ceil(test.shape[0]/2), test.shape[0]),:,:]
                    val = train.iloc[range(train.shape[0]-math.ceil(train.shape[0]/3), train.shape[0]),:]
                    #test = test[range(0, math.ceil(test.shape[0] / 2)), :, :]
                    train = train.iloc[range(0, math.ceil(train.shape[0] / 3)), :]

                    #index_val= index_val.reshape(index_val.shape[0]*index_val.shape[1],1)

                    x_train, y_train,ind_train,dif = LSTM_model.to_supervised(train, pos_y, n_lags,n_steps,horizont, onebyone)
                    x_test, y_test,ind_test,dif = LSTM_model.to_supervised(test, pos_y, n_lags,n_steps,horizont,onebyone)
                    x_val, y_val,ind_val,dif = LSTM_model.to_supervised(val, pos_y, n_lags,n_steps,horizont, onebyone)


                    #if onebyone[0]==True:
                    #    index_val = np.delete(index_val, range(n_lags+horizont), axis=0)
                    #else:
                    #    if isinstance(ind_val, list):
                    #        index_val = index_val[np.concatenate(ind_val)]
                    #    else:
                    #        index_val=index_val[ind_val]

                    # We form the list with the appropiate division based on lags and according to the fold division
                    times_val.append(ind_val)
                    X_test.append(x_test)
                    X_train.append(x_train)
                    X_val.append(x_val)
                    Y_test.append(y_test)
                    Y_train.append(y_train)
                    Y_val.append(y_val)

                    #Update the cuts
                    w = data.shape[0]-w2
                    w2 = data.shape[0]
            except:
                raise NameError('Problems with the sample division in the cv classic')
        res = {'x_test': X_test, 'x_train': X_train, 'x_val':X_val, 'y_test': Y_test, 'y_train': Y_train, 'y_val':Y_val,  'time_val':times_val}
        return res

    def cv_analysis(self, fold,rep, neurons_lstm, neurons_dense,onebyone, pacience, batch,mean_y,values,plot,loss_plot=False,metric_plot=[False,False],limite=False, q=[], model=[]):
        '''
        :param fold: the assumed size of divisions
        :param rep: In this case, the analysis repetitions of each of the two possile divisions (if wanted)
        :param neurons_lstm: vector of LSTM neurons
        :param neurons_dense: vector of Dense neurons
        :param onebyone: how divide the data: [0] True 1 by 1, [1] True lags by lags, [1] False steps by steps
        :param pacience: stopping criterion
        :param batch: batch size for training
        :param mean_y: vector of means
        :param values specific values to divide the sample. specific values of a variable to search division
        :param plot: True - plots
        :param loss_plot: plotting the evolution of loss function
        :param metric_plot: plotting the evolution of a metric function. The seconc will be the title for the plot
        :param limite: error threshold for stop training
        :param q: queue that inform us if paralyse or not
        :param model if model we have a pretrained model

        if mean_y is size 0 the evaluation will be with variation rates

        :return: Considering what zero_problem is mentioned, return thre predictions, real values, errors and computational times needed to train the models
        '''

        names = self.names
        names = np.delete(names ,self.pos_y)
        layers_lstm = len(neurons_lstm)
        layers_neurons = len(neurons_dense)

        #Division of the sample
        res = LSTM_model.cv_division_lstm(self.data, self.horizont, fold, self.pos_y, self.n_lags,self.n_steps, onebyone,values)

        #Get the lists of divisions in three dimensions
        x_test =np.array(res['x_test'])
        x_train=np.array(res['x_train'])
        x_val=np.array(res['x_val'])
        y_test=np.array(res['y_test'])
        y_train =np.array(res['y_train'])
        y_val =np.array(res['y_val'])
        times_val = res['time_val']


        if self.type=='regression':
            #Checking if we hace a pretrained model
            if isinstance(model, list):
                if len(y_train[0].shape)>1:
                    y_trainO=y_train[0]
                else:
                    y_trainO=y_train[0].reshape(-1, 1)

                model1 = self.__class__.built_model_regression(x_train[0],y_trainO,neurons_lstm, neurons_dense, self.mask,self.mask_value, self.repeat_vector, self.dropout, self.optimizer, self.learning_rate, self.activation)

            else:
                model1=model

            #Define the times we will do the analysis
            if values:
                stop = values[0]
            else:
                stop = len(x_train)

            times = [0 for x in range(rep*stop)]
            cv = [0 for x in range(rep*stop)]
            rmse = [0 for x in range(rep*stop)]
            nmbe = [0 for x in range(rep*stop)]
            zz= 0
            predictions = []
            reales = []
            for z in range(stop):
            # Train the model
                print('Fold number', z)
                for zz2 in range(rep):
                    if len(y_train[z].shape) > 1:
                        ytrain = y_train[z]
                        ytest = y_test[z]
                        yval = y_val[z]
                    else:
                        ytrain = y_train[z].reshape(len(y_train[z]), 1)
                        ytest = y_test[z].reshape(len(y_test[z]), 1)
                        yval = y_val[z].reshape(len(y_val[z]), 1)

                    modelF = model1
                    time_start = time()

                    if self.n_steps>1: #condition for n_steps>1
                        batch=1
                    modelF, history = self.__class__.train_model(modelF,x_train[z], ytrain, x_test[z], ytest, pacience, batch,loss_plot, metric_plot, limite)
                    times[zz] = round(time() - time_start, 3)

                    if isinstance(self.pos_y, collections.abc.Sized):
                        outputs = len(self.pos_y)
                    else:
                        outputs = 1

                    res = self.__class__.predict_model(modelF, x_val[z], batch, outputs)

                    #Get the prediction and apply inversely the scalating
                    y_pred = res['y_pred']
                    y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
                    y_real = np.array(self.scalar_y.inverse_transform(yval))

                    #Check the defined limits
                    if isinstance(self.pos_y, collections.abc.Sized):
                        for t in range(len(self.pos_y)):
                            y_pred[np.where(y_pred[:, t] < self.inf_limit[t])[0], t] = np.repeat(self.inf_limit[t], len(
                                np.where(y_pred[:, t] < self.inf_limit[t])[0]))
                            y_pred[np.where(y_pred[:, t] > self.sup_limit[t])[0], t] = np.repeat(self.sup_limit[t], len(
                                np.where(y_pred[:, t] > self.sup_limit[t])[0]))
                        y_real = y_real
                    else:
                        y_pred[np.where(y_pred < self.inf_limit)[0]] = np.repeat(self.inf_limit, len(
                            np.where(y_pred < self.inf_limit)[0]))
                        y_pred[np.where(y_pred > self.sup_limit)[0]] = np.repeat(self.sup_limit, len(
                            np.where(y_pred < self.sup_limit)[0]))
                        y_real = y_real.reshape(-1, 1)

                    if plot == True:

                        if self.n_steps>1:
                            if len(self.pos_y)<2 and onebyone[0]==False and onebyone[1]==False: # only one variable and we move based on n_steps
                                #y_predP = y_pred.reshape(int(y_pred.shape[0] / self.horizont), self.horizont)
                                #y_realP = y_real.reshape(int(y_real.shape[0] / self.horizont), self.horizont)
                                y_predP = pd.DataFrame(np.concatenate(y_pred))
                                y_realP = pd.DataFrame(np.concatenate(y_real))

                                #y_predP = pd.DataFrame(y_predP[:,0])
                                y_predP.index = times_val[z]
                                #y_realP = pd.DataFrame(y_realP[:,0])
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
                                print('Plot is not defined for this situation')
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
                        # Indexes out due to the zero_problem (dates of estimations)
                        res = super().fix_values_0(times_val[z],
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

                        '''After  checking we have data to evaluate:
                        # if the mean_y is empty we use variation rate with or witout weights
                        # on the other hand, we compute the classic error metrics'''
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
                        # Indexes out due to the zero_problem (radiation values in the last values of inputs for each slice)
                        place = np.where(names == 'radiation')[0]
                        scalar_x = self.scalar_x
                        scalar_rad = scalar_x['radiation']
                        res = super().fix_values_0(scalar_rad.inverse_transform(x_val[zz][:, x_val[zz].shape[1] - 1, place]),
                                                   self.zero_problem, self.limits)
                        index_rad = res['indexes_out']

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

                        '''After  checking we have data to evaluate:
                            if the mean_y is empty we use variation rate with or witout weights
                            on the other hand, we compute the classic error metrics'''

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

                        '''After  checking we have data to evaluate:
                            if the mean_y is empty we use variation rate with or witout weights
                            on the other hand, we compute the classic error metrics'''

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

            #Conexions if parallelisation
            z = Queue()
            if type(q) == type(z):
                #q.put(np.array([np.mean(cv), np.std(cv)]))
                q.put(np.array([np.mean(cv), self.complex(layers_lstm,layers_neurons,2000,12)]))
            else:
                return (res_final)

    ###################################################################################################
        #FALTARÍA CLASIFICACION !!!!!!!!!!!!!!!!!
        ###################################################################################################

    def train(self, train, test, dat_supervised,neurons_lstm, neurons_dense, pacience, batch, save_model,onebyone,model=[],loss_plot=False,metric_plot=[False,False],limite=False, testing=False):
        '''
        :param train: data for train
        :param test: data for validate
        :param dat_suerpvised: True if the train/test already enter supervised (3 dimensions)
        :param neurons_lstm: vector of neurons LSTM
        :param neurons_dense: vector of neurons Dens
        :param pacience: stopping criterion
        :param batch: bathc size for training
        :param save_model: True if we want to save the model
        :param onebyone: how divide the data: [0] True 1 by 1, [1] True lags by lags, [1] False steps by steps        :param model: if we have a pretrained model
        :param loss_plot: plotting the evolution of loss function
        :param metric_plot: plotting the evolution of a metric function. The seconc will be the title for the plot
        :param limite: error threshold for stop training
        :param testing: if we want to return the test data
        Instance to train model outside these classes
        :return: the trained model, history and the time required to be trained
        '''

        now = str(datetime.now().microsecond)

        #res = self.__class__.three_dimension(train, self.n_lags)
        #train = res['data']
        #res = self.__class__.three_dimension(test, self.n_lags)
        #test = res['data']
#
        #print('Data in three dimensions')

        #Define the structure of inputs and outputs
        if dat_supervised==True:
            y_test, x_test = test[0], test[1]
            y_train, x_train = train[0], train[1]
            ind_test = test[2]
            ind_train = train[2]
        else:
            x_test, y_test, ind_test, dif = LSTM_model.to_supervised(test, self.pos_y, self.n_lags,self.n_steps, self.horizont,
                                                                         onebyone)
            x_train, y_train, ind_train, dif = LSTM_model.to_supervised(train, self.pos_y, self.n_lags,self.n_steps, self.horizont,
                                                                            onebyone)

        if self.n_steps > 1:
            batch = 1
        elif self.n_steps==1 and not isinstance(self.pos_y, collections.abc.Sized):
            y_train = y_train.reshape(-1,1)
            y_test = y_test.reshape(-1,1)


        #We define a model and train it (or retrain a pretrained model)
        if isinstance(model, list):
            if self.type=='regression':
                model = self.__class__.built_model_regression(x_train, y_train,neurons_lstm, neurons_dense, self.mask, self.mask_value, self.repeat_vector, self.dropout,self.optimizer, self.learning_rate, self.activation)
                time_start = time()

                model_trained, history = self.__class__.train_model(model, x_train, y_train, x_test, y_test, pacience, batch, loss_plot, metric_plot,limite)
                times = round(time() - time_start, 3)
            else:
                model = self.__class__.built_model_classification(x_train, y_train,neurons_lstm, neurons_dense,self.mask, self.mask_value, self.repeat_vector, self.dropout, self.optimizer, self.learning_rate, self.activation)
                time_start = time()
                model_trained, history = self.__class__.train_model(model, x_train, y_train, x_test, y_test, pacience, batch, loss_plot,metric_plot,limite)
                times = round(time() - time_start, 3)
        else:
            time_start = time()
            model_trained, history = self.__class__.train_model(model, x_train, y_train, x_test, y_test, pacience,batch, loss_plot,metric_plot,limite)
            times = round(time() - time_start, 3)

        if save_model==True:
            name='mlp'+now+'.h5'
            model_trained.save(name, save_format='h5')
        if testing==True:
            res={'train':train, 'test':test, 'X_test':x_test, 'X_train':x_train, 'Y_test':y_test, 'Y_train':y_train, 'ind_test':ind_test, 'ind_train':ind_train}
        else:
            res = {'model': model_trained, 'times': times, 'history':history}
        return res


    def predict(self, model, val,mean_y,batch, onebyone, scalated,daily,plotting):
        '''
        :param model: trained model
        :param val: data for testing the model
        :param mean_y: vector of means
        :param batch: batch size for training
        :param times: dates for plot
        :param onebyone: how divide the data: [0] True 1 by 1, [1] True lags by lags, [1] False steps by steps
        :param scalated: if they are scalated: [0] prediction sample [1] real sample
        :param daily: option to generate results day by day
        :param plotting: if True we create plots
         :return: prediction with the built metrics
        Instance to predict certain samples outside these classes
        '''

        #We define the times for the prediction sample  based on horizont and n_steps
        #if self.horizont == 0:
        #    if onebyone[0]==True:
        #        times = np.delete(times, 0, 0)
        #    else:
        #        if onebyone[1] == True:
        #            seq=list()
        #            cont = self.n_lags
        #            while cont <= len(times):
        #                if self.n_steps == 1:
        #                    seq.append(times[cont])
        #                else:
        #                    seq.append(times[range(cont, cont + (self.n_steps))])
        #                cont += self.n_lags
        #            times = np.concatenate(seq)
        #        elif onebyone[1] == False:
        #            seq = list()
        #            cont = self.n_lags
        #            while cont <= len(times):
        #                if self.n_steps == 1:
        #                    seq.append(times[cont])
        #                else:
        #                    seq.append(times[range(cont, cont + (self.n_steps))])
        #                cont += self.n_steps
        #            times = np.concatenate(seq)
#
        #elif self.horizont > 0:
        #    if onebyone[0]==True:
        #        times = np.delete(times, self.horizont, 0)
        #    else:
        #        if onebyone[1] == True:
        #            seq=list()
        #            cont = self.n_lags +self.horizont
        #            while cont <= len(times):
        #                if self.n_steps == 1:
        #                    seq.append(times[cont])
        #                else:
        #                    seq.append(times[range(cont, cont + (self.n_steps))])
        #                cont += self.n_lags
        #            times = np.concatenate(seq)
        #        elif onebyone[1] == False:
        #            seq = list()
        #            cont = self.n_lags +self.horizont
        #            while cont <= len(times):
        #                if self.n_steps == 1:
        #                    seq.append(times[cont])
        #                else:
        #                    seq.append(times[range(cont, cont + (self.n_steps))])
        #                cont += self.n_steps
        #            times = np.concatenate(seq)
        #else:
        #    raise NameError('Negative horizont')

        #Define the structure of inputs and outputs
        x_val, y_val,ind_val,dif = self.__class__.to_supervised(val, self.pos_y, self.n_lags,self.n_steps, self.horizont, onebyone)

        print('Diferencia entre time and y:',dif)
        print('X_val SHAPE in testing',x_val.shape)
        print('Y_val SHAPE in testing',y_val.shape)

        if isinstance(self.pos_y, collections.abc.Sized):
            outputs = len(self.pos_y)
        else:
            outputs = 1

        res = self.__class__.predict_model(model, x_val,batch, outputs)

        y_pred = res['y_pred']
        print('Y_pred SHAPE in testing',y_pred.shape)


        times=pd.to_datetime(pd.Series(ind_val))
        if len(times)!=len(y_pred):
            raise NameError('Dates length is not equal than predictions')

        #Apply the inverse scalating (if neccesary)
        if scalated[0]==True:
            y_pred = np.array(self.scalar_y.inverse_transform(pd.DataFrame(y_pred)))
        if scalated[1]==True:
            if len(y_val.shape)>1:
                y_val = np.array(self.scalar_y.inverse_transform(y_val))
            else:
                y_val=np.array(self.scalar_y.inverse_transform(y_val.reshape(-1,1)))

        #Adjust the limits in the predictions
        if isinstance(self.pos_y, collections.abc.Sized):
            for t in range(len(self.pos_y)):
                y_pred[np.where(y_pred[:,t] < self.inf_limit[t])[0],t] = self.inf_limit[t]
                y_pred[np.where(y_pred[:,t] > self.sup_limit[t])[0], t] = self.sup_limit[t]
            y_real=y_val
        else:
            y_pred[np.where(y_pred < self.inf_limit)[0]] = self.inf_limit
            y_pred[np.where(y_pred > self.sup_limit)[0]] = self.sup_limit
            y_real = y_val.reshape((len(y_val), 1))

        y_predF = y_pred.copy()
        y_predF = pd.DataFrame(y_predF)
        y_predF.index = times
        y_realF = pd.DataFrame(y_real.copy())
        y_realF.index = y_predF.index

        if self.zero_problem == 'schedule':
            print('*****Night-schedule fixed******')
            # Indexes out due to the zero_problem (dates of estimations)
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

            '''After  checking we have data to evaluate:
            # if the mean_y is empty we use variation rate with or witout weights
            # on the other hand, we compute the classic error metrics'''
            if len(y_pred1)>0:
                if np.sum(np.isnan(y_pred1)) == 0 and np.sum(np.isnan(y_real1)) == 0:
                    if daily==True:
                        cv, std_cv = evals(y_pred1, y_real1).cv_rmse_daily(mean_y, times)
                        nmbe, std_nmbe = evals(y_pred1, y_real1).nmbe_daily(mean_y, times)
                        rmse, std_rmse = evals(y_pred1, y_real1).rmse_daily(times)
                        r2 = evals(y_pred1, y_real1).r2()
                        cv = np.mean(cv)
                        nmbe=np.mean(nmbe)
                        res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv,'std_cv': std_cv, 'nmbe': nmbe,'mae':np.nan, 'std_mae':np.nan,'std_nmbe': std_nmbe, 'rmse': rmse,'std_rmse': std_rmse, 'r2': r2}
                    else:
                        if mean_y.size == 0:
                            e = evals(y_pred1, y_real1).variation_rate()
                            if isinstance(self.weights, list):
                                cv= np.mean(e)
                            else:
                                cv= np.sum(e * self.weights)
                            rmse= np.nan
                            nmbe= np.nan
                        else:
                            e_cv = evals(y_pred1, y_real1).cv_rmse(mean_y)
                            e_r = evals(y_pred1, y_real1).rmse()
                            e_n = evals(y_pred1, y_real1).nmbe(mean_y)
                            e_mae = evals(y_pred1, y_real1).mae()
                            r2 = evals(y_pred1, y_real1).r2()
                            if isinstance(self.weights, list):
                                cv = np.mean(e_cv)
                                rmse = np.mean(e_r)
                                nmbe = np.mean(e_n)
                                mae = np.mean(e_mae)
                            else:
                                cv= np.sum(e_cv * self.weights)
                                rmse = np.sum(e_r * self.weights)
                                nmbe= np.sum(e_n * self.weights)
                                mae= np.sum(e_mae * self.weights)

                        res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv, 'mae':mae, 'nmbe': nmbe,
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

        elif self.zero_problem == 'radiation':
            print('*****Night-radiation fixed******')
            # Indexes out due to the zero_problem (radiation values in the last values of inputs for each slice)

            place = np.where(self.names == 'radiation')[0]
            scalar_x = self.scalar_x
            scalar_rad = scalar_x['radiation']
            res = super().fix_values_0(scalar_rad.inverse_transform(x_val[:,x_val.shape[1]-1, place]),
                                       self.zero_problem, self.limits)
            index_rad = res['indexes_out']

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

            '''After  checking we have data to evaluate:
                if the mean_y is empty we use variation rate with or witout weights
                on the other hand, we compute the classic error metrics'''
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
                            e_mae = evals(y_pred1, y_real1).mae()
                            r2 = evals(y_pred1, y_real1).r2()
                            if isinstance(self.weights, list):
                                cv = np.mean(e_cv)
                                rmse = np.mean(e_r)
                                nmbe = np.mean(e_n)
                                mae=np.mean(e_mae)
                            else:
                                cv = np.sum(e_cv * self.weights)
                                rmse = np.sum(e_r * self.weights)
                                nmbe = np.sum(e_n * self.weights)
                                mae = np.sum(e_mae * self.weights)
                        res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv,'mae':mae, 'nmbe': nmbe,
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

            '''After  checking we have data to evaluate:
                if the mean_y is empty we use variation rate with or witout weights
                on the other hand, we compute the classic error metrics'''

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
                               'rmse': rmse, 'std_rmse': std_rmse, 'r2': r2}
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
                            e_mae = evals(y_pred, y_real).mae()
                            r2 = evals(y_pred, y_real).r2()
                            if isinstance(self.weights, list):
                                cv = np.mean(e_cv)
                                rmse = np.mean(e_r)
                                nmbe = np.mean(e_n)
                                mae = np.mean(e_mae)
                            else:
                                cv = np.sum(e_cv * self.weights)
                                rmse = np.sum(e_r * self.weights)
                                nmbe = np.sum(e_n * self.weights)
                                mae = np.sum(e_mae * self.weights)
                        res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv,'mae':mae, 'nmbe': nmbe,
                               'rmse': rmse, 'r2': r2}
                else:
                    print('Missing values are detected when we are evaluating the predictions')
                    cv = 9999
                    nmbe = 9999
                    rmse = 9999
                    r2 = -9999
                    mae=99999
                    res = {'y_pred': y_predF,'y_real':y_realF, 'cv_rmse': cv,'mae':mae, 'nmbe': nmbe,
                           'rmse': rmse, 'r2': r2}
            else:
                raise NameError('Empty prediction')

        y_realF = pd.DataFrame(y_realF)
        y_realF.index = y_predF.index

        if plotting==True:
            import matplotlib.dates as md
            a = np.round(cv, 2)
            up =int(np.max(y_realF)) + int(np.max(y_realF)/4)
            low = int(np.min(y_realF)) - abs(int(np.min(y_realF)/4))
            fig,ax = plt.subplots(1,1)
            plt.ylim(low, up)
            plt.plot(y_realF, color='black', label='Real')
            plt.plot(y_predF, color='blue', label='Prediction')
            plt.legend()
            plt.title("CV(RMSE)={}".format(str(a)))
            ax.xaxis.set_major_locator(md.HourLocator(byhour=[0, 12]))
            ax.xaxis.set_major_formatter(md.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=70)
            plt.show()
            plt.savefig('plot1.png')

        return res

    def optimal_search(self, fold, rep, neurons_dense, neurons_lstm, paciences, onebyone,batch, mean_y,parallel,weights,values):
        '''
        Parallelisation is not work tested!!!

        :param fold: assumed division of data sample
        :param rep: repetitions of cv analysis considering the intial or the final of sample
        :param neurons_dense: list of options for neurons dense
        :param neurons_lstm: list of options for neurons LSTM
        :param paciences: list of options for paciences
        :param onebyone: how divide the data: [0] True 1 by 1, [1] True lags by lags, [1] False steps by steps        :param batch: batch size for training
        :param mean_y: vector of means
        :param parallel: 0 no paralyse
        :param weights: weights between the two objective function (*AL REVES)
        :param values specific values to divide the sample. specific values of a variable to search division
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        :return: errors obtained with the options considered together  with the best solution
        '''

        error = [0 for x in range(len(neurons_lstm) * len(neurons_dense) * len(paciences))]
        complexity = [0 for x in range(len(neurons_lstm) * len(neurons_dense) * len(paciences))]

        options = {'neurons_dense': [], 'neurons_lstm': [], 'pacience': []}
        w = 0
        contador=len(neurons_lstm) * len(neurons_dense) * len(paciences)-1
        ct = time()
        if parallel<2:
            # CV analysis for each of the options
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
                # CV analysis for each of the options
                neuron_dense = neurons_dense[t]
                for j in range(len(neurons_lstm)):
                    neuron_lstm = neurons_lstm[j]

                    for i in range(len(paciences)):
                        print('##################### Option ####################', w)
                        options['neurons_dense'].append(neuron_dense)
                        options['neurons_lstm'].append(neuron_lstm)
                        options['pacience'].append(paciences[i])
                        if z < parallel and w<contador:
                            #We acumulate processes
                            p = Process(target=self.cv_analysis,
                                        args=(fold,rep, neuron_lstm, neuron_dense, paciences[i], batch, mean_y,values,False, q))
                            p.start()

                            processes.append(p)
                            z1 =z+ 1
                        if z == parallel and w < contador:
                            #Reset processes and send the queues
                            p.close()
                            for p in processes:
                                p.join()

                            for v in range(len(processes)):
                                res2.append(q.get()[0])
                                dev2.append(q.get()[1])

                            processes=[]
                            q = Queue()
                            p = Process(target=self.cv_analysis,
                                        args=(fold, rep, neuron_lstm, neuron_dense, paciences[i], batch, mean_y,values,False, q))
                            p.start()

                            processes.append(p)
                            z1 = 1

                        elif w==contador:
                            #We reached the last possible process
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

        ct_total = time()-ct
        r1 = error.copy()
        d1 = complexity.copy()

        #Normalize the results
        scal_cv = MinMaxScaler(feature_range=(0, 1))
        scal_com = MinMaxScaler(feature_range=(0, 1))
        scal_cv.fit(np.array(r1).reshape(-1, 1))
        scal_com.fit(np.array(d1).reshape(-1, 1))
        cv = scal_cv.transform(np.array(r1).reshape(-1, 1))
        com = scal_com.transform(np.array(d1).reshape(-1, 1))
        r_final = np.array([cv[:, 0], com[:, 0]]).T

        #We search for the best solution
        #AASF.do(r_final, weights).argmin()
        I=PBI.do(r_final, weights).argmin()

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
        print('time', ct_total)

        np.savetxt('objectives_selected_brute.txt', np.array([top_result['error'],top_result['complexity']]))
        np.savetxt('x_selected_brute.txt', np.concatenate((top_result['neurons_lstm'],top_result['neurons_dense'],np.array([top_result['pacience']]))))


        #Plot the results on the two objective functions and mark the optimum
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

    def nsga2_individual(self,model,med, contador,n_processes,l_lstm, l_dense, batch,pop_size,tol,n_last, nth_gen, xlimit_inf, xlimit_sup,dictionary,onebyone,values,weights):
        from MyProblem_lstm import MyProblem_lstm
        '''
        :param model: object of ML or DL (class)
        :param med:vector of means
        :param contador: a operator to count the attempts
        :param n_processes: how many processes are parallelise
        :param l_lstm:maximun number of layers lstm
        :param l_dense:maximun number of layers dense
        :param batch: batch size
        :param pop_size: population size selected for NSGA2
        :param tol: tolearance selected to terminate the process
        :param n_last: last generation considered in the search
        :param nth_gen: number of generation to evaluate
        :param xlimit_inf: array with the lower limits to the neuron  lstm , neurons dense and pacience
        :param xlimit_sup:array with the upper limits to the neuron  lstm , neurons dense and pacience
        :param dictionary: dictionary to stored the options tested
        :param onebyone: how divide the data: [0] True 1 by 1, [1] True lags by lags, [1] False steps by steps        :param values specific values to divide the sample. specific values of a variable to search division
        values: list with: 0-how many divisions, 1-values to divide, 2-place of the variable or variables to divide
        :param weigths: weights for the objective functions (*AL REVES)
        :return: options in Pareto front, the optimal selection and the total results. Consider the option of parallelisation with runners
        '''

        #Define the problem class considering if parallelisation is wanted
        if n_processes>1:
            pool = multiprocessing.Pool(n_processes)
            problem = MyProblem_lstm(model,self.names,self.extract_cero,self.horizont, self.scalar_y, self.zero_problem, self.limits,self.times,self.pos_y,self.mask,
                                self.mask_value, self.n_lags,self.n_steps,self.inf_limit, self.sup_limit, self.repeat_vector, self.type, self.data,
                                self.scalar_x,self.dropout,self.weights,med, contador,len(xlimit_inf),l_lstm, l_dense, batch, xlimit_inf, xlimit_sup,dictionary,onebyone,values,self.optimizer,
            self.learning_rate,self.activation,runner = pool.starmap,func_eval=StarmapParallelization)
        else:
            problem = MyProblem_lstm(model,self.names,self.extract_cero,self.horizont, self.scalar_y, self.zero_problem, self.limits,self.times,self.pos_y,self.mask,
                                self.mask_value, self.n_lags,self.n_steps, self.inf_limit, self.sup_limit, self.repeat_vector, self.type, self.data,
                                self.scalar_x, self.dropout,self.weights,med, contador,len(xlimit_inf),l_lstm, l_dense, batch, xlimit_inf, xlimit_sup,dictionary,onebyone,values,
                                     self.optimizer,self.learning_rate, self.activation)

        #Define the algorithm
        algorithm = NSGA2(pop_size=pop_size, repair=MyRepair_lstm(l_lstm, l_dense), eliminate_duplicates=True,
                          sampling=get_sampling("int_random"),
                          crossover=get_crossover("int_sbx"),
                          mutation=get_mutation("int_pm", prob=0.1))

        #Define the termination based on tolerance in feasible space
        termination = MultiObjectiveSpaceTermination(tol=tol,
                                                              n_last=n_last, nth_gen=nth_gen, n_max_gen=None,
                                                              n_max_evals=6000)
        '''
        Termination can be with tolerance or with generations limit
        '''
        ct = time()
        res = minimize(problem,
                       algorithm,
                       termination,
                       # ("n_gen", 20),
                       pf=True,
                       verbose=True,
                       save_history=True,
                       seed=7)
        ct_total = time()-ct

        #We select the final optimum with normalized results
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
            I=PBI.do(r_final, weights).argmin()

            obj_T = res.F
            struct_T = rx
            obj = res.F[I, :]
            struct = rx[I, :]

            plt.figure(figsize=(10, 7))
            plt.scatter(r_final[:, 0], r_final[:, 1], color='black',s=75)
            plt.xlabel('Normalised CV (RMSE)', fontsize=20, labelpad=10)
            plt.ylabel('Normalised Complexity', fontsize=20, labelpad=10)
            plt.scatter(r_final[I, 0], r_final[I, 1], s=200, color='red', alpha=1, marker='o', linewidth=2,facecolors='none',
                        label='Optimum')
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(borderpad=1, fontsize=16)
            plt.savefig('optimisation_plot.png')
        else:
            obj_T = res.F
            struct_T = res.X
            obj = res.F
            struct = res.X


        ##################################################################################################
        #Posible calculation of center distance evolution
        ##################################################################################################
        #ret = [e.pop.get("F") for e in res.history]
        #print('RET', len(ret))
        #print('RET2', ret[0].shape)
        ## Estandarización
        #scalar1 = MinMaxScaler(feature_range=(0, 1))
        #scalar2 = MinMaxScaler(feature_range=(0, 1))
#
        #scalar1.fit(np.concatenate((ret[0][0], ret[1][0], ret[2][0])).reshape(-1, 1))
        #scalar2.fit(np.concatenate((ret[0][1], ret[1][1], ret[2][1])).reshape(-1, 1))
        #scalar3.fit(np.concatenate((ret[0][2], ret[1][2], ret[2][2])).reshape(-1, 1))
        #scalar4.fit(np.concatenate((ret[0][3], ret[1][3], ret[2][3])).reshape(-1, 1))
#
        #for i in range(3):
        #    ret[i][0] = scalar1.transform(ret[i][0].reshape(-1, 1))[:, 0]
        #    ret[i][1] = scalar2.transform(ret[i][1].reshape(-1, 1))[:, 0]
        #    ret[i][2] = scalar3.transform(ret[i][2].reshape(-1, 1))[:, 0]
        #    ret[i][3] = scalar4.transform(ret[i][3].reshape(-1, 1))[:, 0]
#
        ## Center distance
        #F3 = [0 for x in range(len(ret))]
        #for i in range(len(ret)):
        #    F1 = ret[i]
        #    F2 = np.sqrt(np.sum(F1 ** 2, axis=1))
        #    F3[i] = np.sum(F2) / len(ret[i])
#
        #plt.figure(figsize=(10, 7))
        #plt.plot(np.arange(len(ret)), F3, color='black')
        #plt.xticks(fontsize=18)
        #plt.yticks(fontsize=18)
        #plt.xlabel("Generation",fontsize=20, labelpad=10)
        #plt.ylabel("Center distance",fontsize=20, labelpad=10)
        #plt.show()
        #plt.savefig("convergence1.png")

        ###############################################################################
        #Plot of Hypervolume; metric for evaluate the convergence
        ###############################################################################
        from pymoo.indicators.hv import Hypervolume
        X, F = res.opt.get("X", "F")
        n_evals = []  # corresponding number of function evaluations
        hist_F = []  # the objective space values in each generation
        hist_cv = []  # constraint violation in each generation
        hist_cv_avg = []  # average constraint violation in the whole population
        hist = res.history
        for algo in hist:
            # store the number of function evaluations
            n_evals.append(algo.evaluator.n_eval)

            # retrieve the optimum from the algorithm
            opt = algo.opt

            # store the least contraint violation and the average in each population
            hist_cv.append(opt.get("CV").min())
            hist_cv_avg.append(algo.pop.get("CV").mean())

            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])

        approx_ideal = F.min(axis=0)
        approx_nadir = F.max(axis=0)

        metric = Hypervolume(ref_point=np.array([1.1, 1.1]),
                             norm_ref_point=False,
                             zero_to_one=True,
                             ideal=approx_ideal,
                             nadir=approx_nadir)

        hv = [metric.do(_F) for _F in hist_F]
        plt.figure(figsize=(10, 7))
        plt.plot(n_evals, hv, color='black', label="Avg. CV of Pop", linewidth=2)
        plt.scatter(n_evals, hv, facecolor="none", edgecolor='black',linewidths=1.5, marker="o",s=40)
        plt.xlabel("Function Evaluations",fontsize=20, labelpad=10)
        plt.ylabel("Hypervolume",fontsize=20, labelpad=10)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()
        plt.savefig("convergence_HV.png")
        ###############################################################################

        print('The number of evaluations were:', contador)
        if n_processes>1:
            pool.close()
        else:
            pass

        return (obj, struct,obj_T, struct_T,  res,contador,ct_total)

    def optimal_search_nsga2(self,model,l_lstm, l_dense, batch, pop_size, tol,xlimit_inf, xlimit_sup, mean_y,parallel, onebyone, values, weights, n_last=5, nth_gen=5):
        '''
        :param model: object of ML or DL (class)
        :param l_lstm: maximun layers lstm (first layer never 0 neurons (input layer))
        :param l_dense: maximun layers dense
        :param batch: batch size
        :param pop_size: population size for NSGA2-
        :param tol: tolerance to built the pareto front
        :param xlimit_inf: array with lower limits for neurons lstm (range of number multiplied by 10), dense (range of number multiplied by 10) and
        :param xlimit_sup: array with upper limits for neurons lstm, dense and pacience
        :param mean_y: vector of means
        :param parallel: how many processes are parallelise
        :param onebyone: how divide the data: [0] True 1 by 1, [1] True lags by lags, [1] False steps by steps        :param values specific values to divide the sample. specific values of a variable to search division
        :param weigths: weights for the objective functions (*AL REVES)
        :param n_last: last generation considered in the search
        :param nth_gen: number of generation to evaluate
        :return: the option selected for the pareto front, the optimal selection and the total results
        '''

        manager = multiprocessing.Manager()
        dictionary = manager.dict()
        contador = manager.list()
        contador.append(0)
        print('start optimisation!!!')
        obj, x_obj, obj_total, x_obj_total,res,evaluations,tiempo = self.nsga2_individual(model,mean_y, contador,parallel,l_lstm, l_dense, batch,pop_size,tol, n_last, nth_gen,xlimit_inf, xlimit_sup,dictionary, onebyone,values, weights)

        np.savetxt('objectives_selected.txt', obj)
        np.savetxt('x_selected.txt', x_obj)
        np.savetxt('objectives.txt', obj_total)
        np.savetxt('x.txt', x_obj_total)
        np.savetxt('evaluations.txt', evaluations)

        print('Process finished!!!')
        print('The selection is \n', x_obj, 'with a result of \n', obj)
        print('time spent: ', tiempo)
        res = {'total_x': x_obj_total, 'total_obj': obj_total, 'opt_x': x_obj, 'opt_obj':obj, 'res':res,'evaluations':evaluations, 'tiempo':tiempo}
        return res


    def rnsga2_individual(self,model,med, contador,n_processes,l_lstm, l_dense, batch,pop_size,tol,n_last, nth_gen,xlimit_inf, xlimit_sup,dictionary,onebyone,values,weights,ref_points,epsilon):
        from MyProblem_lstm import MyProblem_lstm
        '''
        :param model: object of ML or DL (class)
        :param med:vector of means
        :param contador: a operator to count the attempts
        :param n_processes: how many processes are parallelise
        :param l_lstm:maximun number of layers lstm
        :param l_dense:maximun number of layers dense
        :param batch: batch size
        :param pop_size: population size selected for NSGA2
        :param tol: tolearance selected to terminate the process
        :param n_last: last generation considered in the search
        :param nth_gen: number of generation to evaluate
        :param xlimit_inf: array with the lower limits to the neuron  lstm , neurons dense and pacience
        :param xlimit_sup:array with the upper limits to the neuron  lstm , neurons dense and pacience
        :param dictionary: dictionary to stored the options tested
        :param onebyone: how divide the data: [0] True 1 by 1, [1] True lags by lags, [1] False steps by steps        :param values specific values to divide the sample. specific values of a variable to search division
        :param weigths: weights for the objective functions
                :param ref_points:reference points for algorithm initialisation i.e np.array([[0.3, 0.1], [0.1, 0.3]])
        :param epsilon: smaller generates solutions tighter
        :return: options in Pareto front, the optimal selection and the total results. Consider the option of parallelisation with runners
        '''
        # Define the problem class considering if parallelisation is wanted
        if n_processes>1:
            pool = multiprocessing.Pool(n_processes)
            problem = MyProblem_lstm(model,self.names,self.extract_cero,self.horizont, self.scalar_y, self.zero_problem, self.limits,self.times,self.pos_y,self.mask,
                                self.mask_value, self.n_lags,self.n_steps,self.inf_limit, self.sup_limit, self.repeat_vector, self.type, self.data,
                                self.scalar_x,self.dropout,self.weights,med, contador,len(xlimit_inf),l_lstm, l_dense, batch, xlimit_inf, xlimit_sup,dictionary,onebyone,values,
                                     self.optimizer,self.learning_rate, self.activation,runner = pool.starmap,func_eval=StarmapParallelization)
        else:
            problem = MyProblem_lstm(model,self.names,self.extract_cero,self.horizont, self.scalar_y, self.zero_problem, self.limits,self.times,self.pos_y,self.mask,
                                self.mask_value, self.n_lags,self.n_steps,self.inf_limit, self.sup_limit, self.repeat_vector, self.type, self.data,
                                self.scalar_x, self.dropout,self.weights,med, contador,len(xlimit_inf),l_lstm, l_dense, batch, xlimit_inf, xlimit_sup,dictionary,onebyone,values,
                                     self.optimizer,self.learning_rate, self.activation)

        # Algorithm for optimisation
        algorithm = RNSGA2(ref_points, pop_size=pop_size, sampling=get_sampling("int_random"),
                          crossover=get_crossover("int_sbx"),
                          mutation=get_mutation("int_pm", prob=0.1),
                           normalization='front',
                           extreme_points_as_reference_points=False,
                           weights=weights,
                           epsilon=epsilon)

        # Termination of the algorithm based on tolerance
        termination = MultiObjectiveSpaceTermination(tol=tol,
                                                              n_last=n_last, nth_gen=nth_gen, n_max_gen=None,
                                                              n_max_evals=6000)
        '''
        Termination can be with tolerance or with generations limit
        '''
        ct = time()
        res = minimize(problem,
                       algorithm,
                       termination,
                       pf=True,
                       verbose=True,
                       seed=7)
        ct_total = time()-ct

        #We select the final optimum with normalized results
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
            I=PBI.do(r_final, weights).argmin()

            obj_T = res.F
            struct_T = rx
            obj = res.F[I, :]
            struct = rx[I, :]

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

        from pymoo.indicators.hv import Hypervolume
        X, F = res.opt.get("X", "F")
        n_evals = []  # corresponding number of function evaluations
        hist_F = []  # the objective space values in each generation
        hist_cv = []  # constraint violation in each generation
        hist_cv_avg = []  # average constraint violation in the whole population
        hist = res.history
        for algo in hist:
            # store the number of function evaluations
            n_evals.append(algo.evaluator.n_eval)

            # retrieve the optimum from the algorithm
            opt = algo.opt

            # store the least contraint violation and the average in each population
            hist_cv.append(opt.get("CV").min())
            hist_cv_avg.append(algo.pop.get("CV").mean())

            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])

        approx_ideal = F.min(axis=0)
        approx_nadir = F.max(axis=0)

        metric = Hypervolume(ref_point=np.array([1.1, 1.1]),
                             norm_ref_point=False,
                             zero_to_one=True,
                             ideal=approx_ideal,
                             nadir=approx_nadir)

        hv = [metric.do(_F) for _F in hist_F]
        plt.figure(figsize=(10, 7))
        plt.plot(n_evals, hv, color='black', label="Avg. CV of Pop", linewidth=2)
        plt.scatter(n_evals, hv, facecolor="none", edgecolor='black',linewidths=1.5, marker="o",s=40)
        plt.xlabel("Function Evaluations",fontsize=20, labelpad=10)
        plt.ylabel("Hypervolume",fontsize=20, labelpad=10)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()
        plt.savefig("convergence_HV.png")
        ###############################################################################

        print('The number of evaluations were:', contador)
        if n_processes>1:
            pool.close()
        else:
            pass

        return (obj, struct,obj_T, struct_T,  res,contador,ct_total)


    def optimal_search_rnsga2(self,model,l_lstm, l_dense, batch, pop_size, tol,xlimit_inf, xlimit_sup, mean_y,parallel, onebyone, values, weights,ref_points=np.array([[0.3, 0.1], [0.1, 0.3]]), epsilon=0.01,n_last=5, nth_gen=5):
        '''
        :param model: object of ML or DL (class)
        :param l_dense: maximun layers dense
        :param batch: batch size
        :param pop_size: population size for RVEA
        :param tol: tolerance to built the pareto front
        :param xlimit_inf: array with lower limits for neurons lstm, dense and pacience
        :param xlimit_sup: array with upper limits for neurons lstm, dense and pacience
        :param mean_y: vector of means
        :param dropout: percentage for NN
        :param parallel: how many processes are parallelise
        :param weights: weights for the two objective functions (*AL REVES)
        :param ref_points:reference points for algorithm initialisation i.e np.array([[0.3, 0.1], [0.1, 0.3]])
        :param epsilon: parameter of RNSGA
        n_last: more robust, we consider the last n generations and take the maximum
        nth_gen: whenever the termination criterion is calculated
        if mean_y is empty a variation rate will be applied
        :param onebyone: how divide the data: [0] True 1 by 1, [1] True lags by lags, [1] False steps by steps        :return: the options selected for the pareto front, the optimal selection and the total results
        '''

        # Multiprocessing for possible paralelisation and fill the dictionary and the contador
        manager = multiprocessing.Manager()
        dictionary = manager.dict()
        contador = manager.list()
        contador.append(0)
        print('start optimisation!!!')

        obj, x_obj, obj_total, x_obj_total,res,evaluations,time = self.rnsga2_individual(model,mean_y, contador,parallel,l_lstm, l_dense, batch,pop_size,tol,n_last, nth_gen,xlimit_inf, xlimit_sup,dictionary, onebyone,values, weights,ref_points,epsilon)

        np.savetxt('objectives_selectedR.txt', obj)
        np.savetxt('x_selectedR.txt', x_obj)
        np.savetxt('objectivesR.txt', obj_total)
        np.savetxt('xR.txt', x_obj_total)
        np.savetxt('evaluationsR.txt', evaluations)

        print('Process finished!!!')
        print('The selection is \n', x_obj, 'with a result of \n', obj)
        res = {'total_x': x_obj_total, 'total_obj': obj_total, 'opt_x': x_obj, 'opt_obj':obj, 'res':res,'evaluations':evaluations}
        return res
