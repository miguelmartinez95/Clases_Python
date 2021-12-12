import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class Exploratory_analysis():

    def info(self):
        print(("Tools to make a initial exploratory analysis of the data: \n"
          "- Missing values \n"
          "- Plots of variables \n"
          "- Correlation with the dependent variable \n"
          "- Outliers detection"))

    def __init__(self,data):
        self.data = data


    def missing_values(self):
        data= pd.DataFrame(self.data)
        print('Con', data.shape[0],'de datos los valores fatlantes son', data.isna().sum())
        data.isna().sum().reset_index(name="n").plot.bar(x='index', y='n', rot=40)

    def hists(self):
        data = self.data.reset_index(drop=True)
        for i in range(data.shape[1]):
            plt.figure()
            plt.hist(self.data.iloc[:, i])
            plt.title(self.data.columns[i])

    def plots(self):
        time = self.data.index
        data = self.data.reset_index(drop=True)
        for i in range(data.shape[1]):
            plt.figure()
            try:
                plt.plot(time.to_pydatetime(),self.data.iloc[:,i])
            except:
                time = pd.to_datetime(time)
                plt.plot(time, self.data.iloc[:, i])
            plt.title(self.data.columns[i])

    def correlations(self, position_y):
        correlation = self.data.corr().iloc[:,position_y]
        print(correlation)


    def outliers(self, data_no_NAs,position_y, contamination):
        '''
        :param data_no_NAs: daat without missing values
        :param contamination: assumed proportion of outliers
        :return: dictionary with the outliers detected in each of the variables
        '''
        from sklearn.ensemble import IsolationForest
        if contamination==None:
            iso = IsolationForest(contamination=0.1)
        else:
            iso = IsolationForest(contamination=contamination)
        outliers_total = dict()
        if len(data_no_NAs.shape)>1:
            for t in range(data_no_NAs.shape[1]):
                dd = np.array(data_no_NAs.iloc[:,t]).reshape(-1, 1)
                yhat = iso.fit_predict(dd)
                ind1= np.arange(data_no_NAs.shape[0])
                outliers = ind1[yhat == -1]
                outliers_total[data_no_NAs.columns[t]]=outliers
                print('Siendo la media de',data_no_NAs.columns[t], np.mean(data_no_NAs.iloc[:,t]), 'los outliers detectados son',len(outliers),'y son:', data_no_NAs.iloc[:,t][outliers])

        else:
            dd = np.array(data_no_NAs).reshape(-1,1)
            yhat = iso.fit_predict(data_no_NAs)
            ind1= np.arange(data_no_NAs.shape[0])
            outliers = ind1[yhat == -1]
            outliers_total[data_no_NAs.columns]=outliers
            print('Siendo la media de',data_no_NAs.columns[t], np.mean(dd),'los outliers detectados son:', dd[outliers])

        return outliers_total