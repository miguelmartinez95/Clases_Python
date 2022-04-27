import numpy as np
import pandas as pd
from sklearn import metrics
import numpy as np

class Eval_metrics:

    def info(self):
        print(("Different error metrics, together with the R2, to carry out: \n" 
          "- CV(RMSE) \n"
           "- RMSE \n"
          "- NMBE \n"
          "- R2"))

    #def __init__(self, predict, real):
    #    self.predict= pd.DataFrame(predict).iloc[:,0]
    #    self.real = pd.DataFrame(real).iloc[:,0]
    #
    def __init__(self, predict, real):
        self.predict= predict
        self.real = real

    #def cv_rmse(self, mean):
    #    cv = 100 * (np.sqrt(metrics.mean_squared_error(self.real, self.predict)) / mean)
    #    return(cv)

    def cv_rmse(self, mean):
        if len(self.real.shape) > 1:
            cv = [0 for x in range(self.real.shape[1])]
            for i in range(self.real.shape[1]):
                cv[i] = 100 * (np.sqrt(metrics.mean_squared_error(self.real[:, i], self.predict[:, i])) / mean[i])

            cv= np.mean(cv)
        else:
            cv = 100 * (np.sqrt(metrics.mean_squared_error(self.real, self.predict)) / mean)
        return (cv)

    def cv_rmse_daily(self, mean,times):
        '''
        :return: cv(rmse) day by day
        '''
        days =times.day
        dd= pd.concat([self.real, self.predict,days ])
        cv=[0 for x in range(len(np.unique(days)))]
        for i in np.unique(days):
            ii=np.where(days==i)
            cv[i]=100 * (np.sqrt(metrics.mean_squared_error(self.real.iloc[ii], self.predict.iloc[ii])) / mean)

        return(cv)

    def rmse(self):
        rmse = np.sqrt(metrics.mean_squared_error(self.real, self.predict))
        return(rmse)

    def rmse_daily(self,times):
        '''
        :return: rmse day by day
        '''
        days = times.day
        dd = pd.concat([self.real, self.predict, days])
        rmse= [0 for x in range(len(np.unique(days)))]
        for i in np.unique(days):
            ii = np.where(days == i)
            rmse[i] = np.sqrt(metrics.mean_squared_error(self.real.iloc[ii], self.predict.iloc[ii]))

        return(rmse)

    def nmbe(self,mean):
        y_true = np.array(self.real)
        y_pred = np.array(self.predict)
        y_true = y_true.reshape(len(y_true), 1)
        y_pred = y_pred.reshape(len(y_pred), 1)
        nmbe = np.mean(y_true - y_pred) / mean
        return (nmbe * 100)

    def nmbe_daily(self,mean,times):
        '''
        :return: nmbe day by day
        '''
        y_true = np.array(self.real)
        y_pred = np.array(self.predict)
        y_true = y_true.reshape(len(y_true), 1)
        y_pred = y_pred.reshape(len(y_pred), 1)
        days = times.day
        dd = pd.concat([self.real, self.predict, days])
        nmbe = [0 for x in range(len(np.unique(days)))]
        for i in np.unique(days):
            ii = np.where(days == i)
            nmbe[i] = 100*(np.mean(y_true.iloc[ii] - y_pred.iloc[ii]) / mean)


        return (nmbe)

    def r2(self):
        r2=metrics.r2_score(self.real, self.predict)
        return(r2)