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
            if self.real.shape[1]>=2:
                cv = [0 for x in range(self.real.shape[1])]
                for i in range(self.real.shape[1]):
                    cv[i] = 100 * (np.sqrt(metrics.mean_squared_error(self.real[:, i], self.predict[:, i])) / mean[i])
            else:
                cv = 100 * (np.sqrt(metrics.mean_squared_error(self.real, self.predict)) / mean)
        else:
            cv = 100 * (np.sqrt(metrics.mean_squared_error(self.real, self.predict)) / mean)
        return (cv)

    def cv_rmse_daily(self, mean,times):
        '''
        :return: cv(rmse) day by day
        '''
        days =times.day
        cv=[0 for x in range(len(np.unique(days)))]
        print(self.real.shape)
        print(self.predict.shape)
        for i in range(len(np.unique(days))):
            ii=np.where(days==np.unique(days)[i])[0]
            #cv[i]=100 * (np.sqrt(metrics.mean_squared_error(self.real.iloc[ii], self.predict.iloc[ii])) / mean)
            cv[i]=100 * (np.sqrt(metrics.mean_squared_error(self.real[ii], self.predict[ii])) / mean)

        std = np.std(cv)
        return(cv,std)

    def rmse(self):
        if len(self.real.shape) > 1:
            if self.real.shape[1] >= 2:
                rmse = [0 for x in range(self.real.shape[1])]
                for i in range(self.real.shape[1]):
                    rmse[i] = 100 * (np.sqrt(metrics.mean_squared_error(self.real[:, i], self.predict[:, i])))
            else:
                rmse = np.sqrt(metrics.mean_squared_error(self.real, self.predict))
        else:
            rmse = np.sqrt(metrics.mean_squared_error(self.real, self.predict))
        return(rmse)

    def rmse_daily(self,times):
        '''
        :return: rmse day by day
        '''
        days = times.day
        rmse= [0 for x in range(len(np.unique(days)))]
        for i in range(len(np.unique(days))):
            ii = np.where(days == np.unique(days)[i])[0]
            rmse[i] = np.sqrt(metrics.mean_squared_error(self.real[ii], self.predict[ii]))

        std = np.std(rmse)
        return(rmse,std)

    def nmbe(self,mean):
        if len(self.real.shape) > 1:
            if self.real.shape[1] >= 2:
                nmbe = [0 for x in range(self.real.shape[1])]
                for i in range(self.real.shape[1]):
                    y_true = np.array(self.real[:,i])
                    y_pred = np.array(self.predict[:,i])
                    y_true = y_true.reshape(len(y_true), 1)
                    y_pred = y_pred.reshape(len(y_pred), 1)
                    nmbe[i]= np.mean(y_true - y_pred) / mean[i]
            else:
                y_true = np.array(self.real)
                y_pred = np.array(self.predict)
                y_true = y_true.reshape(len(y_true), 1)
                y_pred = y_pred.reshape(len(y_pred), 1)
                nmbe = np.mean(y_true - y_pred) / mean

        else:
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
        nmbe = [0 for x in range(len(np.unique(days)))]
        for i in range(len(np.unique(days))):
            ii = np.where(days == np.unique(days)[i])[0]
            nmbe[i] = 100*(np.mean(y_true[ii] - y_pred[ii]) / mean)

        std=np.std(nmbe)
        return (nmbe,std)

    def r2(self):
        r2=metrics.r2_score(self.real, self.predict)
        return(r2)

    def variation_rate(self):
        if len(self.real.shape) > 1:
            if self.real.shape[1] >= 2:
                var = [0 for x in range(self.real.shape[1])]
                for i in range(self.real.shape[1]):
                    a=self.real[:,i]
                    a[np.where(a==0)[0]]=1
                    var[i] = np.mean(np.abs((self.real[:, i]- self.predict[:, i])/a))
            else:
                a = self.real
                a[np.where(a == 0)[0]] = 1
                var = np.mean((self.real- self.predict)/a)
        else:
            a = self.real
            a[np.where(a == 0)[0]] = 1
            var = np.mean((self.real - self.predict) / a)
        return(var)
