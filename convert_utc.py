import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import pytz


def convert_timezone_indv(dt, tz1, tz2):
    '''
    :param dt: date value
    :param tz1: time zone original
    :param tz2: time zone objective
    :return: date value converted
    '''
    tz1 = pytz.timezone(tz1)
    tz2 = pytz.timezone(tz2)

    dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    dt = tz1.localize(dt)
    dt = dt.astimezone(tz2)
    dt = dt.strftime("%Y-%m-%d %H:%M:%S")

    return dt
class Solving_utc:

    def info(self):
        print(('Class to convert the datetime in UTC format. Built not to have the hour missing in March 11/10/2021 \n'
           '0: Time zona \n' 
           '1: UTC+1 \n' 
           '2:UTC+2'))



    def __init__(self,case,origin, freq,freq_n, start, end):
        self.case = case
        self.origin=origin
        self.freq = freq
        self.freq_n = freq_n
        self.start = start
        self.end = end


    def convert_join(self, data):
        '''
        Taking into account the case, data is converted in UTC format based on the original case
        '''

        if self.case==0:

            q1 = convert_timezone_indv(self.start, self.origin, 'UTC')
            q2 = convert_timezone_indv(self.end, self.origin, 'UTC')
            times_utc = pd.date_range(start=q1, end=q2, freq=self.freq, tz='UTC')
            try:
                data.index=times_utc
            except:
                raise NameError('Problems with the utc index in time zone')

        elif self.case==1:
            step = int(60/self.freq_n)
            data= data.drop(range(step))
            times_utc = pd.date_range(start=self.start, end=self.end, freq=self.freq, tz='UTC')
            times_utc=times_utc.delete(range(len(times_utc)-(step), len(times_utc)))
            try:
                data.index=times_utc
            except:
                raise NameError('Problems with utc index in UTC+1')

        elif self.case==2:
            step = int(60 / self.freq_n)
            data = data.drop(range(step*2))
            times_utc = pd.date_range(start=self.start, end=self.end, freq=self.freq, tz='UTC')
            times_utc = times_utc.delete(range(len(times_utc) - (step*2), len(times_utc)))
            try:
                data.index = times_utc
            except:
                raise NameError('Problems with utc index in UTC+2')

        return(data)

    def delete_hour_missing(self,data):
        '''
        Find the missing hour (without data for the time format) and delete it.
        '''
        times1 = pd.date_range(start=self.start, end=self.end, freq=self.freq, tz='UTC')

        step = int(60 / self.freq_n)
        months = times1.month
        hours = times1.hour
        weekday = times1.weekday
        ind1 = np.where((months == 3) & (weekday == 6) & (hours == 2))[0]
        if len(ind1)>0:
            ind1 = ind1[(len(ind1) - step):len(ind1)]+1
            print(ind1)
            data = data.drop(data.index[ind1])
            print('Hour deleted')
        else:
            raise NameError('There is not hour missing at March')

        return(data)

