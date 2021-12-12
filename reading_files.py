import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import chain
from datetime import datetime

class Reading_files:

    def info(self):
        print('Reading files .csv or .txt from files or folders or dowloading weather data from cronograph')


    def __init__(self, destination):
        self.destination = destination


    def reading(self, directory, n_folders, separator, decimal, col_index, type, export):
        '''
        :param directory: place where the data is stored
        :param n_folders: there is more than one folder or not
        :param separator: data separator
        :param decimal: separtor symbol
        :param col_index: a initial column working as index
        :param type: csv, txt, etc
        :param export: True or False
        :return: joined data frama
        '''
        sep = "\\"
        try:
            if n_folders > 1:
                count1 = 1
                folders = os.listdir(directory)
                print(folders)
                if len(folders) > 0:
                    for a in folders:
                        print(count1)
                        new = [directory, a]
                        directory1 = Path(sep.join(new))
                        list_f = os.listdir(directory1)
                        print(list_f)
                        if len(list_f) > 0:
                            count = 1
                            for filename in list_f:
                                print(count)
                                if filename.endswith(type):
                                    new = [str(directory1), filename]
                                else:
                                    raise NameError('Some file is not'+type)
                                if count == 1 and type=='csv':
                                    df = pd.read_csv(sep.join(new), sep=separator, decimal=decimal, index_col=col_index)
                                elif count==1 and type=='txt':
                                    df = pd.read_table(sep.join(new), sep=separator, decimal=decimal, index_col=col_index)
                                elif count>1 and type=='csv':
                                    df = pd.concat([df, pd.read_csv(sep.join(new), sep=separator, decimal=decimal,
                                                                    index_col=col_index)], axis=0)
                                else:
                                    df = pd.concat([df, pd.read_table(sep.join(new), sep=separator, decimal=decimal,
                                                                    index_col=col_index)], axis=0)
                                count += 1

                            if count1 == 1:
                                df_total = df
                            else:
                                df_total = pd.concat([df_total, df], axis=0)
                            count1 += 1
                        else:
                            raise NameError('There are not files in the folder')

                else:
                    raise NameError('There are not folders')
            else:
                list_f = os.listdir(directory)
                print(list_f)
                if len(list_f) == 1 and type=='csv':
                    df = pd.read_csv(directory, sep=separator, decimal=decimal, index_col=col_index)
                elif len(list_f) == 1 and type=='txt':
                    df = pd.read_table(directory, sep=separator, decimal=decimal, index_col=col_index)
                else:
                    count = 1
                    if len(list_f) > 0:
                        for filename in list_f:
                            print(count)
                            if filename.endswith(type):
                                new = [str(directory), filename]
                            else:
                                raise NameError('Some file is not' + type)
                            if count == 1 and type == 'csv':
                                df = pd.read_csv(sep.join(new), sep=separator, decimal=decimal, index_col=col_index)
                            elif count == 1 and type == 'txt':
                                df = pd.read_table(sep.join(new), sep=separator, decimal=decimal, index_col=col_index)
                            elif count > 1 and type == 'csv':
                                df = pd.concat([df, pd.read_csv(sep.join(new), sep=separator, decimal=decimal,
                                                                index_col=col_index)], axis=0)
                            else:
                                df = pd.concat([df, pd.read_table(sep.join(new), sep=separator, decimal=decimal,
                                                                  index_col=col_index)], axis=0)
                            count += 1
                        df_total = df

                    else:
                        raise NameError('There are not files in the folder')
            #df_total.index = np.arange(0, df_total.shape[0])
            return df_total
        except:
            raise NameError('General problem')

        new = [self.destination, 'data_created.csv']
        desti = sep.join(new)
        if export==True:
            df_total.to_csv(desti, index=True, sep=';', decimal='.')
        else:
            pass

    def weather_cronograph_new(self, var_name, sensor_name, host, time_start, time_end, step, export):
        '''
        :param var_name: name of selected variables
        :param sensor_name: sensor used for collecting variable
        :param host: weather = none
        :param step: data frequency
        :param export: True or False
        :return:data downloaded from crnograph (new version)
        '''
        import influxdb_client
        from influxdb_client import InfluxDBClient, Point, Dialect
        from influxdb_client.client.write_api import SYNCHRONOUS

        variables=[]
       # client = InfluxDBClient(url="http://172.20.34.127:8086",
       #                         token='_NjAhMS1RtaGJG7pIvENLwVOPEzClu9Krl-em0zfTHuMxR11jDkPT9R9NFUPlGB22GCNdAfl3g5jBgiDMZCl6w==',
       #                         org='gte')
        client = InfluxDBClient(url="http://fluidos.uvigo.es:8086",
                               token='_NjAhMS1RtaGJG7pIvENLwVOPEzClu9Krl-em0zfTHuMxR11jDkPT9R9NFUPlGB22GCNdAfl3g5jBgiDMZCl6w==',
                               org='gte')

        sep = '.'
        time_start_str = time_start.strftime('%Y-%m-%dT%H:%M:%SZ')
        time_end_str = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')
        query_api = client.query_api()
        if host == 'none':

            for u in range(len(var_name)):

                q = f'''
                              from(bucket: "sensors/data")|> range(start:{time_start}, stop:{time_end}) 
                              |> filter(fn: (r) => r["_measurement"] ==  "{sensor_name}")
                                               |> filter(fn: (r) => r["_field"] == "{var_name[u]}")
                                               |> aggregateWindow(every: {step}, fn: mean, createEmpty: true)
                                               |> yield(name: "{var_name[u]}")

                          '''

                tables = query_api.query(q)

                values = [0 for x in range(len(list(tables[0])))]
                j = 0
                for table in tables:
                    # print(table)
                    for record in table.records:
                        # print(record.values)
                        values[j] = record.values['_value']
                        j += 1

                variables.append(values)

        matrix = np.zeros((len(variables[0]), len(variables)))
        for i in range(len(variables)):
            matrix[:, i] = pd.Series(variables[i])

        var_meteo = pd.DataFrame(matrix)

        step2 = step + 'in'
        times = pd.date_range(start=time_start_str, periods=var_meteo.shape[0], freq=step2)
        var_meteo.index = times

        new = [self.destination, 'weather_created.csv']
        desti = sep.join(new)
        var_meteo.columns = var_name
        if export==True:
            var_meteo.to_csv(desti, index=True, sep=';', decimal='.')
        else:
            pass

        return var_meteo



    def weather_cronograph_old(self, var_name, sensor_name, host, time_start, time_end, step, export):
        '''
        :param var_name: name of selected variables
        :param sensor_name: sensor used for collecting variable
        :param host: weather = none
        :param step: data frequency
        :param export: True or False
        :return: data downloaded from cronograph (old version)
        '''
        import influxdb
        variables=[]
        CONN_STR = "influxdb://admin:4dm1n_P4ss*@fluidos.uvigo.es:18086"
        #CONN_STR = "influxdb://admin:4dm1n_P4ss*@172.20.34.127:18086"
        DB_NAME = "sensor_data"

        influx = influxdb.InfluxDBClient.from_dsn(CONN_STR)
        influx.switch_database(DB_NAME)

        place = ["sensor_data.autogen", sensor_name]
        sep = '.'
        place = sep.join(place)
        place2 = [sensor_name, "address"]
        sep = '.'
        place2 = sep.join(place2)
        time_start_str = time_start.strftime('%Y-%m-%dT%H:%M:%SZ')
        time_end_str = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')

        if host == 'none':
            for u in range(len(var_name)):
                print(var_name[u])
                var2 = [var_name[u], 'vc']
                sep = '_'
                var2 = sep.join(var2)
                query = f"""
                            SELECT mean({var_name[u]}) AS {var_name[u]} FROM {place}
                            WHERE time > '{time_start_str}' AND time <= '{time_end_str}' AND {var2}<3
                              AND {place2} != '69' GROUP BY time({step}) FILL(9999)
                        """
                try:
                    results = influx.query(query)
                except:
                    raise NameError('Problem in the conexion')

                point = list(results)[0]
                values = [0 for x in range(len(point))]
                for t in range(len(point)):
                    values[t] = point[t][var_name[u]]

                variables.append(values)

        else:
            print('The data required is not related with the weather')

        matrix = np.zeros((len(variables[0]), len(variables)))
        for i in range(len(variables)):
            matrix[:, i] = pd.Series(variables[i])

        var_meteo = pd.DataFrame(matrix)

        step2 = step + 'in'
        times = pd.date_range(start=time_start_str, periods=var_meteo.shape[0], freq=step2)
        var_meteo.index = times

        new = [self.destination, 'weather_created.csv']
        desti = sep.join(new)
        var_meteo.columns = var_name
        if export==True:
            var_meteo.to_csv(desti, index=True, sep=';', decimal='.')
        else:
            pass

        return (var_meteo)
#
#directory =r'E:\Documents\Doctorado\FDA_Minas\DATA\_15M'
#start = '2021-02-18 00:15:00'
#end = '2021-04-27 00:00:00'
#data = reading_files(directory,'15min',15,start, end, 'Europe/Madrid', False)
#data = data.reading(2, ';', ',', 0, '.csv')
#
#directory =r'E:\Documents\Doctorado\FDA_Minas\DATA\_5M'
#start = '2021-04-27 00:05:00'
#end = '2021-10-01 00:00:00'
#data2 = reading_files(directory,'5min',5,start, end,'Europe/Madrid', False)
#data2 = data2.reading(2, ';', ',', 0, '.csv')
