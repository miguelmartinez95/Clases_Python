from pymoo.core.repair import Repair
import numpy as np
from MyProblem_lstm import MyProblem_lstm

class MyRepair_lstm(Repair):
    def info(self):
        print('Class defining a function to repair the possible error of the genetic algorithm. If a layer is zero the next layer cannot have positive neurons')

    def __init__(self,l_lstm, l_dense):
        self.l_lstm=l_lstm
        self.l_dense = l_dense
        '''
        l_lstm: number of LSTM layers
        l_dense: number of Dense layers
        '''


    def _do(self, problem, pop, **kwargs):
        '''
        :param pop: the different options of the neurons in this case
        :return: the vector of options but corrected with 0 after a layer without neuros (0)
        '''

        print('FIXING X')
        for k in range(len(pop)):
            x = pop[k].X
            #we separate the part in X for layers lstm and dense to repair
            xx = x[range(self.l_lstm + self.l_dense)]
            x1 = xx[range(self.l_lstm)]
            x2 = xx[range(self.l_lstm, self.l_lstm + self.l_dense)]
            x3 = np.delete(x, np.arange(self.l_lstm+self.l_dense))
            #check the results of constraints
            r_lstm, r_dense = MyProblem_lstm.bool4(x1,x2)
            if len(r_lstm) == 1:
                if r_lstm == 0:
                    pass
                elif r_lstm != 0:
                    x1[r_lstm] = 0
            elif len(r_lstm) > 1:
                x1[r_lstm] = 0

            if len(r_dense) == 1:
                if r_dense == 0:
                    pass
                elif r_dense != 0:
                    x2[r_dense] = 0
            elif len(r_dense) > 1:
                x2[r_dense] = 0
            #merge the correction and the other part of x
            x = np.concatenate((x1, x3))
            pop[k].X = x

        print('X FIXED')
        return pop
