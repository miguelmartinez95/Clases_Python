from pymoo.core.repair import Repair
import numpy as np
from MyProblem_lstm import MyProblem_lstm

class MyRepair_lstm(Repair):
    def info(self):
        '''
        l_lstm: number of LSTM layers
        l_dense: number of Dense layers
        :return:
        '''
        print('Class defining a function to repair the possible error of the genetic algorithm. If a layer is zero the next layer cannot have positive neurons')

    def __init__(self,l_lstm, l_dense):
        self.l_lstm=l_lstm
        self.l_dense = l_dense

    def _do(self, problem, pop, **kwargs):
        print('FIXING X')
        for k in range(len(pop)):
            x = pop[k].X
            xx = x[range(self.l_lstm + self.l_dense)]
            x1 = xx[range(self.l_lstm)]
            x2 = xx[range(self.l_lstm, self.l_lstm + self.l_dense)]
            r_lstm, r_dense = MyProblem_lstm.bool4(xx, self.l_lstm, self.l_dense)

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
            x = np.concatenate((x1, x2, np.array([x[len(x) - 1]])))
            pop[k].X = x

        print('X FIXED')
        return pop
