from pymoo.core.repair import Repair
from MyProblem_mlp import MyProblem_mlp
import numpy as np

class MyRepair_mlp(Repair):
    def info(self):
        print('Class defining a function to repair the possible error of the genetic algorithm. If a layer is zero the next layer cannot have positive neurons')
    def __init__(self,l_dense):
        self.l_dense = l_dense
    def _do(self, problem, pop, **kwargs):
        '''
        :param pop: the different options of the neurons in this case
        :return: the vector of options but corrected with 0 after a layer without neuros (0)
        '''
        for k in range(len(pop)):
            x = pop[k].X
            x2 = x[range(self.l_dense)]
            d= len(x)- len(x2)
            r_dense = MyProblem_mlp.bool4(x2, self.l_dense)
            if len(r_dense) == 1:
                if r_dense == 0:
                    pass
                elif r_dense != 0:
                    x2[r_dense] = 0
            elif len(r_dense) > 1:
                x2[r_dense] = 0
            x = np.concatenate((x2, np.array([x[len(x) - d]])))
            pop[k].X = x
        return pop
