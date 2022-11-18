from pymoo.core.repair import Repair
from MyProblem_mlp import MyProblem_mlp
import numpy as np

class MyRepair_mlp(Repair):
    def info(self):
        print('Class defining a function to repair the possible error of the genetic algorithm. If a layer is zero the next layer cannot have positive neurons')
    def __init__(self,l_dense):
        self.l_dense = l_dense
    def _do(self, problem, pop, **kwargs):
        for k in range(len(pop)):
            x = pop[k].X
            x2 = x[range(self.l_dense)]
            r_dense = MyProblem_mlp.bool4(x, self.l_dense)
            if len(r_dense) == 1:
                if r_dense == 0:
                    pass
                elif r_dense != 0:
                    x2[r_dense] = 0
            elif len(r_dense) > 1:
                x2[r_dense] = 0
            x = np.concatenate((x2, np.array([x[len(x) - 1]])))
            pop[k].X = x
        return pop
