from pymoo.core.repair import Repair
from MyProblem_svm import MyProblem_svm

class MyRepair_svm(Repair):
    def info(self):
        print('Class defining a function to repair the possible error of the genetic algorithm. If a layer is zero the next layer cannot have positive neurons')
    def __init__(self):
        pass

    def _do(self, problem, pop, **kwargs):
        for k in range(len(pop)):
            x = pop[k].X
            #x2 = x[range(self.l_dense)]
            r_dense = MyProblem_svm.bool(x)
            if r_dense == 0:
                pass
            elif r_dense != 0:
                x[1] = x[0]-0.1

            pop[k].X = x
        return pop
