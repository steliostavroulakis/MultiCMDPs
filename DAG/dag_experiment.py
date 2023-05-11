from dag_train import train_dag
from lib import create_dag
import unittest


class DAGExperiments(unittest.TestCase):
    G = create_dag()

    def experiment_0(self):
        player_names = ['Alice', 'Bob', 'Charlie']
        player_constrains = [6, 6, 14]
        player_max_lambdas = [10, 10, 10]
        use_max_lambda = False

        iterates = 1000
        primal_step = 0.005
        dual_step = 0.1

        train_dag(player_names=player_names, G=G, player_constrains=player_constrains,
                  player_max_lambdas=player_max_lambdas,iterates=iterates, use_max_lambda=use_max_lambda,
                  gitprimal_step=primal_step, dual_step=dual_step,
                  pic_name="test")
