import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
from scipy.optimize import linprog

import math
import matplotlib.animation as anim
import pprint
import sys
from scipy.optimize import minimize

def total_expected_load(G, players):
    expected_loads = {edge: 0 for edge in G.edges()}
    for player in players.values():
        for path_index, prob in enumerate(player.strategy):
            path = player.paths[path_index]
            for i in range(len(path)-1):
                edge = (path[i], path[i+1])
                expected_loads[edge] += prob
    return expected_loads

def single_expected_load(G, strategy, paths):
    expected_loads = {edge: 0 for edge in G.edges()}
    for path_index, prob in enumerate(strategy):
        path = paths[path_index]
        for i in range(len(path)-1):
            edge = (path[i], path[i+1])
            expected_loads[edge] += prob
    return expected_loads

def projsplx(y):
    s = np.sort(y)
    n = len(y) ; flag = False
    
    parsum = 0
    tmax = -np.inf
    for idx in range(n-2, -1, -1):
        parsum += s[idx+1]
        tmax = (parsum - 1) / (n - (idx + 1) )
        if tmax >= s[idx]:
            flag = True ; break
    
    if not flag:
        tmax = (np.sum(s) - 1) / n
    
    return np.maximum(y - tmax, 0)
   
def total_effective_congestion(true_cong_dict):

    highway = set()
    highway.add(('s','l1'))
    highway.add(('l1','l2'))
    highway.add(('l2','l3'))
    highway.add(('l3','l4'))
    highway.add(('l4','l5'))
    highway.add(('l5','l6'))
    highway.add(('l6','t'))
    
    ret_dict = dict()

    for edge in true_cong_dict:

        if edge in highway:
            ret_dict[edge] = 0.02*true_cong_dict[edge]
        else: 
            ret_dict[edge] = true_cong_dict[edge]

    return ret_dict

def total_gas_dict(true_cong_dict):
    highway = set()
    highway.add(('s','l1'))
    highway.add(('l1','l2'))
    highway.add(('l2','l3'))
    highway.add(('l3','l4'))
    highway.add(('l4','l5'))
    highway.add(('l5','l6'))
    highway.add(('l6','t'))
    
    ret_dict = dict()

    for edge in true_cong_dict:

        if edge in highway:
            ret_dict[edge] = 2*true_cong_dict[edge]
        else: 
            ret_dict[edge] = true_cong_dict[edge]
    return ret_dict

def create_dag():

    # Create a new directed acyclic graph
    G = nx.DiGraph()

    # Add nodes s and t to the graph
    G.add_node('s')
    G.add_node('t')

    # Add some edges to the graph
    G.add_edge('s', 'a')
    G.add_edge('s', 'b')
    G.add_edge('a', 'c')
    G.add_edge('b', 'c')
    G.add_edge('a', 'd')
    G.add_edge('b', 'd')
    G.add_edge('c', 't')
    G.add_edge('d', 't')

    G.add_edge('s', 'l1')
    G.add_edge('l1', 'l2')
    G.add_edge('l2', 'l3')
    G.add_edge('l3', 'l4')
    G.add_edge('l4', 'l5')
    G.add_edge('l5', 'l6')
    G.add_edge('l6', 't')

    return G

    # Define the positions of the nodes for plotting
    pos = {'s': (0, 0), 'a': (1, -1), 'b': (1, 1), 'c': (2, 1), 'd': (2, -1), 't': (3, 0)}

def update_lambda(players, step_size):
    
    for player in players.values():

        # Calculate expected gas consumption
        expected_consumption = sum(x * y for x, y in zip(player.strategy, player.path_lengths))
       
        # Calculate expected constraint violation
        constraint_function = expected_consumption - player.gas

        # Gradient ascent on lambda

        player.l = max(0,player.l + step_size * constraint_function)
        #print(f"{player.name}'s lambda = ", player.l)

def calculate_nash_gap(G, players):
    total_cong_dict = total_expected_load(G,players)
    gap = 0
    for player in players.values():
        gap += player_nash_gap(G, player, total_cong_dict)
    return gap

def player_nash_gap(G, player, exp_visitation):
    self_load = single_expected_load(player.G, player.strategy, player.paths)
    rest_load = {key: exp_visitation[key] - self_load[key] for key in exp_visitation.keys()}
     
    # pure_strategy_cost is a vector containing [f(x_1), f(x_2), ..., f(x_5)]
    # each entry is the cost for pure strategy i
    pure_strategy_cost = [None for _ in player.paths]
    for i in range(len(player.paths)):
        cost = 0
        for edge in player.to_edge_list(player.paths[i]):
            cost += 1 + rest_load[edge]
        pure_strategy_cost[i] = cost

    # same but for gas
    pure_strategy_utility = player.path_lengths
    
    # constraints for the linear program: ask rose about this if you're confused
    # if you're confused and you're rose, this is in your black notebook somewhere. good luck
    obj = pure_strategy_cost
    lhs_ineq = [player.path_lengths]
    rhs_ineq = [player.gas]
    lhs_eq = [[1 for _ in player.paths]]
    rhs_eq = [1]
    bnd = [(0, 1) for _ in player.paths]

    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
                  A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
                  method="interior-point")

    if not opt.success:
        # problem!
        print('linear program was non-feasible somehow? exiting')
        sys.exit(-1)

    return opt.fun
        player.l = max(0,player.l + step_size * constraint_function - 2*0.0001*player.l)
       
def gradient_descent(G, players, x_stepsize):

    total_cong_dict = total_expected_load(G,players)
    all_grads = []
    for _,player in players.items():
        all_grads.append(player.primal_gradient(total_cong_dict))

    # Update Primal
    for idx,(_,player) in enumerate(players.items()):

        new_strategy = player.strategy
        #player.kldiv = sum(rel_entr(new_strategy, player.strategy))
        #player.wasser = wasserstein_distance(new_strategy, player.strategy)
        player.strategy = projsplx(player.strategy - x_stepsize * all_grads[idx])

def f(G, players):
    total_cong_dict = total_expected_load(G, players)
    congestion_dict = total_effective_congestion(total_cong_dict)
    cong_sum = 0
    for cong in congestion_dict.values():
        cong_sum += cong
    return cong_sum

def calculate_nash_gap(G, players):
    nash_gaps = {}

    # For each player
    for player_name, player in players.items():

        #print("Current lambdas = ",player.l)

        # Get the current utility for the player
        current_utility = f(G, players) + player.l * (sum(player.strategy * np.array(player.path_lengths)) - player.gas) - 2*0.0001*player.l

        # Define the objective function for the optimization problem
        def objective(x_i):

            # Update player's strategy with x_i
            player.strategy = x_i

            # Calculate utility with new strategy
            new_utility = f(G, players) + player.l * (sum(player.strategy * np.array(player.path_lengths)) - player.gas)- 2*0.0001*player.l
            
            # Return -new_utility for minimization problem
            return -new_utility

        # Define the constraints for the optimization problem
        constraints = ({'type': 'ineq', 'fun': lambda x_i: sum(x_i * np.array(player.path_lengths) - player.gas)},
                       {'type': 'eq', 'fun': lambda x_i: sum(x_i) - 1}) # the strategy should sum up to 1 

        # Define the bounds for the optimization problem
        bounds = [(0, 1)] * len(player.strategy)

        # Define the initial guess for the optimization problem
        x0 = [0.25,0.25,0.25,0.25,0]#player.strategy
        #print("x0 = ",x0)

        # Solve the optimization problem
        #print(player.gas)
        res = minimize(objective, x0, constraints=constraints, bounds=bounds)
        print(res)

        print("Current Utility = ",current_utility)
        print("Solution = ",-res.fun)
        sys.exit(0)
        
        # Calculate the Nash Gap for the player
        nash_gap = -res.fun - current_utility

        # Store the Nash Gap in the dictionary
        nash_gaps[player_name] = nash_gap

    return nash_gaps

class Player:

    def __init__(self, name, G, gas):

        # Name of agent is used for identifying purposes
        self.name = name

        # Graph is saved in self.G
        self.G = G

        # Action space is stored in self.paths
        self.paths = self.find_all_paths(G,'s','t')
        
        # Path length used for constraints
        self.path_lengths = [len(path) - 1 for path in self.paths]

        # Each agens has a certain amount of gas
        self.gas = gas

        # Initial lambda is zero 
        self.l = 0

        # Each agent starts with an arbitrary initial strategy
        self.strategy = [random.random() for _ in range(len(self.paths))]
        sum_strategy = sum(self.strategy)
        self.strategy = np.array([x / sum_strategy for x in self.strategy])
        self.strategy = np.array([1,0,0,0,0])

    def primal_gradient(self,exp_visitation):

        self_load = single_expected_load(self.G, self.strategy, self.paths)        
                
        congestion_dict = total_effective_congestion(exp_visitation)
        cong_sums = 0
        for i in congestion_dict.values():
            cong_sums += i
        print(cong_sums)

        primal_gradient = [self.congestion_of_path(path, congestion_dict) for path in self.paths]
        #primal_gradient_normalized = primal_gradient / np.linalg.norm(primal_gradient)

        constr_gradient = np.array(self.path_lengths)
        #constr_gradient_normalized = constr_gradient / np.linalg.norm(constr_gradient)

        exp_constr = np.dot(constr_gradient,self.strategy)

        if self.gas - exp_constr > 0:
            return np.array(primal_gradient)
        else:
            return np.array(primal_gradient + self.l*constr_gradient)
    
    def find_all_paths(self,dag, source, sink, path=[]):
        # add the source node to the current path
        path = path + [source]
        # if the source node is the same as the sink node, we have found a path
        if source == sink:
            return [path]
        # initialize an empty list to store the paths
        paths = []
        # loop through the children of the current node
        for child in dag[source]:
            # recursively find all paths from the current child to the sink node
            new_paths = self.find_all_paths(dag, child, sink, path)
            # add the new paths to the list of paths
            for new_path in new_paths:
                paths.append(new_path)
        return paths

    def congestion_of_path(self, path, congestion_dict):
        result = 0
        for i in range(len(path) - 1):
            result += congestion_dict[(path[i], path[i+1])]
        return result
