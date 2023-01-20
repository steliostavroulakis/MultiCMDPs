import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
import math
import matplotlib.animation as anim
import pprint
import sys

def save_animation(str_over_time, name):

    indices = [i for i in range(len(str_over_time[0]))]

    # Define the update function
    def update(i):
        # Clear the previous plots
        plt.cla()
        # Plot the function for the current frame
        plt.bar(indices, str_over_time[i])

    # Create the figure and the FuncAnimation object
    fig = plt.figure(dpi=50)
    ani = anim.FuncAnimation(fig, update, frames=range(len(str_over_time)), interval=150)
    # Save the animation as a video file
    ani.save(str(name)+'.gif', writer='pillow')
    plt.close()

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

def calc_true_congestion(G, players):
    congestion_dict = dict()
    for edge in G.edges:
        congestion_dict[edge] = 0

    # Calculate total load
    for player in players.values():
        for edge in player.edges:
            congestion_dict[edge] +=1
    return congestion_dict
   
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
            ret_dict[edge] = 0.1*true_cong_dict[edge]
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

def print_dag(G):

    # Define the positions of the nodes for plotting
    pos = {'s': (0, 0), 
    'a': (1, -1), 
    'b': (1, 1), 
    'c': (2, 1), 
    'd': (2, -1), 
    't': (3, 0),
    'l1': (1*3/7, -1.5),
    'l2': (2*3/7, -2),
    'l3': (3*3/7, -2),
    'l4': (4*3/7, -2),
    'l5': (5*3/7, -2),
    'l6': (6*3/7, -1.5)}

    nx.draw(G, pos, with_labels=True, node_color='orange', edge_color='blue', node_size=500)#, edgelist=weights.keys(), width=[w for w in weights.values()])
    plt.savefig("base_graph.png")

def primal_update(G,players):

    for _,player in players.items():
        player.choose_action()
        print(f"Player {player.name} selected {player.path_action}")

    total_cong_dict = calc_true_congestion(G,players)

    for _,player in players.items():
        #print("Players selected paths: ",player.path_action)
        player.update_strategy(total_cong_dict)

    #sys.exit(0)

class Player:

    def __init__(self, name, G):

        # Save the graph inside the player class for each player to know the graph they are dealing with
        self.name = name
        self.G = G
        self.paths = self.find_all_paths(G,'s','t')
        self.lamda = 0
        #print(self.paths)
        #self.path = random.sample(self.paths, 1)[0]# = len(self.paths)
        #self.path = ['s', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 't']

        # Uniform Strategy
        # self.strategy = [1 / len(self.paths) for _ in range(len(self.paths))]
        
        # Random strategy
        self.strategy = [random.random() for _ in range(len(self.paths))]
        sum_strategy = sum(self.strategy)
        self.strategy = [x / sum_strategy for x in self.strategy]

    # Takes a step of gradient descent
    # congestion_dict is a dict from edges to numbers
    def update_strategy(self, true_congestion_dict, step_size=0.1, dual_step_size = 1):
        # because the expected congestion for the player is linear w.r.t. their
        # strategy, we can find the gradient by just finding the congestion the player
        # would get for any particular strategy
        # create a useful new dictionary hypthetical_dict 
        # hypothetical_dict is congestion_dict but for each edge, it's the congestion
        #     of that edge given that the player chooses a path with that edge
        # think of it like this: everybody else has already played but you. If you want
        #     to know how good a path is, add 1 to edge in that path and add the edge
        #     weights. We just add 1 to all the edge weights preemptively here.
        #print(self.path_action)

        hypothetical_dict = {edge: congestion if edge in self.to_edge_list(self.path_action) else congestion + 1
                            for (edge, congestion) in true_congestion_dict.items()}
        hypothetical_effective_dict = total_effective_congestion(hypothetical_dict)

        #pprint.pprint(hypothetical_effective_dict)
        # Here we calculate the effective gradient
        primal_gradient = [self.congestion_of_path(path, hypothetical_effective_dict) for path in self.paths]
        norm_grad = math.sqrt(sum([x**2 for x in primal_gradient]))

        print("Primal Gradient: ", primal_gradient)
        #print("Normalized Primal Gradient: ", [num/norm_grad for num in primal_gradient])

        for edge in self.edges:
            true_congestion_dict[edge] -= 1

        dual_gradient = []
        for path in self.paths:
            
            hypothetical_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            for edge_ in hypothetical_edges:
                true_congestion_dict[edge_] += 1
            gas_dict = total_gas_dict(true_congestion_dict)
            #print("Gas Dictinoary:")
            #pprint.pprint(gas_dict)
            total_gas = 0
            for gas in gas_dict.values():
                total_gas += gas

            for edge_ in hypothetical_edges:
                true_congestion_dict[edge_] -= 1

            dual_gradient.append(total_gas)
        dual_norm_grad = math.sqrt(sum([x**2 for x in dual_gradient]))
        print("Dual Gradient: ", dual_gradient)

        #new_strategy = projsplx(np.array([self.strategy[i] - (step_size * primal_gradient[i]/norm_grad) - (self.lambda_ * dual_gradient[i] / dual_norm_grad) for i in range(len(self.paths))]))
        print("Printing old strategy ", self.strategy)
        new_strategy = projsplx(np.array([self.strategy[i] - step_size * primal_gradient[i]/norm_grad + dual_step_size * self.lamda * dual_gradient[i]/dual_norm_grad for i in range(len(self.paths))]))
        #new_strategy = projsplx(np.array([self.strategy[i] - (step_size * primal_gradient[i] / norm_grad) for i in range(len(self.paths))]))

        # For plotting, we will plot the KL divergence and Wasserstein distance of the probability distributions:
        self.kldiv = sum(rel_entr(new_strategy, self.strategy))
        self.wasser = wasserstein_distance(new_strategy, self.strategy)
        #print(self.kldiv)
        self.strategy = new_strategy
        print("Printing new strategy ", self.strategy)
        #sys.exit(0)

        # g(x) is the expected amount of gas you will use based on your currrent strategy self.stragegy * every_gas

        exp_constr_viol = np.dot(self.strategy, dual_gradient)
        print(exp_constr_viol)
        self.lamda = max(0,self.lamda + dual_step_size * (20 - exp_constr_viol))
        #self.lambda_ = min(50,self.lambda_ + dual_step_size * 10)
        #result = [x_i + fraction * y_i for x_i, y_i in zip(x,y)]

        print("New Lambda = ",self.lamda)
        #if self.lamda == 0:
        #    sys.exit(0)
        #sys.exit(0)
        #self.path_action = self.choose_action()
        #self.edges = [(self.path_action[i], self.path_action[i+1]) for i in range(len(self.path_action)-1)]


    def render_path(self):
        self.curr_path = random.choices(self.paths, self.strategy)[0]
        #print(self.curr_path)
        pos = {'s': (0, 0), 
        'a': (1, -1), 
        'b': (1, 1), 
        'c': (2, 1), 
        'd': (2, -1), 
        't': (3, 0),
        'l1': (1*3/7, -1.5),
        'l2': (2*3/7, -2),
        'l3': (3*3/7, -2),
        'l4': (4*3/7, -2),
        'l5': (5*3/7, -2),
        'l6': (6*3/7, -1.5)}
        nx.draw(self.G, pos, with_labels=True, node_color='orange', edge_color='blue', node_size=500)#, edgelist=weights.keys(), width=[w for w in weights.values()])
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.curr_path, node_color='red', node_size=500)
        plt.savefig(self.name+"_path.png")

    # Define a function to turn a path into a list of edges
    def to_edge_list(self,path):
        return [(path[i], path[i+1]) for i in range(len(path) - 1)]

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

    # how congested is a given path, knowing how all players played?
    def congestion_of_path(self, path, congestion_dict):
        result = 0
        for i in range(len(path) - 1):
            result += congestion_dict[(path[i], path[i+1])]
        return result

    def choose_action(self):
        # Choose a random path from the set of available paths
        action_index = np.random.choice(len(self.paths), 1, p=self.strategy)[0]
        action = self.paths[action_index]
        self.path_action = action
        self.edges = [(action[i], action[i+1]) for i in range(len(action)-1)]
        #print(self.path_action)

    def total_gas_of_path(self,path,gas_dict):
        #pprint.pprint(gas_dict)
        #sys.exit(0)
        result = 0
        for edge, gas in gas_dict.items():
            result += gas
        return result

    def dual_update(self, G,players,total_gas_constr):

        true_congestion_dict = 0

        dual_gradient = []
        for path in self.paths:
            hypothetical_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            for edge_ in hypothetical_edges:
                true_congestion_dict[edge_] += 1
            gas_dict = total_gas_dict(true_congestion_dict)
            total_gas = 0
            for gas in gas_dict.values():
                total_gas += gas

            #print(f"Here we have the total gas spent {total_gas} for action {path} selected")

            dual_gradient.append(total_gas)
        dual_norm_grad = math.sqrt(sum([x**2 for x in dual_gradient]))

        #print("Printing Dual Gradient: ", dual_gradient)
        print("Lagrangian Multiplier: ", self.lambda_)
        #sys.exit(0)
        #print(gradient)
        #sys.exit(0)
        print("Printing prev strategy ", self.strategy)

    # def update_lambda(self,true_congestion_dict, total_gas_constr, step_size=0.1):

    #     #print(self.edges)
    #     #pprint.pprint(true_congestion_dict)
    #     for edge in self.edges:
    #         true_congestion_dict[edge] -=1
        
    #     print("Before consideration, reduced congestion dict = ")
    #     pprint.pprint(true_congestion_dict)

    #     gradient = []
    #     for path in self.paths:
    #         hypothetical_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    #         for edge_ in hypothetical_edges:
    #             true_congestion_dict[edge_] += 1
    #         gas_dict = total_gas_dict(true_congestion_dict)
    #         total_gas = 0
    #         for gas in gas_dict.values():
    #             total_gas += gas

    #         print(f"Here we have the total gas spent {total_gas} for action {path} selected")

    #         gradient.append(total_gas)

    #     pprint.pprint(gradient)
    #     self.lambdas = [x + step_size*y for x,y in zip(self.lambdas,gradient)]
    #     self.lambdas = np.maximum(self.lambdas, 0)
    #     print("New Lambda = ", self.lambdas)


    #     #sys.exit(0)

    #     #pprint.pprint(true_congestion_dict)


    #     #hypothetical_dict = {edge: congestion if edge in self.to_edge_list(self.path_action) else congestion + 1
    #     #                    for (edge, congestion) in true_congestion_dict.items()}
    #     #pprint.pprint(hypothetical_dict)
    #     sys.exit(0)
    #     hypothetical_gas_dict = total_gas_consumption(hypothetical_dict)

    #     #pprint.pprint(hypothetical_gas_dict)
    #     #sys.exit(0)
    #     gradient = [self.total_gas_of_path(path, hypothetical_gas_dict) for path in self.paths]
    #     print(gradient)
    #     sys.exit(0)
    #     norm_grad = math.sqrt(sum([x**2 for x in gradient]))

    #     self.lambda_ = self.lambda_ + step_size * (total_gas - total_gas_constr)
    #     print(self.lambda_)
    #     print(self.strategy)
    #     #print([np.exp(self.lambda_*) / sum([self.strategy[i] * np.exp(self.lambda_)]) for i in range(len(self.strategy))])

    #     sys.exit(0)

    #     lambda_ = lambda_ + step_size * (total_gas - total_gas_constr)
    #     policy = policy * np.exp(lambda_ * rewards) / sum(policy * np.exp(lambda_ * rewards))
