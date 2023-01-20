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

def total_true_congestion(G, players):
    congestion_dict = dict()
    for edge in G.edges:
        congestion_dict[edge] = 0

    # Calculate total load
    for player in players.values():
        #print(player.path)
        #player.render_path()
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
            ret_dict[edge] = 1*true_cong_dict[edge]
        else: 
            ret_dict[edge] = true_cong_dict[edge]
    return ret_dict

def total_gas_consumption(true_cong_dict):
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


def play_game(G,players):

    total_cong_dict = total_true_congestion(G,players)
    effec_cong_dict = total_effective_congestion(total_cong_dict)
    #pprint.pprint(effec_cong_dict)
    #sys.exit(0)
    total_cons_dict = total_gas_consumption(total_cong_dict)

    # # Calculate congestion for each player
    for name,player in players.items():
        
        alpha = 0.1
        player.congestion = 0
        player.gas = 0
        for edge in player.edges:
            #print(edge)
            #print(effective_congestion(edge, edge_pop_dict[edge]))
            #print(congestion_dict[edge])
            #break
            #print("Printing congestion: ",effective_congestion(edge, congestion_dict))
            player.congestion += effec_cong_dict[edge]#alpha
            player.gas += total_cons_dict[edge] #alpha
        #print(player.name, ":", player.congestion)
        #print(player.gas)
        #print(f"Player {player.name}: Congestion {player.congestion}, Gas {player.gas}")
        #pprint.pprint(total_cons_dict)
        player.update_strategy(total_cong_dict)

def save_animation(str_over_time, name):

    indices = [i for i in range(len(str_over_time[0]))]

    # Define the update function
    def update(i):
        # Clear the previous plots
        plt.cla()
        # Plot the function for the current frame
        plt.bar(indices, str_over_time[i])

    # Create the figure and the FuncAnimation object
    fig = plt.figure()
    ani = anim.FuncAnimation(fig, update, frames=range(len(str_over_time)), interval=150)
    # Save the animation as a video file
    ani.save(str(name)+'.gif', writer='pillow')
    plt.close()

class Player:

    def __init__(self, name, G, rewards, constraints):
        self.name = name
        # Save the graph inside the player class for each player to know the graph they are dealing with
        self.G = G
        # Create all paths, given the graph G that was passed into the function
        self.paths = self.find_all_paths(G,'s','t')
        # # initialize to uniform strategy
        # # this leads to symmetry, which is boring
        # self.strategy = [1 / len(self.paths) for _ in range(len(self.paths))]
        # initialize to random strategy
        self.strategy = [random.random() for _ in range(len(self.paths))]
        sum_strategy = sum(self.strategy)
        self.strategy = [x / sum_strategy for x in self.strategy]

        self.rewards = rewards
        self.constraints = constraints
        # self.constraints[0] is gonna exist because python but we won't use it
        self.constraints.insert(0, None)
        
        assert len(self.rewards) == len(self.constraints)
        # using mu because lambda is a keyword
        # mu[0] is gonna exist because python but we won't use it
        self.mu = [0 for _ in self.rewards]
        self.mu[0] = None

        self.kldiv = 0

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
    
    # what is the reward from a given path, knowing how all players played?
    def reward_from_path(self, path, hypothetical_dict):
        result = 0
        for i in range(len(path) - 1):
            result += hypothetical_dict[(path[i], path[i+1])]
        return result

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
        
    def choose_action(self):
        # choose a random path from the set of available paths
        action_index = np.random.choice(len(self.paths), 1, p=self.strategy)[0]
        self.path = self.paths[action_index]
        self.edges = [(self.path[i], self.path[i+1]) for i in range(len(self.path)-1)]

    def update_paths(self, new_paths):
        # update the set of available paths
        self.paths = new_paths

    # takes a step in the strategy with respect to the gradient of the lagrangian $\mathcal{L}(x, \lambda) = r_0 (x, \lambda) + \sum_i (\mu_i (c_i - r_i (x)))$
    # congestion_dict is a dict from edges to numbers
    def update_primal(self, congestion_dict, step_size=0.01):
        # first, we find the gradient of the congestion ($\nabla_x r_0 (x, \lambda)$)
        
        # because the expected congestion for the player is linear w.r.t. their
        # strategy, we can find the gradient by just finding the congestion the player
        # would get for any particular strategy
        # create a useful new dictionary hypthetical_dict 
        # hypothetical_dict is congestion_dict but for each edge, it's the congestion
        #     of that edge given that the player chooses a path with that edge
        # think of it like this: everybody else has already played but you. If you want
        #     to know how good a path is, add 1 to edge in that path and add the edge
        #     weights. We just add 1 to all the edge weights preemptively here.
       
        reward_gradients = [None for _ in self.rewards]
        reward_gradient_norms = [None for _ in self.rewards]
        for i in range(len(self.rewards)):
            hypothetical_dict = {edge: self.rewards[i][edge](congestion) if edge in self.edges else self.rewards[i][edge](congestion + 1)
                                for (edge, congestion) in congestion_dict.items()}
            reward_gradients[i] = [self.reward_from_path(path, hypothetical_dict) for path in self.paths]
            reward_gradient_norms[i] = math.sqrt(sum([x**2 for x in reward_gradients[i]]))
        
        gradient = [None for _ in self.paths]
        gradient = [reward_gradients[0][j]/reward_gradient_norms[0] for j in range(len(self.paths))]
        for i in range(1, len(self.rewards)):
            print(i)
            gradient = [gradient[j] + self.mu[i] * reward_gradients[i][j]/reward_gradient_norms[i] for j in range(len(self.paths))]

        norm_grad = math.sqrt(sum([x**2 for x in gradient]))

        # project to simplex
        new_strategy = projsplx(np.array([self.strategy[i] - (step_size * gradient[i] / norm_grad) for i in range(len(self.paths))]))

        # For plotting, we will plot the KL divergence of the probability distributions:
        self.kldiv = sum(rel_entr(new_strategy, self.strategy))
        self.wasser = wasserstein_distance(new_strategy, self.strategy)
        self.strategy = new_strategy

    def update_dual(self, congestion_dict, step_size=0.01):
        for i in range(1, len(self.mu)):
            hypothetical_dict = {edge: self.rewards[i][edge](congestion) if edge in self.edges else self.rewards[i][edge](congestion + 1)
                                for (edge, congestion) in congestion_dict.items()}
            expected_reward = sum([self.reward_from_path(self.paths[j], hypothetical_dict) * self.strategy[j] for j in range(len(self.paths))])
            gradient = expected_reward - self.constraints[i]
            self.mu[i] += gradient * step_size
            self.mu[i] = max(0, self.mu[i])

"""
    # takes a step of gradient descent
    # congestion_dict is a dict from edges to numbers
    def update_strategy(self, true_congestion_dict, step_size=0.1):
        # because the expected congestion for the player is linear w.r.t. their
        # strategy, we can find the gradient by just finding the congestion the player
        # would get for any particular strategy
        # create a useful new dictionary hypthetical_dict 
        # hypothetical_dict is congestion_dict but for each edge, it's the congestion
        #     of that edge given that the player chooses a path with that edge
        # think of it like this: everybody else has already played but you. If you want
        #     to know how good a path is, add 1 to edge in that path and add the edge
        #     weights. We just add 1 to all the edge weights preemptively here.

        hypothetical_dict = {edge: congestion if edge in self.to_edge_list(self.path) else congestion + 1
                            for (edge, congestion) in true_congestion_dict.items()}
        hypothetical_effective_dict = total_effective_congestion(hypothetical_dict)
        #print(self.strategy)
        #print("Hypothetical dict of ", self.name)     
        #pprint.pprint(hypothetical_effective_dict)                   
        gradient = [self.congestion_of_path(path, hypothetical_effective_dict) for path in self.paths]
        norm_grad = math.sqrt(sum([x**2 for x in gradient]))
        # project to simplex
        new_strategy = projsplx(np.array([self.strategy[i] - (step_size * gradient[i] / norm_grad) for i in range(len(self.paths))]))
        sum_new_strategy = sum(new_strategy)
        new_strategy = [x / sum_new_strategy for x in new_strategy]
        
        # For plotting, we will plot the KL divergence of the probability distributions:
        self.kldiv = sum(rel_entr(new_strategy, self.strategy))
        self.wasser = wasserstein_distance(new_strategy, self.strategy)
        #print(self.kldiv)
        self.strategy = new_strategy
        #print(self.strategy)
        #sys.exit(0)
"""
