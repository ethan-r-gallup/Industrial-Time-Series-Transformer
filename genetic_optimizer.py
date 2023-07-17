from functools import reduce
from operator import add
import random
from builderfuncs import build_transformer
import tensorflow as tf
import json

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Genetic_Algorithm():
    def __init__(self, parameter_space, retain=0.3, random_select=0.1, mutate_prob=0.25):
        self.mutate_prob = mutate_prob
        self.random_select = random_select
        self.retain = retain
        self.parameter_space = parameter_space

    def create_population(self, count):
        population = []
        for _ in range(0, count):
            network = Network(parameter_space=self.parameter_space)
            network.set_random_parameters()
            population.append(network)
        return population
    
    def save_population(self, location, pop):
        with open(f"{location}/parameter_space.json", "w") as f:
            json.dump(self.parameter_space, f)
        pop_dicts = [{key : int(net.network_parameters[key]) if net.network_parameters[key]%1==0 else float(net.network_parameters[key]) for key in net.network_parameters} for net in pop]
        with open(f"{location}/population.json", "w") as f:
            json.dump(pop_dicts, f)

    def load_population(self, location):
        with open(f"{location}/parameter_space.json", "r") as f:
            self.parameter_space = json.load(f)
        with open(f"{location}/population.json", "r") as f:
            pop_dicts = json.load(f)
        pop_dicts = [{key : int(dict[key]) if dict[key]%1.0==0 else float(dict[key]) for key in dict} for dict in pop_dicts]
        population = [Network(parameter_space=self.parameter_space, parameters=dict) for dict in pop_dicts]
        return population
    

    def get_grade(self, population):
        total = reduce(add, (network.get_accuracy() 
        for network in population))
        return float(total) / len(population)

    def breed(self, mother, father):
        children = []
        for _ in range(2):
            child = {}
            for param in self.parameter_space:
                child[param] = random.choice(
                    [mother.network_parameters[param],
                    father.network_parameters[param]]
                )
            network = Network(self.parameter_space)
            network.create_network(child)
            if self.mutate_prob > random.random():
                network = self.mutate(network)
            children.append(network)
        return children

    def mutate(self, network):
        mutation = random.choice(list
        (self.parameter_space.keys()))
        network.network_parameters[mutation] = random.choice(self.parameter_space[mutation])
        return network

    def evolve(self, pop):
        graded = [(network.get_accuracy(), network) for network in pop]
        graded = [x[1] for x in sorted(graded,
        key=lambda x: x[0], reverse=True)]
        retain_length = int(len(graded)*self.retain)
        parents = graded[:retain_length]

        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        while len(children) < desired_length:
            male = random.randint(0, parents_length-1)
            female = random.randint(0, 
            parents_length-1)

            if male != female:
                male = parents[male]
                female = parents[female]

                children_new = self.breed(male,
                 female)

                for child_new in children_new:
                    if len(children) < desired_length:
                        children.append(child_new)

        parents.extend(children)

        return parents
    
from numpy import random
class Network():
    def __init__(self, parameter_space=None, parameters={}):
        self.accuracy = 0.
        self.parameter_space = parameter_space
        self.network_parameters = parameters  
        
    def set_random_parameters(self):
        for parameter in self.parameter_space:
            self.network_parameters[parameter] = random.choice(self.parameter_space[parameter])
    
    def create_network(self, network):
        self.network_parameters = network

    def train(self, name, optimizer, loss_func, r2, train_data, val_data,
              early_stopping_monitor, checkpoint):
       
        model = build_transformer(self.network_parameters, name)
        model.compile(optimizer=optimizer,
            loss=[loss_func],
            metrics=[r2])
        history = model.fit(train_data, epochs=20,
                            verbose=0,
                            validation_data=val_data,
                            callbacks=[early_stopping_monitor, checkpoint])
        self.accuracy = max(history.history['val_r2']) 
    
    def get_accuracy(self):
        return self.accuracy 
        