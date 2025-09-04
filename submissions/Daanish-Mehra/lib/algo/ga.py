import numpy as np
import random

class EnhancedGAEngine:
    def __init__(self, problem):
        self.problem = problem
        self.size = 50
        self.pop = self.make_pop()
    
    def make_pop(self):
        if self.problem["type"] == "bitstring":
            return np.random.randint(0, 2, (self.size, self.problem["length"]))
        elif self.problem["type"] == "permutation":
            result = []
            for i in range(self.size):
                perm = list(range(self.problem["length"]))
                random.shuffle(perm)
                result.append(perm)
            return np.array(result)
        elif self.problem["type"] == "expression":
            return np.random.rand(self.size, 5)
        else:
            return np.random.rand(self.size, self.problem["length"])
    
    def get_fitness(self, pop):
        if self.problem["type"] == "bitstring":
            return self.bit_fitness(pop)
        elif self.problem["type"] == "permutation":
            return self.perm_fitness(pop)
        elif self.problem["type"] == "expression":
            return self.expr_fitness(pop)
        else:
            return np.zeros(self.size)
    
    def bit_fitness(self, pop):
        name = self.problem["name"].lower()
        if "onemax" in name:
            return np.sum(pop, axis=1)
        elif "leadingones" in name:
            result = []
            for person in pop:
                count = 0
                for bit in person:
                    if bit == 1:
                        count += 1
                    else:
                        break
                result.append(count)
            return np.array(result)
        else:
            return np.sum(pop, axis=1)
    
    def perm_fitness(self, pop):
        name = self.problem["name"].lower()
        if "queens" in name:
            result = []
            for p in pop:
                result.append(self.queens_fit(p))
            return np.array(result)
        elif "tsp" in name:
            result = []
            for p in pop:
                result.append(self.tsp_fit(p))
            return np.array(result)
        else:
            result = []
            for p in pop:
                result.append(len(np.unique(p)))
            return np.array(result)
    
    def expr_fitness(self, pop):
        result = []
        for person in pop:
            mse = 0
            for x, y_true in self.problem["dataset"]:
                y_pred = person[0] * x + person[1]
                mse += (y_pred - y_true) ** 2
            result.append(1.0 / (1.0 + mse))
        return np.array(result)
    
    def queens_fit(self, perm):
        n = len(perm)
        bad = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(i - j) == abs(perm[i] - perm[j]):
                    bad += 1
        return n * (n - 1) // 2 - bad
    
    def tsp_fit(self, perm):
        n = len(perm)
        if n <= 1:
            return 0
        cities = np.random.rand(n, 2)
        dist = 0
        for i in range(n):
            p1 = cities[perm[i]]
            p2 = cities[perm[(i + 1) % n]]
            dist += np.linalg.norm(p1 - p2)
        return 1.0 / (1.0 + dist)
    
    def next_gen(self, pop, llm_help=None):
        new_pop = pop.copy()
        if llm_help is not None and len(llm_help) > 0:
            fit = self.get_fitness(pop)
            bad_idx = np.argsort(fit)[:len(llm_help)]
            for i, help in enumerate(llm_help):
                if i < len(bad_idx):
                    new_pop[bad_idx[i]] = help
        for i in range(len(new_pop)):
            if random.random() < 0.7:
                new_pop[i] = self.mutate(new_pop[i])
            if random.random() < 0.7:
                j = random.randint(0, len(new_pop) - 1)
                new_pop[i], new_pop[j] = self.cross(new_pop[i], new_pop[j])
        return new_pop
    
    def mutate(self, person):
        if self.problem["type"] == "bitstring":
            for _ in range(5):
                idx = random.randint(0, len(person) - 1)
                person[idx] = 1 - person[idx]
        elif self.problem["type"] == "permutation":
            i, j = random.sample(range(len(person)), 2)
            person[i], person[j] = person[j], person[i]
        else:
            person += np.random.normal(0, 0.1, len(person))
        return person
    
    def cross(self, p1, p2):
        if self.problem["type"] == "bitstring":
            point = random.randint(1, len(p1) - 1)
            new1 = np.concatenate([p1[:point], p2[point:]])
            new2 = np.concatenate([p2[:point], p1[point:]])
        elif self.problem["type"] == "permutation":
            new1, new2 = self.order_cross(p1, p2)
        else:
            alpha = random.random()
            new1 = alpha * p1 + (1 - alpha) * p2
            new2 = alpha * p2 + (1 - alpha) * p1
        return new1, new2
    
    def order_cross(self, p1, p2):
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        child1 = [-1] * size
        child1[start:end] = p1[start:end]
        left = [x for x in p2 if x not in child1[start:end]]
        j = 0
        for i in range(size):
            if child1[i] == -1:
                child1[i] = left[j]
                j += 1
        child2 = [-1] * size
        child2[start:end] = p2[start:end]
        left = [x for x in p1 if x not in child2[start:end]]
        j = 0
        for i in range(size):
            if child2[i] == -1:
                child2[i] = left[j]
                j += 1
        return np.array(child1), np.array(child2)
