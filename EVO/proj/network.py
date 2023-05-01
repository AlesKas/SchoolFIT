import copy
import random
import numpy as np

class SortingNetwork():
    comparators = []
    seen = set()
    max_pairs = 0

    def objective(self, comps):
        return len(comps)

    def genPair(self, m, n):
        x, y = random.randint(m, n), random.randint(m, n)
        while True:
            self.seen.add((x, y))
            yield (x, y)
            x, y = random.randint(m, n), random.randint(m, n)
            while (x, y) in self.seen or x == y:
                x, y = random.randint(m, n), random.randint(m, n)

    def __init__(self, arr) -> None:
        self.max_pairs =  5 * len(arr)
        gen = self.genPair(0, len(arr)-1)
        for _ in range(4 * len(arr)):
            self.comparators.append(next(gen))
        self.arr = arr

    def sort(self, comps):
        if comps is None:
            comps = self.comparators
        arr = copy.deepcopy(self.arr)
        for x, y in comps:
            if x > y and arr[x] < arr[y]:
                arr[x], arr[y] = arr[y], arr[x]
            if x < y and self.arr[x] > self.arr[y]:
                arr[x], arr[y] = arr[y], arr[x]
        return arr

    def simulated_annealing(self, n_iterations, temp):
        best = self.comparators
        comps = copy.deepcopy(self.comparators)
        best_eval = self.objective(self.comparators)
        curr, curr_eval = best, best_eval
        scores = []
        for i in range(n_iterations):
            selection = random.randint(0, len(self.arr)-1)
            comps.pop(selection)
            candidate = comps
            candidate_eval = self.objective(comps)
            if candidate_eval < best_eval and self.sort(None) == self.sort(comps):
                best, best_eval = candidate, candidate_eval
                scores.append(best_eval)
                print('new best')
            diff = candidate_eval - curr_eval
            t = temp / float(i + 1)
            metropolis = np.exp(-diff / t)
            if diff < 0 or np.random.rand() < metropolis:
                # store the new current point
                curr, curr_eval = candidate, candidate_eval
        return [best, best_eval, scores]