import copy
import random
import numpy as np

class SortingNetwork():
    comparators = []
    seen = set()

    def objective(self, comps):
        return len(comps)

    def genPair(self, m, n):
        for i in range(m, n):
            for j in range(m, n):
                if i == j:
                    continue
                self.comparators.append((i, j))

    def __init__(self, arr) -> None:
        self.genPair(0, len(arr))
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
            pop = comps.pop(selection)
            candidate = comps
            candidate_eval = self.objective(comps)
            if candidate_eval < best_eval and sorted(self.arr) == self.sort(candidate):
                best, best_eval = candidate, candidate_eval
                scores.append(best_eval)
                print('new best')
            if sorted(self.arr) != self.sort(candidate):
                comps.append(pop)
            diff = candidate_eval - curr_eval
            t = temp / float(i + 1)
            metropolis = np.exp(-diff / t)
            if diff < 0 or np.random.rand() < metropolis:
                # store the new current point
                curr, curr_eval = candidate, candidate_eval
        return [best, best_eval, scores]