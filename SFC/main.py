import numpy as np

ADDR_LENGTH = 10
NUM_ADDRESSES = 1024
DIAMETER = 4
COUNTERS_LENGTH = 10

NUM_OF_DATA = 100

class SDM():
    def __init__(self, addr_length, num_addresses, diametr, counter_length) -> None:
        self.addr_length = addr_length
        self.num_addresses = num_addresses
        self.diameter = diametr
        self.counter_length = counter_length

        self.sparse_addresses = np.random.randint(2, size=(self.num_addresses, self.addr_length))
        self.weights = np.zeros((self.num_addresses, self.counter_length), dtype=np.int16)

    def write(self, input_data):
        # Calculate hamming distance, vectors are different on place i when x_i != w_i
        # https://www.fit.vutbr.cz/~grebenic/Publikace/mosis2000.pdf
        distance = np.logical_xor(input_data, self.sparse_addresses).sum(axis=1)
        # Get indexes of weights whose distance is lower than on equal to diameter
        indexes = np.where(distance <= self.diameter)
        # v_jk = v_jk + (2d_pj - 1)
        self.weights[indexes] += 2 * input_data - 1

    def read(self, data):
        # Calculate distance again
        distance = np.logical_xor(data, self.sparse_addresses).sum(axis=1)
        # Get indexes
        indexes = np.where(distance <= self.diameter)
        # Sum all neurons
        sum_neurons = self.weights[indexes].sum(axis=0)
        # Calculate threshold, sum all weights and devide the, by 2
        threshold = self.weights.sum(axis=0)
        threshold = threshold / 2
        threshold = threshold.astype(np.int16)
        # Output[i] = (sum_neurons[i] >= threshold[i]) ? 1 : 0
        output = (sum_neurons >= threshold).astype(np.int8)
        return output

sdm = SDM(ADDR_LENGTH, NUM_ADDRESSES, DIAMETER, COUNTERS_LENGTH)
data = np.random.randint(2, size=(NUM_OF_DATA, ADDR_LENGTH), dtype=np.int8)

for dato in data:
    sdm.write(dato)

error = 0
for dato in data:
    error += np.mean(dato != sdm.read(dato)) / NUM_OF_DATA
print(f'Reconstruction error: {100*error:.2f}%')