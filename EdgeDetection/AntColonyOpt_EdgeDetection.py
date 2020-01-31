import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

from itertools import product


class AntColony:
    def __init__(self, image, tau=0.1, N=10, L=40, K=512, qo=0.2, alpha=1, beta=1, phi=0.05, rho=0.1):
        """

        :param image: image used for edge detection
        :param tau: initial pheromone values
        :param N: number of iterations
        :param L: number of construction steps
        :param K: number of Ants
        :param qo: parameter for controlling the degree of exploration of the ants
        :param alpha: parameter for controlling influence of pheromone trails (fixed to 1 for ACS)
        :param beta: parameter for controlling influence of heuristic information
        :param phi: pheromone decay coefficient
        :param rho: pheromone evaporation coefficient
        """
        # Set Image if RGB convert to Grey scale
        if len(image.shape) == 3:
            self.image = np.dot(image, [0.2989, 0.5870, 0.1140])
        else:
            self.image = image

        self.tau = tau
        self.N = N
        self.L = L
        self.K = K
        self.qo = qo
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.rho = rho
        # Pheromone Matrix
        self.Pheromones = np.ones(self.image.shape) * self.tau
        # heuristic Information Matrix
        self.H = self.construct_heuristic_matrix()
        # Ants
        self.ants = None
        # Delta Pheromone
        self.delta_tau = None
        # Edge Map
        self.edge_image = None

    def run(self):
        """
        for each iteration n=1:N do
            for each construction_step l=1:L do
                for each ant k=1:K do
                    Select and go to the next pixel
                    Update pixel's pheromone (local)
                end
            end
            Update Visited Pixel's Pheromones (global)
        end

        :return:
        """
        self.ants = self.init_ants()
        for iter in range(self.N):
            self.delta_tau = np.zeros(self.image.shape)
            for step in range(self.L - 1):
                for ant in self.ants:
                    self.step_ant(ant)
                    self.local_update(ant[-1], step)
            self.global_update()

            # reset ant paths
            # self.ant = self.init_ants();
            self.ants = [[ant[-1]] for ant in self.ants]

        self.decision()

    def decision(self):
        """

        Pheromone(i,j) > threshold = Pixel(i,j) = 1
        Else Pixel(i,j) = 0
        :return:
        """
        # Use Tau as Threshold
        self.edge_image = (~(self.Pheromones > self.tau)).astype(int)
        plt.imshow(self.edge_image, cmap='gray', interpolation='nearest')
        plt.show()

    def local_update(self, pixel, step):
        """
        Update the Pheromone Value of the pixel with eq.
        And the change in pheromone matrix

        Pheromone(i,j) = (1 - phi) * Pheromone(i,j) + phi * tau

        delta_tau(i,j) = 1 / Lk if K used edge (i,j)
                        0 otherwise

         Where (i,j) = pixel

        :param pixel: pixel to update pheromone level
        :param step: step in ant path
        :return:
        """
        self.Pheromones[pixel] = (1 - self.phi) * self.Pheromones[pixel] + self.phi * self.tau
        self.delta_tau[pixel] += self.H[pixel] / (step + 1)

    def global_update(self):
        """

        Update the Pheromones for pixels that have been visited by at least one ant

        Pheromone(i,j) = (1 - phi) * Pheromone(i,j) + rho * Sum(delta_tau(i,j))
        :return:
        """
        v = np.array(self.ants)
        v = np.reshape(v, (v.shape[0] * v.shape[1], 2))
        visited = list(set([tuple(pixel) for pixel in v]))
        for pixel in visited:
            self.Pheromones[pixel] = (1 - self.phi) * self.Pheromones[pixel] + self.rho * self.delta_tau[pixel]

    def step_ant(self, ant):
        """

        Pseudorandomly select the next pixel in the ant path

        select from the pixel neighbors that the ant has not recently visited

        :param ant:
        :return:
        """
        m, n = self.image.shape

        # get neighbors of current pixel
        curr = ant[-1]
        i_ = [x for x in range(curr[0] - 1, curr[0] + 2) if (0 <= x <= m - 1)]
        j_ = [y for y in range(curr[1] - 1, curr[1] + 2) if (0 <= y <= n - 1)]
        # Remove Pixels that have been visited
        n = [s for s in list(product(i_, j_)) if s not in ant[:8]]

        # Check if There is no valid neighbors
        if n == []:
            ant.append(curr)
            return

        # T(i,j)**alpha * h(i,j)**beta
        num = [pow(self.Pheromones[n[ind]], self.alpha) * pow(self.H[n[ind]], self.beta) for ind in range(len(n))]

        # ACS State Transition Rule
        step = 0
        q = np.random.uniform(0, 1)
        if q <= self.qo:
            # exploitation
            step = np.argmax(num)
        else:
            # exploration
            # Pseudorandom proportional rule
            den = sum(num)
            probs = [x / den for x in num]
            p = 0
            for step, x in enumerate(probs):
                p += x
                if q <= p:
                    break

        ant.append(n[step])

    def init_ants(self):
        """
        Randomly select a starting pixel for each ant
        :return:
        """
        i_ = np.random.randint(self.image.shape[0], size=self.K)
        j_ = np.random.randint(self.image.shape[1], size=self.K)
        return [[start] for start in list(zip(i_, j_))]

    def construct_heuristic_matrix(self):
        def _calculate_intensity_variation(image, i, j):
            """
            :param image: Image Matrix
            :param i: pixel 'x' coordinate
            :param j: pixel 'y' coordinate
            :return: pixel intensity variation
             ________ ________ ________
            |        |        |        |
            |i-1, j-1|  i, j-1|i+1, j-1|
            |________|________|________|
            |        |        |        |
            |i-1, j  |  i, j  |i+1, j  |
            |________|________|________|
            |        |        |        |
            |i-1, j+1|  i, j+1|i+1, j+1|
            |________|________|________|


            Vc(I[ij]) = |I[i-1,j-1] - I[i+1,j+1]| + |I[i-1,j] - I[i+1,j]| + \
                        [I[i-1,j+1] - I[i+1,j-1]| + |I[i,j-1] - I[i,j+1]|

            """
            # check bounds
            if i < 1 or j < 1 or i > image.shape[0] - 2 or j > image.shape[1] - 2:
                return 0

            var = (
                    abs(image[i - 1, j - 1] - image[i + 1, j + 1]) +
                    abs(image[i - 1, j] - image[i + 1, j]) +
                    abs(image[i - 1, j + 1] - image[i + 1, j - 1]) +
                    abs(image[i, j - 1] - image[i, j + 1])
            )
            return var

        m, n = self.image.shape
        i_values = np.array([_calculate_intensity_variation(self.image, i, j)
                             for i, j in list(product(range(m), range(n)))])

        norm_i_values = i_values / np.amax(i_values)
        return np.reshape(norm_i_values, self.image.shape)


if __name__ == "__main__":
    img = img.imread('imgs/lenna.png')

    antColony = AntColony(img, qo=0.2)

    print(antColony.run())
