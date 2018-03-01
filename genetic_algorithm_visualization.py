import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# performance function parameters
STANDARD_NORM = 1 / np.sqrt(2 * np.pi)
ROWS = np.arange(0, 8)
COLS = np.arange(0, 8)


# performance function
def f(x, y):
    elements = []
    for row in ROWS:
        for col in COLS:
            sigma = (10 / (row + col + 25))
            elements.append(gaussian(x, y, row, col, sigma))
    return tf.reduce_sum(elements, axis=0)


# hills of performance function
def gaussian(x, y, x_mean, y_mean, sigma):
    normalizing = STANDARD_NORM / sigma
    x_exponent = -0.5 * tf.square((x - x_mean) / sigma)
    y_exponent = -0.5 * tf.square((y - y_mean) / sigma)
    exponent = x_exponent + y_exponent
    return normalizing * tf.exp(exponent)


# visualize performance function
x_steps = np.arange(-1, 9, 0.02)
y_steps = np.arange(-1, 9, 0.02)
x_val, y_val = np.meshgrid(x_steps, y_steps)
performance = f(x_val, y_val)
with tf.Session() as sess:
    z_val = sess.run(performance)
plt.ion()
plt.contour(x_val, y_val, z_val)
plt.axis([-1, 8, -1, 8])
plt.draw()
plt.pause(0.001)


class Organism:

    def __init__(self, x=0, y=0, random=False, scale=1):
        if random:
            self.random(x, y, scale)
        else:
            self.x = x
            self.y = y

    def random(self, x_mean, y_mean, scale):
        self.x = np.random.normal(x_mean, scale=scale)
        self.y = np.random.normal(y_mean, scale=scale)

    def mutate(self, scale):
        self.x = np.random.normal(self.x, scale=scale)
        self.y = np.random.normal(self.y, scale=scale)

    def get_offspring(self, other):
        child = Organism((self.x + other.x) / 2, (self.y + other.y) / 2)
        child.mutate(((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5)
        return child


class Population:

    def __init__(self, pop_size):
        self.pop_size = pop_size
        self.pool = list()
        for _ in range(pop_size):
            self.pool.append(Organism(0, 0, random=True, scale=0.5))

    def _order_by_perf(self):
        xs = np.array([ind.x for ind in self.pool])
        ys = np.array([ind.y for ind in self.pool])
        perf = f(xs, ys).eval()
        self.pool = [x for _, x in sorted(zip(perf, self.pool), key=lambda pair: pair[0])]

    def step(self, mutation_scale):
        end = pop_size // 3
        # sort from lowest perf to greatest perf
        self._order_by_perf()

        # increasing index of child index
        c_i = 0
        # decreasing indices of current progenitors
        for p1_i in reversed(range(0, self.pop_size)):
            for p2_i in reversed(range(end, self.pop_size)):
                self.pool[c_i] = self.pool[p1_i].get_offspring(self.pool[p2_i])
                c_i += 1
                if c_i > end:
                    break
            else:
                continue
            break

        # mutate survivors, except top 3%
        mutation_bound = int(self.pop_size * .97)
        for i in range(end, mutation_bound):
            self.pool[i].mutate(mutation_scale)


# simulation parameters
pop_size = 100
mutation_scale = 0.5
iterations = 1000

sess = tf.InteractiveSession()

recent_scatters = list()
pop = Population(pop_size)
for i in range(iterations):
    print(i)
    pop.step(mutation_scale)

    xs = np.array([ind.x for ind in pop.pool])
    ys = np.array([ind.y for ind in pop.pool])

    if len(recent_scatters) > 1:
        recent_scatters.pop(0).remove()
        recent_scatters[0].set_color("#FE9494")
    recent_scatters.append(plt.scatter(xs, ys, color="#FD0000"))

    plt.pause(0.001)
    plt.savefig("figures/" + str(i) + '.png')

