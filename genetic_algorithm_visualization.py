import struct

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# performance function parameters
STANDARD_NORM = 1 / (2 * np.pi)
ROWS = np.arange(0, 8)
COLS = np.arange(0, 8)
MOAT_ROWS = np.arange(5, 7)
MOAT_COLS = np.arange(5, 7)


# performance function
def f():
    x = tf.placeholder(tf.float64, shape=[None, 1])
    y = tf.placeholder(tf.float64, shape=[None, 1])
    elements = []
    for row in ROWS:
        for col in COLS:
            if row in MOAT_ROWS or col in MOAT_COLS:
                continue
            sigma = (10 / (row + col + 25))
            elements.append(gaussian(x, y, row, col, sigma, sigma))
    return x, y, tf.reduce_sum(elements, axis=0)


# hills of performance function
def gaussian(x_var, y_var, x_mean, y_mean, x_sigma, y_sigma):
    normalizing = 1 / (2 * np.pi * x_sigma * y_sigma)
    x_exponent = -1 * tf.square(x_var - x_mean) / (2 * tf.square(x_sigma))
    y_exponent = -1 * tf.square(y_var - y_mean) / (2 * tf.square(y_sigma))
    exponent = x_exponent + y_exponent
    return normalizing * tf.exp(exponent)


# visualize performance function
x_steps = y_steps = np.arange(-1, 9, 0.02)
x_val, y_val = np.meshgrid(x_steps, y_steps)
x_val_flat, y_val_flat = x_val.reshape([-1, 1]), y_val.reshape([-1, 1])
x, y, perf_func = f()
with tf.Session() as sess:
    z_val = sess.run(perf_func, feed_dict={x: x_val_flat, y: y_val_flat})
plt.ion()
plt.contour(x_val, y_val, z_val.reshape(x_val.shape))
plt.axis([-1, 8, -1, 8])
plt.draw()
plt.pause(0.001)


class Trait:

    def __init__(self, x=0, y=0):
        self.set(float(x), float(y))

    def set(self, x, y):
        self.x = x
        self.y = y
        self.x_b = self.dec_to_bin(x)
        self.y_b = self.dec_to_bin(y)

    def random(self, scale):
        x = np.random.normal(self.x, scale=scale)
        y = np.random.normal(self.y, scale=scale)
        self.set(x, y)

    def dec_to_bin(self, decimal):
        return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', decimal))

    def bin_to_dec(self, binary):
        f = int(str(binary), 2)
        return struct.unpack('f', struct.pack('I', f))[0]

    def mutate(self, scale):
        x = np.random.normal(self.x, scale=scale)
        y = np.random.normal(self.y, scale=scale)
        self.set(x, y)

    def create_inheritance(self, other):
        x_sign = self.x_b[0]
        y_sign = self.y_b[0]

        x_exp_i = np.random.randint(4, 9)
        y_exp_i = np.random.randint(4, 9)
        x_exp = self.x_b[1:x_exp_i] + other.x_b[x_exp_i:9]
        y_exp = self.y_b[1:y_exp_i] + other.y_b[y_exp_i:9]

        x_mnt_i = np.random.randint(9, 32)
        y_mnt_i = np.random.randint(9, 32)
        x_mnt = self.x_b[9:x_mnt_i] + other.x_b[x_mnt_i:32]
        y_mnt = self.y_b[9:y_mnt_i] + other.y_b[y_mnt_i:32]

        x = self.bin_to_dec(x_sign + x_exp + x_mnt)
        y = self.bin_to_dec(y_sign + y_exp + y_mnt)

        child_trait = Trait(x, y)
        return child_trait


class Organism:

    def __init__(self, trait=None, random=False, scale=1):
        if not trait:
            self.trait = Trait()
        else:
            self.trait = trait
        if random:
            self.trait.random(scale)

    def mutate(self, scale):
        self.trait.mutate(scale)

    def get_offspring(self, other):
        child_trait = self.trait.create_inheritance(other.trait)
        return Organism(child_trait)


class Population:

    def __init__(self, pop_size, perf_func, x, y):
        self.pop_size = pop_size
        self.pool = list()
        self.perf_func = perf_func
        self.x = x
        self.y = y

        for _ in range(pop_size):
            self.pool.append(Organism(random=True, scale=0.5))

    def _order_by_perf(self, sess):
        xs, ys = self.get_attrib()
        perf = sess.run(self.perf_func, feed_dict={self.x: xs, self.y: ys})
        self.pool = [x for _, x in sorted(zip(perf, self.pool), key=lambda pair: pair[0])]

    def _regularization_factor(self, xs, ys):
        pass

    def step(self, mutation_scale, sess):
        # TODO - probability of survival
        end = pop_size // 3
        # sort from lowest perf to greatest perf
        self._order_by_perf(sess)

        # increasing index of child index
        c_i = 0
        # decreasing indices of current progenitors
        for p1_i in reversed(range(0, self.pop_size)):
            for p2_i in reversed(range(int(self.pop_size * 0.90), self.pop_size)):
                self.pool[c_i] = self.pool[p1_i].get_offspring(self.pool[p2_i])
                c_i += 1
                if c_i > end:
                    break
            else:
                continue
            break

        # mutate survivors, except select top percentage
        mutation_bound = int(self.pop_size * .95)
        for i in range(end, mutation_bound):
            self.pool[i].mutate(mutation_scale)

    def get_x(self):
        return np.array([[ind.trait.x] for ind in self.pool])

    def get_y(self):
        return np.array([[ind.trait.y] for ind in self.pool])

    def get_attrib(self):
        return self.get_x(), self.get_y()


# simulation parameters
pop_size = 100
mutation_scale = 1.0
iterations = 1000

with tf.Session() as sess:
    recent_scatters = list()
    pop = Population(pop_size, perf_func, x, y)
    for i in range(iterations):
        print(i)
        pop.step(mutation_scale, sess)

        xs, ys = pop.get_attrib()

        if len(recent_scatters) > 1:
            recent_scatters.pop(0).remove()
            recent_scatters[0].set_color("#FE9494")
        recent_scatters.append(plt.scatter(xs, ys, color="#FD0000"))

        plt.pause(0.001)
        plt.savefig("figures/" + str(i) + '.png')

