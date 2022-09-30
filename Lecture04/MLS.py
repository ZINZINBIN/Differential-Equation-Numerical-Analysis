import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Literal

def LinearRegression(xs : np.array, ys : np.array):
    sum_x2 = np.sum(np.power(xs, 2))
    sum_x = np.sum(xs)
    
    sum_xy = np.sum([xs[i] * ys[i] for i in range(0,len(xs))])
    sum_y = np.sum(ys)

    a = sum_xy * len(xs) - sum_x * sum_y
    a /= (sum_x2 * len(xs) - sum_x * sum_x)

    b = -sum_x * sum_xy + sum_x2 * sum_y
    b /= (sum_x2 * len(xs) - sum_x * sum_x)

    return a, b

def generate_uniform_nodes(n_nodes : int, lower : float, upper : float):
    result = np.array([i / n_nodes for i in range(n_nodes)])
    result = lower + (upper - lower) * result
    return result

if __name__ =="__main__":

    xs = np.array([i for i in range(0,8)])
    ys = np.array([1.15, 2.32, 3.32, 4.53, 5.65, 6.97, 8.02, 9.23])

    a,b = LinearRegression(xs, ys)

    print("a : {:.5f}, b : {:.5f}".format(a,b))

    xs_ = generate_uniform_nodes(32, 0, 8)
    ys_ = np.array([a * x + b for x in xs_])

    plt.plot(xs_, ys_, label = 'Linear Regression')
    plt.plot(xs, ys, 'ro', label = 'Data')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.savefig("./linear_regression.png")