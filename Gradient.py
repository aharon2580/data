import numpy as np
from typing import Callable


def find_min_foo(foo: Callable[[np.ndarray], float], n):
    step_size = 1e-2
    point = np.random.randint(-10, 10, size=n).astype('float')
    for i in range(50000):
        # step_size *= 1.01
        if i % 10 == 0:
            print(i, point, foo(point), step_size, )
        point -= step_size * get_gradient(foo, point)
    return point


def get_gradient(foo: Callable[[np.ndarray], float], point: np.ndarray):
    gradient = []
    for idx in range(point.size):
        gradient.append(derivative(foo, point, idx))
    return np.array(gradient)/np.linalg.norm(gradient)


def derivative(foo, x, idx):
    h = 0.00001
    xh = x.copy()
    xh[idx] += h
    return (foo(xh) - foo(x)) / h


def f(x):
    return (x**3).sum()


print(find_min_foo(f, 2))
