import numpy as np
import pandas as pd
from scipy import integrate

# area of romb, side = 4, angle = 30

xmin, xmax = -4 * np.sin(np.deg2rad(15)), 4 * np.sin(np.deg2rad(15))  # x domain
ymin, ymax = -4 * np.sin(np.deg2rad(75)), 4 * np.sin(np.deg2rad(75))  # y domain
t = 2.2622  # t-crit

n_random = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
indexes = [*range(1, 11, 1)] + ["M", "D"]
results = pd.DataFrame(columns=n_random, index=indexes)

# get n random points from domains
def random_points(n):
    x_random = (xmax - xmin) * np.random.random_sample(n) + xmin
    y_random = (ymax - ymin) * np.random.random_sample(n) + ymin
    points = np.concatenate((x_random.reshape(n, 1), y_random.reshape(n, 1)), axis=1)
    return points.reshape(n, 2)  # array of n*2

# get number of points inside the figure
def inside_count(points):
    shift = 4 * np.sin(np.deg2rad(75))  # y shift
    coef = np.tan(np.deg2rad(75))  # angle
    inside = 0
    for p in points:
        cur_shift = -shift if p[1] < 0 else shift  # y pos/neg check
        cur_coef = -coef if p[0] * p[1] > 0 else coef  # y and x both pos/neg
        cur_val = cur_shift + cur_coef * p[0]  # function of one side
        inside += int(p[1] < cur_val) if p[1] > 0 else int(p[1] > cur_val)  # above/below side line
    return inside


print("Trust intervals for n random points:")
for n in n_random:
    for r in range(1, 11):
        results.loc[r][n] = (xmax-xmin) * (ymax-ymin) * inside_count(random_points(n))/n  # area
    mean = results[n].mean()
    std = results[n].std()
    results.loc['M'][n] = mean
    results.loc['D'][n] = std
    print(f"{n}:\t{round(mean - np.sqrt(std) * t / np.sqrt(n), 4)} < S < {round(mean + np.sqrt(std) * t / np.sqrt(n), 4)}")

print(results)

# integral

a, b = 0, 3
n = 1000000
x = np.random.uniform(a, b, n)  # random points
f_x = np.power(np.abs(np.abs(-7*x)**4 - 3*x + 11), (1/3))/np.power(np.abs(np.abs(-2*x)**6 - 14*x - 8), (1/4))  # func
print(f"Monte Carlo:\t{np.mean(f_x)*(b-a)}")

res = integrate.quad(lambda x: np.power(np.abs(np.abs(-7*x)**4 - 3*x + 11), (1/3))/np.power(np.abs(np.abs(-2*x)**6 - 14*x - 8), (1/4)), a, b)
print(f"Scipy:\t\t{res[0]}")

# double integral

a, b = 0, 2
c, d = 0, 1
n = 10000000
x, y = np.random.uniform(c, d, n), np.random.uniform(a, b, n)  # random points
f_xy = 7 * np.power(x, 2) * np.power(y, 5)  # func
print(f"Monte Carlo:\t{np.mean(f_xy)*(b-a)*(d-c)}")

res = integrate.dblquad(lambda x, y: 7 * np.power(x, 2) * np.power(y, 5), a, b, c, d)
print(f"Scipy:\t\t{res[0]}")



