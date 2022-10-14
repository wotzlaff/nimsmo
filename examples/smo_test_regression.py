import time
import nimsmo
import numpy as np

np.random.seed(42)

n, nft = 500, 10
x = np.random.rand(n, nft)
y = np.sin(np.pi * x).sum(axis=1)
w = np.random.rand(n)

lmbda = 0.1
gamma = 1.5
smoothingParam = 0.01
epsilon = 1e-2

t0 = time.time()
res = nimsmo.solveRegression(
    x.tolist(),
    y.tolist(),
    lmbda,
    gamma,
    w=w.tolist(),
    smoothingParam=smoothingParam,
    epsilon=epsilon,
    verbose=100,
    logObjective=True,
)
t1 = time.time()
print(t1 - t0)

a = np.array(res['a'])
b = res['b']
c = res['c']
idx_sv = a != 0
print(f'N(SV) = {idx_sv.sum()} = {idx_sv.mean() * 100:.1f}%')

def kernel(x1, x2, gamma):
    x1sqr = (x1 * x1).sum(axis=1)
    x2sqr = (x2 * x2).sum(axis=1)
    d = x1sqr[:, None] + x2sqr[None, :] - 2.0 * x1.dot(x2.T)
    return np.exp(-gamma * d)


def smooth_max_p2(t, s):
    return np.piecewise(
        t, [t <= -s, t >= s],
        [0.0, lambda t: t, lambda t: 0.25 / s * (t + s)**2]
    )


def dsmooth_max_p2(t, s):
    return np.piecewise(
        t, [t <= -s, t >= s], [0.0, 1.0, lambda t: 0.5 / s * (t + s)]
    )


def ddsmooth_max_p2(t, s):
    return np.piecewise(t, [t <= -s, t >= s], [0.0, 0.0, 0.5 / s])


x_sv = np.concatenate((x[idx_sv[:n]], x[idx_sv[n:]]))
k = kernel(np.concatenate((x, x)), x_sv, gamma)

ka = k.dot(a[idx_sv]) / lmbda
sign = np.where(np.arange(2*n) < n, 1.0, -1.0)
d = ka + b - sign * c

r = d.copy()
r[:n] -= y
r[n:] -= y
val = smooth_max_p2(-sign * r - epsilon, smoothingParam)
reg = ka.dot(a)
val[:n] *= w
val[n:] *= w
loss = val.sum()
obj = 0.5 * reg + loss
print(f'objective value {obj} = 0.5 * {reg} + {loss}')

g = -sign * dsmooth_max_p2(-sign * r - epsilon, smoothingParam)
g[:n] *= w
g[n:] *= w
r = a + g
print('residual:', np.mean(np.array(r)**2))
