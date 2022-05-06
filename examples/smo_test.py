import time
import nimsmo
import numpy as np

np.random.seed(42)

n, nft = 500, 10
x = np.random.rand(n, nft)
yr = np.sin(np.pi * x).sum(axis=1)
y = np.where(yr > np.median(yr), 1.0, -1.0)

lmbda = 0.01
gamma = 1.5
smoothingParam = np.log(2.0) / 0.25
shift = 0.0

t0 = time.time()
res = nimsmo.solve(
    x.tolist(),
    y.tolist(),
    lmbda,
    gamma,
    smoothingParam=smoothingParam,
    shift=shift
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


x_sv = x[idx_sv]
k = kernel(x, x_sv, gamma)

ka = k.dot(a[idx_sv]) / lmbda
d = ka + b

val = smooth_max_p2(shift - y * d, smoothingParam)
obj = 0.5 * ka.dot(a) + val.sum()
print('objective value:', obj)

g = -y * dsmooth_max_p2(shift - y * d, smoothingParam)
r = a + g
print('residual:', np.mean(np.array(r)**2))

val = np.log(1.0 + np.exp(-y * d))
obj = 0.5 * ka.dot(a) + val.sum()
print('objective value:', obj)

g = -y / (1.0 + np.exp(-y * d)) * np.exp(-y * d)
r = a + g
print('residual:', np.mean(np.array(r)**2))
