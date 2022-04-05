import time
import nimporter
import nimsmo
import numpy as np

np.random.seed(42)

n, nft = 10000, 10
x = np.random.rand(n, nft)
yr = np.sin(np.pi * x).sum(axis=1)
y = np.where(yr > np.median(yr), 1.0, -1.0)

lmbda = 0.01
gamma = 1.5

t0 = time.time()
res = nimsmo.solve(x.tolist(), y.tolist(), lmbda, gamma)
t1 = time.time()
res = nimsmo.solveFromBuffer(x, y, lmbda, gamma)
t2 = time.time()

print(t2 - t1, t1 - t0)
# print(res)
