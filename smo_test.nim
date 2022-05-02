import std/[random, math, sequtils, stats, strformat, sugar, times]
import smo
import problem/classification
import kernel/[gaussian, cache]

when isMainModule:
  randomize(42)

  # define training set
  let n = 10000
  let nft = 5
  let x = collect:
    for i in 0..<n:
      collect:
        for j in 0..<nft:
          rand(1.0)

  # define parameters
  let gamma = 1.5
  let lmbda = 0.01
  let kernel = newGaussianKernel(x, gamma).cache(2000)

  let yr = collect:
    for xi in x:
      xi.foldl(a + sin(math.PI * b), 0.0)
  let ym = yr.mean()
  let y = collect:
    for yi in yr:
      if yi > ym: +1.0 else: -1.0
  var p = newProblem(kernel, y, lmbda)
  p.maxAsum = 0.01 * n.float64
  p.smoothingParam = 4.0 * ln(2.0)
  p.shift = 0.0
  let res = smo(p, verbose=1000, shrinkingPeriod=n, logObjective=true)
  echo fmt"It took {res.steps} steps in {res.time:.1f} seconds..."
  echo kernel.cacheSummary()
  var
    asum = 0.0
    acnt = 0
  for ai in res.a:
    asum += ai.abs()
    if ai.abs() > 1e-6:
      acnt += 1

  echo "asum = ", asum
  echo "acnt = ", acnt
  echo "c = ", res.c
  # echo "eps = ", p.epsilon + res.c
