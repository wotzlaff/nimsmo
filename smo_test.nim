import std/[random, math, sequtils, stats, strformat, sugar, times]
import smo
import kernel
import cached_kernel

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
  let k = newCachedKernel(newGaussianKernel(x, gamma), 500)

  let yr = collect:
    for xi in x:
      xi.foldl(a + sin(math.PI * b), 0.0)
  let ym = yr.mean()
  let y = collect:
    for yi in yr:
      if yi > ym: +1.0 else: -1.0
  let res = smo(k, y, lmbda, verbose=1000)
  echo fmt"It took {res.steps} steps in {res.time:.1f} seconds..."
  echo k.cacheSummary()
