import std/[random, math, sequtils, stats, strformat, sugar, times]
import lrucache

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

# define kernel function
let xsqr = collect:
  for xi in x:
    var xisqr = 0.0
    for xik in xi:
      xisqr += xik * xik
    xisqr

type KernelRow = ref object
  data: seq[float64]

proc `[]`(r: KernelRow, i: int): float64 =
  r.data[i]

proc rawKernel(i: int): KernelRow =
  let xi = x[i]
  let data = collect(newSeqOfCap(x.len)):
    for j in 0..<x.len:
      let xj = x[j]
      var dsqr = xsqr[i] + xsqr[j]
      for k in 0..<xi.len:
        dsqr -= 2.0 * xi[k] * xj[k]
      exp(-gamma * dsqr)
  KernelRow(data: data)

let cache = newLRUCache[int, KernelRow](200)
var
  accesses = 0
  misses = 0
proc kernel(i: int): KernelRow =
  accesses += 1
  if i notin cache:
    misses += 1
    cache[i] = rawKernel(i)
  cache[i]

# compute diagonal
let kdiag = newSeqWith[float64](n, 1.0)

let yr = collect:
  for xi in x:
    xi.foldl(a + sin(math.PI * b), 0.0)
let ym = yr.mean()
let y = collect:
  for yi in yr:
    if yi > ym:
      +1.0
    else:
      -1.0

# set parameters
let
  tol = 1e-4
  regParam = 1e-10
  secondOrder = true
  verbose = 100

# initialize
var
  a = newSeq[float64](n)
  ka = newSeq[float64](n)
  g = newSeq[float64](n)
  dUp = newSeq[float64](n)
  dDn = newSeq[float64](n)
  activeSet = (0..<n).toSeq()
let t0 = cpuTime()

for l in 0..<n:
  if y[l] > 0:
    dUp[l] = 1.0
  else:
    dDn[l] = 1.0

for step in 1..100000:
  # find max violation pair
  var
    gmin = +Inf
    gmax = -Inf
    i0 = -1
    j1 = -1
  for l in activeSet:
    let gl = ka[l] / lmbda - y[l]
    g[l] = gl

    if dDn[l] > 0.0 and gl > gmax:
      i0 = l
      gmax = gl
    if dUp[l] > 0.0 and gl < gmin:
      j1 = l
      gmin = gl

  # determine working set
  let (i, j, ki, kj) = (
    let
      ki0 = kernel(i0)
      kj1 = kernel(j1)
    if not secondOrder:
      (i0, j1, ki0, kj1)
    else:
      block:
        var
          dmax0 = 0.0
          dmax1 = 0.0
          j0 = -1
          i1 = -1
        let
          gi0 = g[i0]
          gj1 = g[j1]
          ki0i0 = ki0[i0]
          kj1j1 = kj1[j1]
          ti0Max = dDn[i0]
          tj1Max = dUp[j1]
        for l in activeSet:
          let
            gl = g[l]
            kll = kdiag[l]
          if dUp[l] > 0.0:
            let
              qi0l = ki0i0 + kll - 2.0 * ki0[l]
              pi0l = gi0 - gl
            if pi0l > 0.0:
              let
                ti0l = min(lmbda * pi0l / max(qi0l, regParam), min(ti0Max, dUp[l]))
                di0l = ti0l * (0.5 / lmbda * qi0l * ti0l + pi0l)
              if di0l > dmax0:
                j0 = l
                dmax0 = di0l
          if dDn[l] > 0.0:
            let
              qj1l = kj1j1 + kll - 2.0 * kj1[l]
              pj1l = gl - gj1
            if pj1l > 0.0:
              let
                tj1l = min(lmbda * pj1l / max(qj1l, regParam), min(tj1Max, dDn[l]))
                dj1l = tj1l * (0.5 / lmbda * qj1l * tj1l + pj1l)
              if dj1l > dmax1:
                i1 = l
                dmax1 = dj1l
        if dmax0 > dmax1:
          (i0, j0, ki0, kernel(j0))
        else:
          (i1, j1, kernel(i1), kj1)
  )
    
  let
    pij = g[i] - g[j]
    b = -0.5 * (g[i] + g[j])
  
  # print progress
  if verbose > 0 and (step mod verbose == 0 or pij < tol):
    var
      reg = 0.0
      lossPrimal = 0.0
      lossDual = 0.0
    for l in activeSet:
      reg += ka[l] * a[l]
      lossPrimal += max(0.0, 1.0 - y[l] * (ka[l] / lmbda + b))
      lossDual -= y[l] * a[l]
    let
      objPrimal = 0.5 / lmbda * reg + lossPrimal
      objDual = 0.5 / lmbda * reg + lossDual
      gap = objPrimal + objDual
      cacheProps = fmt"{misses:9d} of {accesses:9d} = {misses / accesses * 100:6.1f}%"
      dt = cpuTime() - t0
    echo fmt"{step:10d} {dt:10.2f} {pij:10.6f} {gap:10.6f} {objPrimal:10f} {-objDual:10f} {cacheProps}"

  # check convergence
  if pij < tol:
    let dt = cpuTime() - t0
    echo fmt"done in {step} steps and {dt:.2f} seconds"
    break
  
  # find optimal step size
  let
    qij = ki[i] + kj[j] - 2.0 * ki[j]
    tij = min(
      # unconstrained min
      lmbda * pij / max(qij, regParam),
      min(dDn[i], dUp[j])
    )

  # update
  a[i] -= tij
  dDn[i] -= tij
  dUp[i] += tij
  a[j] += tij
  dDn[j] += tij
  dUp[j] -= tij
  for l in 0..<n:
    ka[l] += tij * (kj[l] - ki[l])
