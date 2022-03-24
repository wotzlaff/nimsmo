import std/[random, math, sequtils, stats, strformat, sugar]

randomize(42)

# define training set
let n = 10
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

proc kernel(i: int): seq[float] =
  let xi = x[i]
  return collect:
    for j in 0..<x.len:
      let xj = x[j]
      var dsqr = xsqr[i] + xsqr[j]
      for k in 0..<xi.len:
        dsqr -= 2.0 * xi[k] * xj[k]
      exp(-gamma * dsqr)

# compute diagonal
let kdiag = collect:
  for xi in x:
    1.0

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
  tol = 1e-6
  regParam = 1e-10
  firstOrder = true
  verbose = 10

# initialize
var
  a = newSeq[float64](n)
  ka = newSeq[float64](n)
  g = newSeq[float64](n)

for step in 1..1000:
  # find max violation pair
  var
    gmin = +Inf
    gmax = -Inf
    i0 = -1
    j1 = -1
  for l in 0..<n:
    let
      yl = y[l]
      al = a[l]
      gl = ka[l] / lmbda - y[l]
    g[l] = gl

    let goDn = (yl > 0 and al > 0.0) or (yl < 0 and al > -1.0)
    let goUp = (yl > 0 and al < 1.0) or (yl < 0 and al < 0.0)
    if goDn and gl > gmax:
      i0 = l
      gmax = gl
    if goUp and gl < gmin:
      j1 = l
      gmin = gl
  let (i, j, ki, kj) = (
    if firstOrder:
      block:
        let
          (i, j) = (i0, j1)
          ki = kernel(i)
          kj = kernel(j)
        (i, j, ki, kj)
    else:
      block:
        let
          (i, j) = (i0, j1)
          ki = kernel(i)
          kj = kernel(j)
        (i, j, ki, kj)
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
    for l in 0..<n:
      reg += ka[l] * a[l]
      lossPrimal += max(0.0, 1.0 - y[l] * (ka[l] / lmbda + b))
      lossDual -= y[l] * a[l]
    let
      objPrimal = 0.5 / lmbda * reg + lossPrimal
      objDual = 0.5 / lmbda * reg + lossDual
      gap = objPrimal + objDual
    echo fmt"{step:10d} {pij:10.6f} {gap:10.6f} {objPrimal:10f} {-objDual:10f}"

  # check convergence
  if pij < tol:
    echo fmt"done in {step} steps"
    break
  
  # find optimal step size
  let
    qij = ki[i] + kj[j] - 2.0 * ki[j]
    tij = min(
      # unconstrained min
      lmbda * pij / max(qij, regParam),
      min(
        # max step for ai
        (if y[i] > 0: 0.0 else: 1.0) + a[i],
        # max step for aj
        (if y[j] > 0: 1.0 else: 0.0) - a[j],
      )
    )

  # update
  a[i] -= tij
  a[j] += tij
  for l in 0..<n:
    ka[l] += tij * (kj[l] - ki[l])
