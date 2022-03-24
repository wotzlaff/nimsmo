import std/[sequtils, strformat, times]

type Result* = object
  a*: seq[float64]
  b*: float64
  steps*: int
  time*: float64
  violation*: float64
  gap*: float64

proc computeDesc(kii, kij, kjj, p, tMax0, tMax1, lmbda, regParam: float64): float64 {.inline.} =
  if p <= 0.0 or tMax1 == 0.0:
    return 0.0
  let
    q = kii + kjj - 2.0 * kij
    t = min(lmbda * p / max(q, regParam), min(tMax0, tMax1))
  return t * (0.5 / lmbda * q * t + p)


proc smo*[K](
  k: K, y: seq[float64],
  lmbda: float64;
  tolViolation: float64 = 1e-4;
  regParam: float64 = 1e-10;
  secondOrder: bool = true;
  verbose: int = 0;
  maxSteps: int = 1_000_000_000;
): Result =
  # initialize
  let
    n = k.size
    t0 = cpuTime()
  var
    a = newSeq[float64](n)
    b = 0.0
    violation = Inf
    ka = newSeq[float64](n)
    g = newSeq[float64](n)
    dUp = newSeq[float64](n)
    dDn = newSeq[float64](n)
    activeSet = (0..<n).toSeq()

  for l in 0..<n:
    if y[l] > 0:
      dUp[l] = 1.0
    else:
      dDn[l] = 1.0

  block mainPart:
    for step in 1..maxSteps:
      # find max violation pair
      var
        gmin = +Inf
        gmax = -Inf
        i0 = -1
        j1 = -1
      for l in activeSet:
        let gl = ka[l] - y[l]
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
          ki0 = k[i0]
          kj1 = k[j1]
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
                kll = k.diag(l)
                pi0l = gi0 - gl
                pj1l = gl - gj1
              if dUp[l] > 0.0 and pi0l > 0.0:
                let di0l = computeDesc(ki0i0, ki0[l], kll, pi0l, ti0Max, dUp[l], lmbda, regParam)
                if di0l > dmax0:
                  j0 = l
                  dmax0 = di0l
              if dDn[l] > 0.0 and pj1l > 0.0:
                let dj1l = computeDesc(kj1j1, kj1[l], kll, pj1l, tj1Max, dDn[l], lmbda, regParam)
                if dj1l > dmax1:
                  i1 = l
                  dmax1 = dj1l
            if dmax0 > dmax1: (i0, j0, ki0, k[j0]) else: (i1, j1, k[i1], kj1)
      )
      
      violation = g[i0] - g[j1]
      b = -0.5 * (g[i0] + g[j1])
      
      let optimal = violation < tolViolation
      # print progress
      if verbose > 0 and (step mod verbose == 0 or optimal):
        var
          reg = 0.0
          lossPrimal = 0.0
          lossDual = 0.0
        for l in activeSet:
          reg += ka[l] * a[l]
          lossPrimal += max(0.0, 1.0 - y[l] * (ka[l] + b))
          lossDual -= y[l] * a[l]
        let
          objPrimal = 0.5 * reg + lossPrimal
          objDual = 0.5 * reg + lossDual
          gap = objPrimal + objDual
          dt = cpuTime() - t0
        echo fmt"{step:10d} {dt:10.2f} {violation:10.6f} {gap:10.6f} {objPrimal:10f} {-objDual:10f}"

      # check convergence
      if optimal:
        let dt = cpuTime() - t0
        result.steps = step
        result.time = dt
        result.violation = violation
        echo fmt"done in {step} steps and {dt:.2f} seconds"
        break mainPart
      
      # find optimal step size
      let
        pij = g[i] - g[j]
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
      let tijL = tij / lmbda
      for l in 0..<n:
        ka[l] += tijL * (kj[l] - ki[l])

    let dt = cpuTime() - t0
    result.steps = maxSteps
    result.time = dt
    result.violation = violation
  result.a = a
  result.b = b
