import std/[algorithm, sequtils, strformat, times, sugar]

type
  State = ref object
    a, g, ka, dUp, dDn: seq[float64]
    b, violation, gap, value: float64
    activeSet: seq[int]

  Problem[K] = ref object
    y: seq[float64]
    k: K
    lmbda: float64
    regParam: float64

  Result* = object
    a*: seq[float64]
    b*: float64
    steps*: int
    time*: float64
    violation*: float64
    gap*: float64
    value*: float64

proc size*[K](problem: Problem[K]): int {.inline.} = problem.y.len
# proc activeSize*[K](problem: Problem[K]): int {.inline.} = problem.k.activeSize
proc isShrunk*[K](problem: Problem[K]): bool {.inline.} = problem.k.activeSize < problem.size

proc grad*[K](problem: Problem[K], state: State, l: int): float64 {.inline.} =
  state.ka[l] - problem.y[l]

proc upperBound*[K](problem: Problem[K], l: int): float64 {.inline.} =
  if problem.y[l] > 0.0: 1.0 else: 0.0

proc lowerBound*[K](problem: Problem[K], l: int): float64 {.inline.} =
  if problem.y[l] > 0.0: 0.0 else: -1.0

proc findMVP[P](problem: P, state: var State): (int, int) {.inline.} =
  var
    gmin = +Inf
    gmax = -Inf
    i0Idx = -1
    j1Idx = -1
  for lIdx, l in state.activeSet:
    let gl = problem.grad(state, l)
    state.g[l] = gl

    if state.dDn[l] > 0.0 and gl > gmax:
      i0Idx = lIdx
      gmax = gl
    if state.dUp[l] > 0.0 and gl < gmin:
      j1Idx = lIdx
      gmin = gl
  let
    i0 = state.activeSet[i0Idx]
    j1 = state.activeSet[j1Idx]
  state.violation = state.g[i0] - state.g[j1]
  state.b = -0.5 * (state.g[i0] + state.g[j1])
  (i0Idx, j1Idx)


proc computeDesc(kii, kij, kjj, p, tMax0, tMax1, lmbda, regParam: float64): float64 {.inline.} =
  if p <= 0.0 or tMax1 == 0.0:
    return 0.0
  let
    q = kii + kjj - 2.0 * kij
    t = min(lmbda * p / max(q, regParam), min(tMax0, tMax1))
  return t * (p - 0.5 / lmbda * q * t)


proc findWS2[P](
  problem: P,
  i0Idx, j1Idx: int,
  state: var State,
): (int, int) {.inline.} =
  let
    i0 = state.activeSet[i0Idx]
    j1 = state.activeSet[j1Idx]
    ki0 = problem.k.getRow(i0)
    kj1 = problem.k.getRow(j1)
  var
    dmax0 = 0.0
    dmax1 = 0.0
    j0Idx = -1
    i1Idx = -1
  let
    gi0 = state.g[i0]
    gj1 = state.g[j1]
    ki0i0 = ki0[i0Idx]
    kj1j1 = kj1[j1Idx]
    ti0Max = state.dDn[i0]
    tj1Max = state.dUp[j1]
  for lIdx, l in state.activeSet:
    let
      gl = state.g[l]
      kll = problem.k.diag(l)
      pi0l = gi0 - gl
      pj1l = gl - gj1
    if state.dUp[l] > 0.0 and pi0l > 0.0:
      let di0l = computeDesc(
        ki0i0, ki0[lIdx], kll,
        pi0l, ti0Max, state.dUp[l],
        problem.lmbda, problem.regParam
      )
      if di0l > dmax0:
        j0Idx = lIdx
        dmax0 = di0l
    if state.dDn[l] > 0.0 and pj1l > 0.0:
      let dj1l = computeDesc(
        kj1j1, kj1[lIdx], kll,
        pj1l, tj1Max, state.dDn[l],
        problem.lmbda, problem.regParam
      )
      if dj1l > dmax1:
        i1Idx = lIdx
        dmax1 = dj1l
  if dmax0 > dmax1:
    (i0Idx, j0Idx)
  else:
    (i1Idx, j1Idx)


proc shrink[K](problem: Problem[K], state: var State, shrinkingThreshold: float64) =
  state.activeSet = collect:
    for l in state.activeSet:
      let
        glb = state.g[l] + state.b
        glbSqr = glb * glb
        fixUp = state.dUp[l] == 0.0 and glb < 0 and glbSqr > shrinkingThreshold * state.violation
        fixDn = state.dDn[l] == 0.0 and glb > 0 and glbSqr > shrinkingThreshold * state.violation
      if not (fixUp or fixDn):
        l
  problem.k.restrictActive(state.activeSet)

proc unshrink[K](problem: Problem[K], state: var State) {.inline.} =
  let n = problem.size
  echo "Reactivate..."
  problem.k.resetActive()
  state.ka.fill(0.0)
  for l in 0..<n:
    let al = state.a[l]
    if al != 0.0:
      let kl = problem.k.getRow(l)
      for r in 0..<n:
        state.ka[r] += al / problem.lmbda * kl[r]
  state.activeSet = (0..<n).toSeq()

proc update[K](problem: Problem[K], iIdx, jIdx: int, state: var State) {.inline.} =
  let
    i = state.activeSet[iIdx]
    j = state.activeSet[jIdx]
    ki = problem.k.getRow(i)
    kj = problem.k.getRow(j)
  # find optimal step size
  let
    pij = state.g[i] - state.g[j]
    qij = ki[iIdx] + kj[jIdx] - 2.0 * ki[jIdx]
    tij = min(
      # unconstrained min
      problem.lmbda * pij / max(qij, problem.regParam),
      min(state.dDn[i], state.dUp[j])
    )
  # update
  state.a[i] -= tij
  state.dDn[i] -= tij
  state.dUp[i] += tij
  state.a[j] += tij
  state.dDn[j] += tij
  state.dUp[j] -= tij
  let tijL = tij / problem.lmbda
  state.value -= tij * (0.5 * qij * tijL - pij)
  for lIdx, l in state.activeSet:
    state.ka[l] += tijL * (kj[lIdx] - ki[lIdx])


proc newState[P](problem: P): State =
  let n = problem.size
  result = State(
    a: newSeq[float64](n),
    ka: newSeq[float64](n),
    g: newSeq[float64](n),
    dUp: newSeq[float64](n),
    dDn: newSeq[float64](n),
  )
  for l in 0..<n:
    result.dUp[l] = problem.upperBound(l)
    result.dDn[l] = -problem.lowerBound(l)
  result.activeSet = (0..<n).toSeq()


proc objectives[K](problem: Problem[K], state: State): (float64, float64) {.inline.} =
  var
    reg = 0.0
    lossPrimal = 0.0
    lossDual = 0.0
  for l in 0..<problem.size:
    reg += state.ka[l] * state.a[l]
    lossPrimal += max(0.0, 1.0 - problem.y[l] * (state.ka[l] + state.b))
    lossDual -= problem.y[l] * state.a[l]
  let
    objPrimal = 0.5 * reg + lossPrimal
    objDual = 0.5 * reg + lossDual
  (objPrimal, objDual)


proc smo*[K](
  k: K, y: seq[float64],
  lmbda: float64;
  tolViolation: float64 = 1e-4;
  regParam: float64 = 1e-10;
  secondOrder: bool = true;
  logObjective: bool = false;
  verbose: int = 0;
  maxSteps: int = 1_000_000_000;
  shrinkingPeriod: int = 1000;
  shrinkingThreshold: float = 1.0;
): Result =
  # initialize
  let t0 = cpuTime()
  let problem = Problem[K](k: k, y: y, lmbda: lmbda, regParam: regParam)
  var state = newState(problem)
  block mainPart:
    for step in 1..maxSteps:
      if shrinkingPeriod > 0 and step mod shrinkingPeriod == 0:
        problem.shrink(state, shrinkingThreshold)

      # find max violation pair
      let (i0Idx, j1Idx) = problem.findMVP(state)
      let optimal = state.violation < tolViolation

      # print progress
      if verbose > 0 and (step mod verbose == 0 or optimal):
        let dt = cpuTime() - t0
        if logObjective:
          let (objPrimal, objDual) = problem.objectives(state)
          state.gap = objPrimal + objDual
          echo fmt"{step:10d} {dt:10.2f} {state.violation:10.6f} {state.gap:10.6f} {objPrimal:10f} {-objDual:10f} {state.value:10f} {state.activeSet.len:8d} of {problem.size:8d}"
        else:
          echo fmt"{step:10d} {dt:10.2f} {state.violation:10.6f} {state.value:10f} {state.activeSet.len:8d} of {problem.size:8d}"

      # check convergence
      if optimal:
        if problem.isShrunk:
          problem.unshrink(state)
          continue
        else:
          result.steps = step
          break mainPart

      # determine working set
      let (iIdx, jIdx) = if not secondOrder:
        (i0Idx, j1Idx)
      else:
        problem.findWS2(i0Idx, j1Idx, state)

      # solve subproblem and update
      problem.update(iIdx, jIdx, state)

    result.steps = maxSteps
  let dt = cpuTime() - t0
  result.time = dt
  result.violation = state.violation
  result.a = state.a
  result.b = state.b
  result.value = state.value
