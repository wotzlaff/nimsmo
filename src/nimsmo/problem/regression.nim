import std/[algorithm, sugar, sequtils]
import ../smooth_max

type
  Problem*[K] = ref object
    k*: K
    y*: seq[float64]
    w*: seq[float64]
    lmbda*: float64
    regParam*: float64
    smoothingParam*: float64
    maxAsum*: float64
    epsilon*: float64

proc newProblem*[K](
  k: K, y: seq[float64], lmbda: float64;
  w: seq[float64] = @[];
  regParam: float64 = 1e-10;
  maxAsum: float64 = Inf;
  epsilon: float64 = 1e-6;
): Problem[K] =
  result = Problem[K](
    k: k, y: y,
    w: if w.len > 0: w else: repeat(1.0, y.len),
    lmbda: lmbda,
    regParam: regParam,
    maxAsum: maxAsum,
    epsilon: epsilon,
  )
  let idxs = (0..<y.len).toSeq()
  result.k.setActive(idxs & idxs)

proc objectives*[S](problem: Problem, state: S): (float64, float64) {.inline.} =
  var
    reg = 0.0
    lossPrimal = 0.0
    lossDual = 0.0
  for l in 0..<problem.size:
    reg += state.ka[l] * state.a[l]
    let
      yl = problem.y[l mod problem.y.len]
      wl = problem.w[l mod problem.y.len]
      dec = state.ka[l] + state.b - problem.sign(l) * state.c
      ya = yl * state.a[l]
    lossPrimal += wl * smoothMax2(problem.sign(l) * (dec - yl) - problem.epsilon, problem.smoothingParam)
    lossDual += dualSmoothMax2(state.a[l] * problem.sign(l), problem.smoothingParam) - ya + problem.epsilon * problem.sign(l) * state.a[l]
  let
    asumTerm = if problem.maxAsum < Inf: problem.maxAsum * state.c else: 0.0
    objPrimal = 0.5 * reg + lossPrimal + asumTerm
    objDual = 0.5 * reg + lossDual
  (objPrimal, objDual)


proc size*(problem: Problem): int {.inline.} = 2 * problem.y.len
proc isShrunk*(problem: Problem): bool {.inline.} = problem.k.activeSize < problem.size

proc quad*[S](problem: Problem, state: S, l: int): float64 {.inline.} =
  2.0 * problem.smoothingParam * problem.lmbda

proc grad*[S](problem: Problem, state: S, l: int): float64 {.inline.} =
  let yl = problem.y[l mod problem.y.len]
  state.ka[l] - yl + problem.sign(l) * (problem.epsilon + problem.smoothingParam * (2.0 * problem.sign(l) * state.a[l] - 1.0))

proc upperBound*(problem: Problem, l: int): float64 {.inline.} =
  if l < problem.y.len: problem.w[l] else: 0.0

proc lowerBound*(problem: Problem, l: int): float64 {.inline.} =
  if l < problem.y.len: 0.0 else: -problem.w[l - problem.y.len]

proc sign*(problem: Problem, l: int): float64 {.inline.} =
  if l < problem.y.len: 1.0 else: -1.0

proc kernelRow*(p: Problem, i: int): auto = p.k.getRow(i mod p.y.len)
proc kernelDiag*(p: Problem, i: int): auto {.inline.} = p.k.diag(i mod p.y.len)
proc kernelSetActive*(p: Problem, activeSet: seq[int]) {.inline.} = p.k.setActive(activeSet.map(x => x mod p.y.len))
proc kernelRestrictActive*(p: Problem, activeSet: seq[int]) {.inline.} = p.k.restrictActive(activeSet.map(x => x mod p.y.len))

proc supportIndex*[S](p: Problem, s: S; tol: float64 = 1e-6): seq[int] =
  collect:
    for l in 0..<p.y.len:
      if s.a[l] > tol or -s.a[l+p.y.len] > tol:
        l

proc dualCoeffs*[S](p: Problem, s: S): seq[float64] =
  collect:
    for l in 0..<p.y.len:
      s.a[l] + s.a[l+p.y.len]
