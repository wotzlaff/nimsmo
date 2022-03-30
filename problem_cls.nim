import std/algorithm
import smooth_max

type
  Problem*[K] = ref object
    k*: K
    y: seq[float64]
    lmbda*: float64
    regParam*: float64
    maxAsum*: float64
    smoothingParam*: float64

proc newProblem*[K](
  k: K, y: seq[float64];
  lmbda: float64;
  regParam: float64 = 1e-10;
  maxAsum: float64 = Inf;
): Problem[K] =
  result = Problem[K](k: k, y: y, lmbda: lmbda, regParam: regParam, maxAsum: maxAsum)
  result.k.setActive((0..<result.size).toSeq())

proc objectives*[S](problem: Problem, state: S): (float64, float64) {.inline.} =
  var
    reg = 0.0
    lossPrimal = 0.0
    lossDual = 0.0
  for l in 0..<problem.size:
    reg += state.ka[l] * state.a[l]
    let
      dec = state.ka[l] + state.b + problem.sign(l) * state.c
      ya = problem.y[l] * state.a[l]
    lossPrimal += smoothMax2(1.0 - problem.y[l] * dec, problem.smoothingParam)
    lossDual += dualSmoothMax2(ya, problem.smoothingParam) - ya
  let
    asumTerm = if problem.maxAsum < Inf: problem.maxAsum * state.c else: 0.0
    objPrimal = 0.5 * reg + lossPrimal + asumTerm
    objDual = 0.5 * reg + lossDual
  (objPrimal, objDual)


proc size*(problem: Problem): int {.inline.} = problem.y.len
proc isShrunk*(problem: Problem): bool {.inline.} = problem.k.activeSize < problem.size

proc quad*[S](problem: Problem, state: S, l: int): float64 {.inline.} =
  2.0 * problem.smoothingParam * problem.lmbda

proc grad*[S](problem: Problem, state: S, l: int): float64 {.inline.} =
  state.ka[l] - problem.y[l] + problem.smoothingParam * problem.y[l] * (2.0 * problem.y[l] * state.a[l] - 1.0)

proc upperBound*(problem: Problem, l: int): float64 {.inline.} =
  if problem.y[l] > 0.0: 1.0 else: 0.0

proc lowerBound*(problem: Problem, l: int): float64 {.inline.} =
  if problem.y[l] > 0.0: 0.0 else: -1.0

proc sign*(problem: Problem, l: int): float64 {.inline.} =
  if problem.y[l] > 0.0: 1.0 else: -1.0

proc kernelRow*(p: Problem, i: int): auto = p.k.getRow(i)
proc kernelDiag*(p: Problem, i: int): auto {.inline.} = p.k.diag(i)
proc kernelSetActive*(p: Problem, activeSet: seq[int]) {.inline.} = p.k.setActive(activeSet)
proc kernelRestrictActive*(p: Problem, activeSet: seq[int]) {.inline.} = p.k.restrictActive(activeSet)
