import std/[algorithm, sequtils, sugar]
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
    shift*: float64

proc newProblem*[K](
  k: K, y: seq[float64], lmbda: float64;
  w: seq[float64] = @[];
  regParam: float64 = 1e-10;
  maxAsum: float64 = Inf;
  shift: float64 = 1.0;
): Problem[K] =
  result = Problem[K](
    k: k, y: y,
    w: if w.len > 0: w else: repeat(1.0, y.len),
    lmbda: lmbda,
    regParam: regParam,
    maxAsum: maxAsum,
    shift: shift,
  )
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
    lossPrimal += problem.w[l] * smoothMax2(problem.shift - problem.y[l] * dec, problem.smoothingParam)
    lossDual += problem.w[l] * dualSmoothMax2(ya / problem.w[l], problem.smoothingParam) - problem.shift * ya
  let
    asumTerm = if problem.maxAsum < Inf: problem.maxAsum * state.c else: 0.0
    objPrimal = 0.5 * reg + lossPrimal + asumTerm
    objDual = 0.5 * reg + lossDual
  (objPrimal, objDual)


proc size*(problem: Problem): int {.inline.} = problem.y.len
proc isShrunk*(problem: Problem): bool {.inline.} = problem.k.activeSize < problem.size

proc quad*[S](problem: Problem, state: S, l: int): float64 {.inline.} =
  2.0 * problem.smoothingParam * problem.lmbda / problem.w[l]

proc grad*[S](problem: Problem, state: S, l: int): float64 {.inline.} =
  (
    state.ka[l] - problem.shift * problem.y[l] +
    problem.smoothingParam * problem.y[l] * (2.0 * problem.y[l] * state.a[l] / problem.w[l] - 1.0)
  )

proc upperBound*(problem: Problem, l: int): float64 {.inline.} =
  if problem.y[l] > 0.0: problem.w[l] else: 0.0

proc lowerBound*(problem: Problem, l: int): float64 {.inline.} =
  if problem.y[l] > 0.0: 0.0 else: -problem.w[l]

proc sign*(problem: Problem, l: int): float64 {.inline.} =
  if problem.y[l] > 0.0: 1.0 else: -1.0

proc kernelRow*(p: Problem, i: int): auto = p.k.getRow(i)
proc kernelDiag*(p: Problem, i: int): auto {.inline.} = p.k.diag(i)
proc kernelSetActive*(p: Problem, activeSet: seq[
    int]) {.inline.} = p.k.setActive(activeSet)
proc kernelRestrictActive*(p: Problem, activeSet: seq[
    int]) {.inline.} = p.k.restrictActive(activeSet)

proc supportIndex*[S](p: Problem, s: S; tol: float64 = 1e-6): seq[int] =
  collect:
    for l in 0..<p.size:
      if s.a[l] > tol:
        l

proc dualCoeffs*[S](p: Problem, s: S): seq[float64] =
  s.a
