import std/algorithm

type
  Problem*[K] = ref object
    k*: K
    y: seq[float64]
    lmbda*: float64
    regParam*: float64
    maxAsum*: float64

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
    let dec = state.ka[l] + state.b + problem.sign(l) * state.c
    lossPrimal += max(0.0, 1.0 - problem.y[l] * dec)
    lossDual -= problem.y[l] * state.a[l]
  let
    asumTerm = if problem.maxAsum < Inf: problem.maxAsum * state.c else: 0.0
    objPrimal = 0.5 * reg + lossPrimal + asumTerm
    objDual = 0.5 * reg + lossDual
  (objPrimal, objDual)


proc size*(problem: Problem): int {.inline.} = problem.y.len
proc isShrunk*(problem: Problem): bool {.inline.} = problem.k.activeSize < problem.size

proc grad*[S](problem: Problem, state: S, l: int): float64 {.inline.} =
  state.ka[l] - problem.y[l]

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
