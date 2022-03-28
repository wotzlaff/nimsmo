import std/[algorithm, sugar]

type
  Problem*[K] = ref object
    k: K
    y: seq[float64]
    lmbda*: float64
    regParam*: float64

proc newProblem*[K](k: K, y: seq[float64], lmbda, regParam: float64): Problem[K] =
  Problem[K](k: k, y: y, lmbda: lmbda, regParam: regParam)

proc objectives*[S](problem: Problem, state: S): (float64, float64) {.inline.} =
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


proc size*(problem: Problem): int {.inline.} = problem.y.len
proc isShrunk*(problem: Problem): bool {.inline.} = problem.k.activeSize < problem.size

proc grad*[S](problem: Problem, state: S, l: int): float64 {.inline.} =
  state.ka[l] - problem.y[l]

proc upperBound*(problem: Problem, l: int): float64 {.inline.} =
  if problem.y[l] > 0.0: 1.0 else: 0.0

proc lowerBound*(problem: Problem, l: int): float64 {.inline.} =
  if problem.y[l] > 0.0: 0.0 else: -1.0

proc kernelRow*(problem: Problem, i: int): auto = problem.k.getRow(i)
proc kernelDiag*(problem: Problem, i: int): auto {.inline.} = problem.k.diag(i)


proc shrink*[S](problem: Problem, state: S, shrinkingThreshold: float64) =
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

proc unshrink*[S](problem: Problem, state: S) {.inline.} =
  let n = problem.size
  echo "Reactivate..."
  problem.k.resetActive()
  state.ka.fill(0.0)
  for l in 0..<n:
    let al = state.a[l]
    if al != 0.0:
      let kl = problem.kernelRow(l)
      for r in 0..<n:
        state.ka[r] += al / problem.lmbda * kl[r]
  state.activeSet = (0..<n).toSeq()