import std/[algorithm, sequtils, sugar]

proc shrink*[P, S](problem: P, state: S, shrinkingThreshold: float64) =
  state.activeSet = collect:
    for l in state.activeSet:
      let
        glb = state.g[l] + state.b + state.c * problem.sign(l)
        glbSqr = glb * glb
        canShrink = glbSqr > shrinkingThreshold * state.violation
        fixUp = state.dUp[l] == 0.0 and glb < 0 and canShrink
        fixDn = state.dDn[l] == 0.0 and glb > 0 and canShrink
      if not (fixUp or fixDn):
        l
  problem.kernelRestrictActive(state.activeSet)

proc unshrink*[P, S](problem: P, state: S) {.inline.} =
  let n = problem.size
  echo "Reactivate..."
  problem.kernelSetActive((0..<n).toSeq())
  state.ka.fill(0.0)
  for l in 0..<n:
    let al = state.a[l]
    if al != 0.0:
      let kl = problem.kernelRow(l)
      for r in 0..<n:
        state.ka[r] += al / problem.lmbda * kl[r]
  state.activeSet = (0..<n).toSeq()
