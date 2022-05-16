import nimpy
import std/[strformat, sugar]

import smo
import problem/[classification]
import kernel/[gaussian, cache]

proc solve*(
  x, y: PyObject,
  lmbda, gamma: float;
  smoothingParam: float = 0.0;
  tolViolation: float = 1e-6;
  shift: float = 1.0;
  verbose: int = 0;
  shrinkingPeriod: int = 0;
  maxSteps: int = 1_000_000_000;
): Result {.exportpy.} =
  let x: seq[seq[float64]] = x.to(seq[seq[float64]])
  let y: seq[float64] = y.to(seq[float64])
  let n = x.len

  # define parameters
  let kernel = newGaussianKernel(x, gamma).cache(2000)

  var p = newProblem(kernel, y, lmbda)
  p.smoothingParam = smoothingParam
  p.shift = shift
  let res = smo(
    p,
    verbose=verbose,
    shrinkingPeriod=if shrinkingPeriod > 0: shrinkingPeriod else: n,
    # logObjective=true,
    tolViolation=tolViolation,
    maxSteps=maxSteps,
  )
  if verbose > 0:
    echo fmt"It took {res.steps} steps in {res.time:.1f} seconds..."
    echo kernel.cacheSummary()
  res
