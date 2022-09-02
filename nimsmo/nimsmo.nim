import nimpy
import std/[strformat, sugar]

import smo
import problem/[classification, regression]
import kernel/[gaussian, cache]

proc solveClassification*(
  x, y: PyObject,
  lmbda, gamma: float;
  smoothingParam: float = 0.0;
  maxAsum: float = Inf;
  tol: float = 1e-6;
  shift: float = 1.0;
  verbose: int = 0;
  shrinkingPeriod: int = 0;
  maxSteps: int = 1_000_000_000;
  cacheSize: int = 10_000;
  logObjective: bool = false;
): Result {.exportpy.} =
  let x: seq[seq[float64]] = x.to(seq[seq[float64]])
  let y: seq[float64] = y.to(seq[float64])
  let n = x.len

  # define parameters
  let kernel = newGaussianKernel(x, gamma).cache(cacheSize)

  var p = classification.newProblem(kernel, y, lmbda)
  p.maxAsum = maxAsum
  p.smoothingParam = smoothingParam
  p.shift = shift
  let res = smo(
    p,
    verbose = verbose,
    shrinkingPeriod = if shrinkingPeriod > 0: shrinkingPeriod else: n,
    tol = tol,
    maxSteps = maxSteps,
    logObjective = logObjective,
  )
  if verbose > 0:
    echo fmt"It took {res.steps} steps in {res.time:.1f} seconds..."
    echo kernel.cacheSummary()
  res

proc solveRegression*(
  x, y: PyObject,
  lmbda, gamma: float;
  epsilon: float = 1e-6;
  smoothingParam: float = 0.0;
  maxAsum: float = Inf;
  tol: float = 1e-6;
  shift: float = 0.0;
  verbose: int = 0;
  shrinkingPeriod: int = 0;
  maxSteps: int = 1_000_000_000;
  cacheSize: int = 10_000;
  logObjective: bool = false;
): Result {.exportpy.} =
  let x: seq[seq[float64]] = x.to(seq[seq[float64]])
  let y: seq[float64] = y.to(seq[float64])
  let n = x.len

  # define parameters
  let kernel = newGaussianKernel(x, gamma).cache(cacheSize)

  var p = regression.newProblem(kernel, y, lmbda)
  p.epsilon = epsilon
  p.maxAsum = maxAsum
  p.smoothingParam = smoothingParam
  let res = smo(
    p,
    verbose = verbose,
    shrinkingPeriod = if shrinkingPeriod > 0: shrinkingPeriod else: n,
    tol = tol,
    maxSteps = maxSteps,
    logObjective = logObjective,
  )
  if verbose > 0:
    echo fmt"It took {res.steps} steps in {res.time:.1f} seconds..."
    echo kernel.cacheSummary()
  res
