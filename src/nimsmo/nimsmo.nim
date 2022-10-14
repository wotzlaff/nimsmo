import nimpy
import std/[strformat, sugar, sequtils]

import smo
import problem/[classification, regression]
import kernel/[gaussian, cache]

let py = pyBuiltinsModule()

proc solveClassification*(
  x, y: PyObject,
  lmbda, gamma: float;
  w: PyObject = py.None;
  shift: float = 1.0;
  smoothingParam: float = 0.0;
  maxAsum: float = Inf;
  tol: float = 1e-6;
  verbose: int = 0;
  shrinkingPeriod: int = 0;
  maxSteps: int = 1_000_000_000;
  cacheSize: int = 10_000;
  logObjective: bool = false;
  timeLimit: float = 0.0;
  callback: proc(res: Result): bool = nil;
  callbackPeriod: int = 1;
): Result {.exportpy.} =
  let x: seq[seq[float64]] = x.to(seq[seq[float64]])
  let y: seq[float64] = y.to(seq[float64])
  let n = x.len
  let w: seq[float64] = if w == py.None: repeat(1.0, n) else: w.to(seq[float64])

  if verbose > 0:
    echo "Starting classification SMO on dataset"
    echo fmt" {n} x {x[0].len}"
    echo "with parameters"
    echo fmt" lmbda           = {lmbda}"
    echo fmt" gamma           = {gamma}"
    echo fmt" shift           = {shift}"
    echo fmt" smoothingParam  = {smoothingParam}"
    echo fmt" maxAsum         = {maxAsum}"
    echo fmt" tol             = {tol}"
    echo fmt" shrinkingPeriod = {shrinkingPeriod}"
    echo fmt" cacheSize       = {cacheSize}"

  # define parameters
  let kernel = newGaussianKernel(x, gamma).cache(cacheSize)

  var p = classification.newProblem(kernel, y, lmbda, w=w)
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
    timeLimit = timeLimit,
    callback = callback,
    callbackPeriod = callbackPeriod,
  )
  if verbose > 0:
    echo fmt"It took {res.steps} steps in {res.time:.1f} seconds..."
    echo kernel.cacheSummary()
  res

proc solveRegression*(
  x, y: PyObject,
  lmbda, gamma: float;
  w: PyObject = py.None;
  epsilon: float = 1e-6;
  smoothingParam: float = 0.0;
  maxAsum: float = Inf;
  tol: float = 1e-6;
  verbose: int = 0;
  shrinkingPeriod: int = 0;
  maxSteps: int = 1_000_000_000;
  cacheSize: int = 10_000;
  logObjective: bool = false;
  timeLimit: float = 0.0;
  callback: proc(res: Result): bool = nil;
  callbackPeriod: int = 1;
): Result {.exportpy.} =
  let x: seq[seq[float64]] = x.to(seq[seq[float64]])
  let y: seq[float64] = y.to(seq[float64])
  let n = x.len
  let w: seq[float64] = if w == py.None: repeat(1.0, n) else: w.to(seq[float64])

  if verbose > 0:
    echo "Starting regression SMO on dataset"
    echo fmt" {n} x {x[0].len}"
    echo "with parameters"
    echo fmt" lmbda           = {lmbda}"
    echo fmt" gamma           = {gamma}"
    echo fmt" epsilon         = {epsilon}"
    echo fmt" smoothingParam  = {smoothingParam}"
    echo fmt" maxAsum         = {maxAsum}"
    echo fmt" tol             = {tol}"
    echo fmt" shrinkingPeriod = {shrinkingPeriod}"
    echo fmt" cacheSize       = {cacheSize}"

  # define parameters
  let kernel = newGaussianKernel(x, gamma).cache(cacheSize)

  var p = regression.newProblem(kernel, y, lmbda, w=w)
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
    timeLimit = timeLimit,
    callback = callback,
    callbackPeriod = callbackPeriod,
  )
  if verbose > 0:
    echo fmt"It took {res.steps} steps in {res.time:.1f} seconds..."
    echo kernel.cacheSummary()
  res
