import arraymancer
import nimpy
import std/[strformat, sugar]

import from_numpy
import smo
import problem_cls
import kernel/[gaussian, cache]

proc solve*(
  x, y: PyObject,
  lmbda, gamma: float;
  smoothingParam: float = 0.0;
  tolViolation: float = 1e-6;
): Result {.exportpy.} =
  let x: seq[seq[float64]] = x.to(seq[seq[float64]])
  let y: seq[float64] = y.to(seq[float64])

  # define parameters
  let kernel = newGaussianKernel(x, gamma).cache(2000)

  var p = newProblem(kernel, y, lmbda)
  p.smoothingParam = smoothingParam
  let res = smo(
    p, verbose = 1000,
    shrinkingPeriod = 1000, logObjective = true,
      tolViolation = tolViolation
    )
  echo fmt"It took {res.steps} steps in {res.time:.1f} seconds..."
  echo kernel.cacheSummary()
  res


proc solveFromBuffer*(
  x, y: PyObject,
  lmbda, gamma: float;
  smoothingParam: float = 0.0;
  tolViolation: float = 1e-6;
): Result {.exportpy.} =
  var
    xNp = fromNumpy[float64](x)
    yNp = fromNumpy[float64](y)

  let
    x = xNp.data
    y = yNp.data

  # define parameters
  let kernel = newGaussianKernel(x, gamma).cache(2000)

  var p = newProblem(kernel, y.toSeq1D(), lmbda)
  p.smoothingParam = smoothingParam
  let res = smo(
    p, verbose = 1000,
    shrinkingPeriod = 1000, logObjective = true,
    tolViolation = tolViolation
  )
  echo fmt"It took {res.steps} steps in {res.time:.1f} seconds..."
  echo kernel.cacheSummary()

  xNp.release()
  yNp.release()
  res