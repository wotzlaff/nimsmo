import nimpy
import std/[strformat]

import smo
import problem_cls
import kernel/[gaussian, cache]

proc solve*(
  x, y: PyObject,
  lmbda, gamma: float;
  smoothingParam: float = 0.0,
): Result {.exportpy.} =
  let x: seq[seq[float64]] = x.to(seq[seq[float64]])
  let y: seq[float64] = y.to(seq[float64])

  # define parameters
  let kernel = newGaussianKernel(x, gamma).cache(2000)

  var p = newProblem(kernel, y, lmbda)
  p.smoothingParam = smoothingParam
  # p.epsilon = 0.5
  # p.maxAsum = 0.01 * n.float64
  # let res = smo(p, verbose=1000, shrinkingPeriod=n)
  let res = smo(p, verbose=1000, shrinkingPeriod=1000, logObjective=true)
  echo fmt"It took {res.steps} steps in {res.time:.1f} seconds..."
  echo kernel.cacheSummary()
  res