
proc smoothMax2*(x, d: float64): float64 =
  if x >= d:
    x
  elif x <= -d:
    0.0
  else:
    0.25 / d * (x + d) * (x + d)

proc dualSmoothMax2*(a, d: float64): float64 =
  d * a * (a - 1.0)