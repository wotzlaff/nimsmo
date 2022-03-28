import std/[math, sugar, sequtils]
import base
export base

type
  GaussianKernel* = ref object of Kernel[seq[seq[float64]]]
    xsqr: seq[float64]
    gamma: float64

proc prepare(k: GaussianKernel) =
  k.xsqr = collect:
    for xi in k.data:
      var xisqr = 0.0
      for xik in xi:
        xisqr += xik * xik
      xisqr

proc newGaussianKernel*(data: seq[seq[float64]], gamma: float64): GaussianKernel =
  result = GaussianKernel(
    data: data,
    gamma: gamma
  )
  result.resetActive()
  result.prepare()

proc compute*(k: GaussianKernel, i: int): KernelRow =
  let xi = k.data[i]
  let data = collect(newSeqOfCap(k.activeSize)):
    for j in k.activeSet:
      var dsqr = k.xsqr[i] + k.xsqr[j]
      for (xik, xjk) in zip(xi, k.data[j]):
        dsqr -= 2.0 * xik * xjk
      exp(-k.gamma * dsqr)
  KernelRow(data: data)

proc diag*(k: GaussianKernel, i: int): float64 {.inline.} =
  1.0

proc getDiag*(k: GaussianKernel, i: int): float64 {.inline.} =
  1.0
