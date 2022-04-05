import std/[math, sugar, sequtils]
import arraymancer
import base
import ../from_numpy

type
  GaussianKernel* = ref object of Kernel
    data: seq[seq[float64]]
    xsqr: seq[float64]
    gamma: float64

proc prepare(k: GaussianKernel) =
  k.xsqr = collect:
    for xi in k.data:
      var xisqr = 0.0
      for xik in xi:
        xisqr += xik * xik
      xisqr

proc newGaussianKernel*(data: seq[seq[float64]],
    gamma: float64): GaussianKernel =
  result = GaussianKernel(
    data: data,
    gamma: gamma
  )
  result.prepare()

proc getRow*(k: GaussianKernel, i: int): KernelRow =
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

method size*(k: GaussianKernel): int =
  k.data.len


type
  GaussianKernelNumpy* = ref object of Kernel
    data: Tensor[float64]
    xsqr: seq[float64]
    gamma: float64

proc prepare(k: GaussianKernelNumpy) =
  k.xsqr = collect:
    for xi in k.data.axis(0):
      var xisqr = 0.0
      for xik in xi:
        xisqr += xik * xik
      xisqr

proc newGaussianKernel*(data: Tensor[float64],
    gamma: float64): GaussianKernelNumpy =
  result = GaussianKernelNumpy(
    data: data,
    gamma: gamma
  )
  result.prepare()

proc getRow*(k: GaussianKernelNumpy, i: int): KernelRow =
  let xi = k.data[i, _]
  let data = collect(newSeqOfCap(k.activeSize)):
    for j in k.activeSet:
      var dsqr = k.xsqr[i] + k.xsqr[j]
      for (xik, xjk) in zip(xi, k.data[j, _]):
        dsqr -= 2.0 * xik * xjk
      exp(-k.gamma * dsqr)
  KernelRow(data: data)

proc diag*(k: GaussianKernelNumpy, i: int): float64 {.inline.} =
  1.0

method size*(k: GaussianKernelNumpy): int =
  k.data.shape[0]
