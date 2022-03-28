import std/[math, sugar, sequtils]

type
  Data = seq[seq[float64]]

  KernelRow* = ref object
    data: seq[float64]

  Kernel* = ref object of RootObj
    x: Data
    activeSet: seq[int]

  GaussianKernel* = ref object of Kernel
    xsqr: seq[float64]
    gamma: float64


proc `[]`*(r: KernelRow, i: int): float64 {.inline.} =
  r.data[i]

proc size*(k: Kernel): int {.inline.} =
  k.activeSet.len

proc `activeSet=`*(k: Kernel, activeSet: seq[int]) =
  k.activeSet = activeSet

proc prepare(k: GaussianKernel) =
  k.xsqr = collect:
    for xi in k.x:
      var xisqr = 0.0
      for xik in xi:
        xisqr += xik * xik
      xisqr

proc newGaussianKernel*(x: Data, gamma: float64): GaussianKernel =
  result = GaussianKernel(
    x: x,
    activeSet: (0..<x.len).toSeq(),
    gamma: gamma
  )
  result.prepare()

proc compute(k: GaussianKernel, i: int): KernelRow =
  let xi = k.x[i]
  let data = collect(newSeqOfCap(k.size)):
    for j in k.activeSet:
      var dsqr = k.xsqr[i] + k.xsqr[j]
      for (xik, xjk) in zip(xi, k.x[j]):
        dsqr -= 2.0 * xik * xjk
      exp(-k.gamma * dsqr)
  KernelRow(data: data)

proc diag*(k: GaussianKernel, i: int): float64 {.inline.} =
  1.0

proc getRow*[K](k: K, i: int): KernelRow {.inline.} =
  k.compute(i)
