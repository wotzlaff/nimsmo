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

proc restrict*(row: KernelRow, activeOld, activeNew: seq[int]) =
  var it0 = 0
  row.data = collect:
    for (idx, val) in zip(activeOld, row.data):
      if it0 < activeNew.len and idx == activeNew[it0]:
        it0 += 1
        val

proc resetActive*(k: Kernel) {.inline.} =
  k.activeSet = (0..<k.x.len).toSeq()

proc restrictActive*(k: Kernel, activeNew: seq[int]) {.inline.} =
  k.activeSet = activeNew

proc activeSet*(k: Kernel): seq[int] {.inline.} = k.activeSet

proc activeSize*(k: Kernel): int {.inline.} =
  k.activeSet.len

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
    gamma: gamma
  )
  result.resetActive()
  result.prepare()

proc compute(k: GaussianKernel, i: int): KernelRow =
  let xi = k.x[i]
  let data = collect(newSeqOfCap(k.activeSize)):
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
