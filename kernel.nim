import lrucache
import std/[math, sugar, strformat, sequtils]

type
  Data = seq[seq[float64]]

  KernelRow = ref object
    data: seq[float64]

  Kernel = ref object
    x: Data
    xsqr: seq[float64]
    gamma: float64
    activeSet: seq[int]
    cache: LruCache[int, KernelRow]
    accesses: int
    misses: int


proc `[]`*(r: KernelRow, i: int): float64 =
  r.data[i]

proc prepare(k: Kernel) =
  k.xsqr = collect:
    for xi in k.x:
      var xisqr = 0.0
      for xik in xi:
        xisqr += xik * xik
      xisqr

proc size*(k: Kernel): int =
  k.activeSet.len

proc newKernel*(x: Data, gamma: float64, cap: int): Kernel =
  result = new(Kernel)
  result.x = x
  result.activeSet = (0..<x.len).toSeq()
  result.gamma = gamma
  result.cache = newLRUCache[int, KernelRow](cap)
  result.prepare()

proc compute(k: Kernel, i: int): KernelRow =
  let xi = k.x[i]
  let data = collect(newSeqOfCap(k.size)):
    for j in 0..<k.size:
      let xj = k.x[j]
      var dsqr = k.xsqr[i] + k.xsqr[j]
      for (xik, xjk) in zip(xi, xj):
        dsqr -= 2.0 * xik * xjk
      exp(-k.gamma * dsqr)
  KernelRow(data: data)

proc `[]`*(k: Kernel, i: int): KernelRow =
  k.accesses += 1
  if i notin k.cache:
    k.misses += 1
    k.cache[i] = k.compute(i)
  k.cache[i]

proc restrict*(k: Kernel, activeSet: seq[int]) =
  k.activeSet = activeSet

proc diag*(k: Kernel, i: int): float64 =
  1.0

proc cacheSummary*(k: Kernel): string =
  fmt"{k.misses} of {k.accesses} = {k.misses / k.accesses * 100:.1f}%"
