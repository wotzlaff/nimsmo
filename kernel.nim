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

  # CachedKernel[K] = ref Object
    cache: LruCache[int, KernelRow]
    # kernel: K
    accesses: int
    misses: int


proc `[]`*(r: KernelRow, i: int): float64 {.inline.} =
  r.data[i]

proc prepare(k: Kernel) =
  k.xsqr = collect:
    for xi in k.x:
      var xisqr = 0.0
      for xik in xi:
        xisqr += xik * xik
      xisqr

proc size*(k: Kernel): int {.inline.} =
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
    for j in k.activeSet:
      var dsqr = k.xsqr[i] + k.xsqr[j]
      for (xik, xjk) in zip(xi, k.x[j]):
        dsqr -= 2.0 * xik * xjk
      exp(-k.gamma * dsqr)
  KernelRow(data: data)

proc `[]`*(k: Kernel, i: int): KernelRow =
  k.accesses += 1
  if i notin k.cache:
    k.misses += 1
    k.cache[i] = k.compute(i)
  k.cache[i]

proc `activeSet=`*(k: Kernel, activeSet: seq[int]) =
  k.activeSet = activeSet
  # TODO: restrict the available data?
  k.cache.clear()

proc diag*(k: Kernel, i: int): float64 {.inline.} =
  1.0

proc cacheSummary*(k: Kernel): string =
  fmt"{k.misses} of {k.accesses} = {k.misses / k.accesses * 100:.1f}%"
