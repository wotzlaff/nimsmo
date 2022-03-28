import std/[sugar, sequtils]

type
  KernelRow* = ref object
    data*: seq[float64]

  Kernel*[D] = ref object of RootObj
    data*: D
    activeSet*: seq[int]

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

proc resetActive*[D](k: Kernel[D]) {.inline.} =
  k.activeSet = (0..<k.data.len).toSeq()

proc restrictActive*[D](k: Kernel[D], activeNew: seq[int]) {.inline.} =
  k.activeSet = activeNew

proc activeSet*[D](k: Kernel[D]): seq[int] {.inline.} = k.activeSet

proc activeSize*[D](k: Kernel[D]): int {.inline.} =
  k.activeSet.len

proc getRow*[K](k: K, i: int): KernelRow {.inline.} =
  k.compute(i)
