import std/[sugar, sequtils]

type
  KernelRow* = ref object
    data*: seq[float64]

  Kernel* = ref object of RootObj
    size*: int
    activeSet*: seq[int]

proc getRow*(k: Kernel, i: int): KernelRow =
  assert(false, "Not implemented")

proc diag*(k: Kernel, i: int): float64 =
  assert(false, "Not implemented")

proc `[]`*(r: KernelRow, i: int): float64 {.inline.} =
  r.data[i]

proc double*(r: KernelRow): KernelRow {.inline.} =
  result.data = r.data & r.data

proc restrict*(row: KernelRow, activeOld, activeNew: seq[int]) =
  var it0 = 0
  row.data = collect:
    for (idx, val) in zip(activeOld, row.data):
      if it0 < activeNew.len and idx == activeNew[it0]:
        it0 += 1
        val

proc resetActive*(k: Kernel) {.inline.} =
  k.activeSet = (0..<k.size).toSeq()

proc restrictActive*(k: Kernel, activeNew: seq[int]) {.inline.} =
  k.activeSet = activeNew

proc activeSet*(k: Kernel): seq[int] {.inline.} =
  k.activeSet

proc activeSize*(k: Kernel): int {.inline.} =
  k.activeSet.len
