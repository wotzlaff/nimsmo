import std/[sugar, sequtils]

type
  KernelRow* = ref object
    data*: seq[float64]

  Kernel* = ref object of RootObj
    activeSet*: seq[int]

proc getRow*(k: Kernel, i: int): KernelRow =
  assert(false, "Not implemented")

proc diag*(k: Kernel, i: int): float64 =
  assert(false, "Not implemented")

method size*(k: Kernel): int {.base.} =
  assert(false, "Not implemented")

proc `[]`*(r: KernelRow, i: int): float64 {.inline.} =
  r.data[i]

proc restrict*(row: KernelRow, activeOld, activeNew: seq[int]) =
  var it0 = 0
  row.data = collect:
    for (idx, val) in zip(activeOld, row.data):
      if it0 < activeNew.len and idx == activeNew[it0]:
        it0 += 1
        val

proc activeSize*(k: Kernel): int {.inline.} =
  k.activeSet.len
