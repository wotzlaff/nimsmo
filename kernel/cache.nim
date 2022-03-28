import lrucache
import std/strformat
import base

type
  CachedKernel*[K] = ref object
    cache: LruCache[int, KernelRow]
    kernel: K
    accesses: int
    misses: int

proc newCachedKernel*[K](kernel: K, capacity: int): CachedKernel[K] =
  CachedKernel[K](
    cache: newLruCache[int, KernelRow](capacity),
    kernel: kernel,
    accesses: 0,
    misses: 0,
  )

proc getRow*[C](k: C, i: int): KernelRow {.inline.} =
  k.accesses += 1
  if i notin k.cache:
    k.misses += 1
    k.cache[i] = k.kernel.compute(i)
  k.cache[i]

proc resetActive*(k: CachedKernel) {.inline.} =
  k.kernel.resetActive()
  k.cache.clear()

proc restrictActive*(k: CachedKernel, activeSet: seq[int]) =
  for (key, row) in k.cache.mitems:
    row.restrict(k.kernel.activeSet, activeSet)
  k.kernel.restrictActive(activeSet)

proc diag*(k: CachedKernel, i: int): float64 {.inline.} =
  k.kernel.getDiag(i)

proc activeSize*(k: CachedKernel): int {.inline.} =
  k.kernel.activeSize

proc cacheSummary*(k: CachedKernel): string =
  fmt"{k.misses} of {k.accesses} = {k.misses / k.accesses * 100:.1f}%"
