import lrucache
import std/strformat
import base

type
  CachedKernel[K] = ref object
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

proc getRow*[K](k: CachedKernel[K], i: int): KernelRow =
  k.accesses += 1
  if i notin k.cache:
    k.misses += 1
    k.cache[i] = k.kernel.getRow(i)
  k.cache[i]

proc resetActive*[K](k: CachedKernel[K]) =
  k.kernel.resetActive()
  k.cache.clear()

proc restrictActive*[K](k: CachedKernel[K], activeSet: seq[int]) =
  for (key, row) in k.cache.mitems:
    row.restrict(k.kernel.activeSet, activeSet)
  k.kernel.restrictActive(activeSet)

proc diag*[K](k: CachedKernel[K], i: int): float64 {.inline.} =
  k.kernel.getDiag(i)

proc activeSize*[K](k: CachedKernel[K]): int {.inline.} =
  k.kernel.activeSize

proc cacheSummary*[K](k: CachedKernel[K]): string =
  fmt"{k.misses} of {k.accesses} = {k.misses / k.accesses * 100:.1f}%"
