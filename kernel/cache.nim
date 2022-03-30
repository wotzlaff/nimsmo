import lrucache
import std/strformat
import base
export base

type
  CachedKernel*[K] = ref object
    cache: LruCache[int, KernelRow]
    kernel: K
    accesses: int
    misses: int

proc cache*[K](kernel: K, capacity: int): CachedKernel[K] =
  CachedKernel[K](
    cache: newLruCache[int, KernelRow](capacity),
    kernel: kernel,
    accesses: 0,
    misses: 0,
  )

proc getRow*(k: CachedKernel, i: int): KernelRow {.inline.} =
  k.accesses += 1
  if i notin k.cache:
    k.misses += 1
    k.cache[i] = k.kernel.getRow(i)
  k.cache[i]

proc setActive*(k: CachedKernel, activeSet: seq[int]) {.inline.} =
  k.kernel.activeSet = activeSet
  k.cache.clear()

proc restrictActive*(k: CachedKernel, activeSet: seq[int]) =
  for (key, row) in k.cache.mitems:
    row.restrict(k.kernel.activeSet, activeSet)
  k.kernel.activeSet = activeSet

proc diag*(k: CachedKernel, i: int): float64 {.inline.} =
  k.kernel.diag(i)

proc activeSize*(k: CachedKernel): int {.inline.} =
  k.kernel.activeSize

proc cacheSummary*(k: CachedKernel): string =
  fmt"{k.misses} of {k.accesses} = {k.misses / k.accesses * 100:.1f}%"

proc size*(k: CachedKernel): int =
  k.kernel.size
