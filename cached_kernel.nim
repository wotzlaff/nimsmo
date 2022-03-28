import lrucache
import std/[strformat]
import kernel

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

proc `activeSet=`*[K](k: CachedKernel[K], activeSet: seq[int]) =
  k.kernel.activeSet = activeSet
  # TODO: restrict the available data?
  k.cache.clear()

proc diag*[K](k: CachedKernel[K], i: int): float64 {.inline.} =
  k.kernel.diag(i)

proc size*[K](k: CachedKernel[K]): int {.inline.} =
  k.kernel.size

proc cacheSummary*[K](k: CachedKernel[K]): string =
  fmt"{k.misses} of {k.accesses} = {k.misses / k.accesses * 100:.1f}%"
