import std/sugar
import system/dollars
import arraymancer
import nimpy, nimpy/raw_buffers

proc `+`[T](a: ptr T, b: int): ptr T =
  cast[ptr T](cast[uint](a) + cast[uint](b * a[].sizeof))

type NumpyArray*[T] = ref object
  data*: Tensor[T]
  buf: RawPyBuffer

proc release*(arr: NumpyArray) =
  arr.buf.release()

proc fromNumpy*[T](obj: PyObject): NumpyArray[T] =
  var buf: RawPyBuffer
  obj.getBuffer(buf, PyBUF_ND)
  doAssert(buf.ndim > 0)
  doAssert(buf.strides == nil)
  let shape = collect:
    for i in 0..<buf.ndim:
      (buf.shape + i)[]
  result.new()
  result.data = fromBuffer(cast[ptr UncheckedArray[T]](buf.buf), shape)
  result.buf = buf

proc `$`*[T](arr: NumpyArray[T]): string = $arr.data
