package ravel.internal

import ravel.*

private[ravel] object FiniteReduction:
  def all[A](array: ArraySource[A, ?]): Boolean =
    array.dtype.tag match
      case DType.DoubleTag =>
        val raw = array.storage.asInstanceOf[DoubleStorage].raw
        scan(array.layout)(index => raw(index).isFinite)
      case DType.FloatTag =>
        val raw = array.storage.asInstanceOf[FloatStorage].raw
        scan(array.layout)(index => raw(index).isFinite)
      case _ => throw new UnsupportedOperationException("allFinite requires Float or Double")

  // The inline predicate becomes a primitive load in each dtype branch. Layout admission has
  // proved every reachable address in storage; Long intermediates avoid stride rewind overflow.
  private inline def scan(layout: Layout)(inline finite: Int => Boolean): Boolean =
    if layout.size == 0 then true
    else if layout.isPhysicallyDense then
      var address = layout.minimumPhysicalAddress
      val end = address.toLong + layout.size.toLong
      var result = true
      while address.toLong < end && result do
        result = finite(address)
        address += 1
      result
    else
      val counters = new Array[Int](layout.rank)
      var address = layout.offset.toLong
      var remaining = layout.size
      var result = true
      while remaining > 0 && result do
        result = finite(address.toInt)
        remaining -= 1
        if remaining > 0 && result then
          var axis = layout.rank - 1
          var advanced = false
          while axis >= 0 && !advanced do
            counters(axis) += 1
            address += layout.strides(axis).toLong
            if counters(axis) < layout.shape(axis) then advanced = true
            else
              address -= counters(axis).toLong * layout.strides(axis).toLong
              counters(axis) = 0
              axis -= 1
      result
