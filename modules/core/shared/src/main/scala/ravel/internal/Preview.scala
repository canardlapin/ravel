package ravel.internal

import ravel.*

private[ravel] object Preview:
  private val MaxPerAxis = 4
  private val MaxElements = 24

  def render[A](array: NDArray[A, ?]): String =
    if array.rank == 0 then String.valueOf(array.at(IArray.empty))
    else
      val indices = new Array[Int](array.rank)
      var emitted = 0

      def axisPreview(axis: Int): String =
        if emitted >= MaxElements then "…"
        else if axis == array.rank then
          emitted += 1
          String.valueOf(array.at(IArray.unsafeFromArray(indices.clone())))
        else
          val dimension = array.layout.shape(axis)
          val shown = math.min(dimension, MaxPerAxis)
          val builder = new StringBuilder("[")
          var index = 0
          while index < shown && emitted < MaxElements do
            if index > 0 then builder.append(", ")
            indices(axis) = index
            builder.append(axisPreview(axis + 1))
            index += 1
          if dimension > shown || (index < dimension && emitted >= MaxElements) then
            if index > 0 then builder.append(", ")
            builder.append("…")
          builder.append("]")
          builder.result()

      axisPreview(0)
