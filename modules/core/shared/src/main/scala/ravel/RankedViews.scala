package ravel

import ravel.internal.ViewLayout

extension [A, R <: AnyRank](array: NDArray[A, R])
  def select(axis: Int, index: Int)(using
      CanDropAxis[R]
  ): NDArray[A, DropAxis[R]] =
    new NDArray[A, DropAxis[R]](
      array.storage,
      ViewLayout.select(array.layout, axis, index, array.storage.length),
      array.dtype
    )

  def newAxis(axis: Int): NDArray[A, AddAxis[R]] =
    new NDArray[A, AddAxis[R]](
      array.storage,
      ViewLayout.newAxis(array.layout, axis, array.storage.length),
      array.dtype
    )

  def squeeze(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    new NDArray[A, DropAxis[R]](
      array.storage,
      ViewLayout.squeeze(array.layout, axis, array.storage.length),
      array.dtype
    )
