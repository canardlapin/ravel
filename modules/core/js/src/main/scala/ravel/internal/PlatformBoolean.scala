package ravel.internal

private[ravel] object PlatformBoolean:
  inline def get(storage: BooleanStorage, index: Int): Boolean =
    storage.raw(index) != 0

  inline def set(storage: BooleanStorage, index: Int, value: Boolean): Unit =
    storage.raw(index) = (if value then 1 else 0)
