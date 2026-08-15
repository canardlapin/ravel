package ravel.internal

private[ravel] object PlatformBoolean:
  inline def get(storage: BooleanStorage, index: Int): Boolean =
    storage.raw(index)

  inline def set(storage: BooleanStorage, index: Int, value: Boolean): Unit =
    storage.raw(index) = value
