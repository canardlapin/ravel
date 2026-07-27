package ravel.internal

private[ravel] object PlatformBoolean:
  inline def set(storage: BooleanStorage, index: Int, value: Boolean): Unit =
    storage.raw(index) = (if value then 1 else 0)
