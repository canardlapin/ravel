package ravel.stencil

/** Border handling for neighborhood reads outside the source domain.
  *
  * Deliberately avoids the bare word "Mirror": libraries disagree on whether the edge sample is
  * duplicated.
  */
enum BorderMode derives CanEqual:
  case Constant
  case Replicate
  case ReflectWithoutEdge
  case ReflectWithEdge
  case Wrap

/** Result of mapping a possibly out-of-domain index. */
enum MappedIndex derives CanEqual:
  case Inside(index: Int)

  /** Out of domain; only produced for [[BorderMode.Constant]]. */
  case Outside
