package ravel.stencil

import munit.FunSuite

final class BorderIndexSuite extends FunSuite:
  test("interior indices are identity for every mode"):
    BorderMode.values.foreach { mode =>
      assertEquals(BorderIndex.map(0, 5, mode), MappedIndex.Inside(0))
      assertEquals(BorderIndex.map(4, 5, mode), MappedIndex.Inside(4))
    }

  test("Constant yields Outside outside the domain"):
    assertEquals(BorderIndex.map(-1, 4, BorderMode.Constant), MappedIndex.Outside)
    assertEquals(BorderIndex.map(4, 4, BorderMode.Constant), MappedIndex.Outside)

  test("Replicate clamps to the nearest edge"):
    assertEquals(BorderIndex.mapInside(-3, 4, BorderMode.Replicate), 0)
    assertEquals(BorderIndex.mapInside(10, 4, BorderMode.Replicate), 3)

  test("Wrap is modular with a positive representative"):
    assertEquals(BorderIndex.mapInside(-1, 4, BorderMode.Wrap), 3)
    assertEquals(BorderIndex.mapInside(4, 4, BorderMode.Wrap), 0)
    assertEquals(BorderIndex.mapInside(5, 4, BorderMode.Wrap), 1)
    assertEquals(BorderIndex.mapInside(-5, 4, BorderMode.Wrap), 3)

  test("ReflectWithoutEdge does not duplicate the edge sample"):
    // extent 5, indices around the left edge: ... 2,1,0,1,2 ...
    assertEquals(
      BorderIndex.mapInside(-1, 5, BorderMode.ReflectWithoutEdge),
      1
    )
    assertEquals(
      BorderIndex.mapInside(-2, 5, BorderMode.ReflectWithoutEdge),
      2
    )
    assertEquals(
      BorderIndex.mapInside(5, 5, BorderMode.ReflectWithoutEdge),
      3
    )
    assertEquals(
      BorderIndex.mapInside(6, 5, BorderMode.ReflectWithoutEdge),
      2
    )
    assertEquals(
      BorderIndex.mapInside(0, 1, BorderMode.ReflectWithoutEdge),
      0
    )
    assertEquals(
      BorderIndex.mapInside(-3, 1, BorderMode.ReflectWithoutEdge),
      0
    )

  test("ReflectWithEdge duplicates the edge sample"):
    // extent 4, left exterior: ... 2,1,0,0,1,2 ...
    assertEquals(BorderIndex.mapInside(-1, 4, BorderMode.ReflectWithEdge), 0)
    assertEquals(BorderIndex.mapInside(-2, 4, BorderMode.ReflectWithEdge), 1)
    assertEquals(BorderIndex.mapInside(4, 4, BorderMode.ReflectWithEdge), 3)
    assertEquals(BorderIndex.mapInside(5, 4, BorderMode.ReflectWithEdge), 2)

  test("Reflect modes disagree on the first exterior sample"):
    assertNotEquals(
      BorderIndex.mapInside(-1, 5, BorderMode.ReflectWithoutEdge),
      BorderIndex.mapInside(-1, 5, BorderMode.ReflectWithEdge)
    )

  test("logical coordinates widen before addition"):
    assertEquals(
      StencilArithmetic.logicalCoordinate(Int.MaxValue, Int.MaxValue, Int.MaxValue),
      6442450941L
    )
    assertEquals(
      StencilArithmetic.logicalCoordinate(Int.MinValue, Int.MinValue, Int.MinValue),
      -6442450944L
    )

  test("Long-domain wrapping and reflection do not overflow Int periods"):
    assertEquals(BorderIndex.mapInside(Long.MinValue, 5, BorderMode.Wrap), 2)
    assertEquals(
      BorderIndex.mapInside(
        Int.MaxValue.toLong,
        Int.MaxValue,
        BorderMode.ReflectWithEdge
      ),
      Int.MaxValue - 1
    )
    assertEquals(
      BorderIndex.mapInside(
        Int.MaxValue.toLong,
        Int.MaxValue,
        BorderMode.ReflectWithoutEdge
      ),
      Int.MaxValue - 2
    )
