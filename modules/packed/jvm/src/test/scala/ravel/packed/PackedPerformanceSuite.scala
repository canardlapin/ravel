package ravel.packed

import java.lang.management.ManagementFactory

import com.sun.management.ThreadMXBean
import munit.FunSuite
import ravel.Shape

final class PackedPerformanceSuite extends FunSuite:
  private var retainedLong: Long = 0L
  private var retained: AnyRef = null

  test("block iteration over 1-bit and 4-bit codes allocates nothing"):
    val samples = 1 << 20
    val oneBit =
      packedRight(
        PackedArray.tabulate(Shape(samples), PackedBits.B1)(index => (index * 5) % 7 & 1)
      )
    val fourBit =
      packedRight(
        PackedArray.tabulate(Shape(samples), PackedBits.B4)(index => (index * 11) % 16)
      )

    Vector(("1-bit", oneBit), ("4-bit", fourBit)).foreach { (name, packed) =>
      var warmup = 0
      while warmup < 5 do
        retainedLong = packed.sumCodes
        warmup += 1
      val allocated =
        Vector
          .fill(7)(allocatedBytes { retainedLong = packed.sumCodes })
          .sorted
          .apply(3)

      assert(
        allocated <= 256L,
        s"$name block iteration allocated $allocated B; expected none"
      )
      println(
        s"RAVEL-PACKED JVM baseline: case=block-iteration-$name, " +
          s"samples=$samples, allocated=$allocated B, checksum=$retainedLong"
      )
    }

  test("specialized wordwise mask operations beat Boolean expansion baselines"):
    val samples = 1 << 20
    val leftPacked =
      packedRight(
        PackedArray.tabulate(Shape(samples), PackedBits.B1)(index =>
          if (index * 5) % 7 < 3 then 1 else 0
        )
      )
    val rightPacked =
      packedRight(
        PackedArray.tabulate(Shape(samples), PackedBits.B1)(index =>
          if (index * 3) % 5 < 2 then 1 else 0
        )
      )
    val leftBooleans = Array.tabulate(samples)(index => (index * 5) % 7 < 3)
    val rightBooleans = Array.tabulate(samples)(index => (index * 3) % 5 < 2)

    def wordwiseUnion(): Unit =
      retained = packedRight(PackedBitOps.union(leftPacked, rightPacked))

    def wordwiseIntersection(): Unit =
      retained = packedRight(PackedBitOps.intersection(leftPacked, rightPacked))

    def wordwiseDifference(): Unit =
      retained = packedRight(PackedBitOps.difference(leftPacked, rightPacked))

    def wordwiseXor(): Unit =
      retained = packedRight(PackedBitOps.xor(leftPacked, rightPacked))

    def expansionUnion(): Unit =
      val output = new Array[Boolean](samples)
      var index = 0
      while index < samples do
        output(index) = leftBooleans(index) || rightBooleans(index)
        index += 1
      retained = output

    def expansionIntersection(): Unit =
      val output = new Array[Boolean](samples)
      var index = 0
      while index < samples do
        output(index) = leftBooleans(index) && rightBooleans(index)
        index += 1
      retained = output

    def expansionDifference(): Unit =
      val output = new Array[Boolean](samples)
      var index = 0
      while index < samples do
        output(index) = leftBooleans(index) && !rightBooleans(index)
        index += 1
      retained = output

    def expansionXor(): Unit =
      val output = new Array[Boolean](samples)
      var index = 0
      while index < samples do
        output(index) = leftBooleans(index) != rightBooleans(index)
        index += 1
      retained = output

    val cases = Vector(
      ("union", () => wordwiseUnion(), () => expansionUnion()),
      ("intersection", () => wordwiseIntersection(), () => expansionIntersection()),
      ("difference", () => wordwiseDifference(), () => expansionDifference()),
      ("xor", () => wordwiseXor(), () => expansionXor())
    )
    val wordBytes = PackedArray.wordCount(samples, PackedBits.B1).toLong * 4L

    cases.foreach { (name, wordwise, expansion) =>
      var warmup = 0
      while warmup < 10 do
        wordwise()
        expansion()
        warmup += 1

      val wordwiseNanos = Vector.fill(9)(elapsedNanos(wordwise())).sorted.apply(4)
      val expansionNanos = Vector.fill(9)(elapsedNanos(expansion())).sorted.apply(4)
      val wordwiseAllocated =
        Vector.fill(7)(allocatedBytes(wordwise())).sorted.apply(3)

      assert(
        wordwiseAllocated <= wordBytes + 4096L,
        s"wordwise $name allocated $wordwiseAllocated B; words need $wordBytes B"
      )
      assert(
        wordwiseNanos < expansionNanos,
        s"wordwise $name ($wordwiseNanos ns) must beat Boolean expansion ($expansionNanos ns)"
      )
      println(
        s"RAVEL-PACKED JVM baseline: case=mask-$name, samples=$samples, " +
          s"wordwise=$wordwiseNanos ns, expansion=$expansionNanos ns, " +
          s"wordwiseAllocated=$wordwiseAllocated B"
      )
    }
    assert(retained != null)

  private def elapsedNanos(body: => Unit): Long =
    val before = System.nanoTime()
    body
    System.nanoTime() - before

  private def allocatedBytes(body: => Unit): Long =
    val bean =
      ManagementFactory.getThreadMXBean match
        case value: ThreadMXBean if value.isThreadAllocatedMemorySupported =>
          if !value.isThreadAllocatedMemoryEnabled then value.setThreadAllocatedMemoryEnabled(true)
          value
        case _ =>
          fail("thread allocation accounting is unavailable")
    val thread = Thread.currentThread().threadId()
    val before = bean.getThreadAllocatedBytes(thread)
    body
    bean.getThreadAllocatedBytes(thread) - before

  private def packedRight[A](value: Either[PackedError, A]): A =
    value match
      case Right(result) => result
      case Left(error) => fail(error.message)
