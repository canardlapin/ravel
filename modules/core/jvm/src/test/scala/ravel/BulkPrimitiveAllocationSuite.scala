package ravel

import java.lang.management.ManagementFactory
import ravel.DType.given

class BulkPrimitiveAllocationSuite extends munit.FunSuite:
  private var sink: Any = null
  test("primitive copy allocation is one payload plus bounded metadata"):
    val values = Array.tabulate(65536)(i => i.toDouble + 0.125)
    val shape = Shape(256, 256)
    def construct(): Unit = sink = NDArray.fromArrayCopy(shape, values)
    var i = 0
    while i < 30 do { construct(); i += 1 }
    val bean = ManagementFactory.getThreadMXBean match
      case value: com.sun.management.ThreadMXBean if value.isThreadAllocatedMemorySupported => value
      case _ => fail("HotSpot allocation counter required")
    val enabled = bean.isThreadAllocatedMemoryEnabled
    if !enabled then bean.setThreadAllocatedMemoryEnabled(true)
    try
      val thread = Thread.currentThread().threadId()
      val before = bean.getThreadAllocatedBytes(thread)
      construct()
      val allocated = bean.getThreadAllocatedBytes(thread) - before
      assert(sink != null)
      println(s"Ravel primitive copy allocated=$allocated payload=${values.length * 8}")
      assert(
        allocated <= values.length * 8L + 4096,
        s"$allocated bytes includes per-element boxing"
      )
    finally if !enabled then bean.setThreadAllocatedMemoryEnabled(false)

  test("finite scans allocate bounded metadata independent of logical size"):
    val bean = ManagementFactory.getThreadMXBean.asInstanceOf[com.sun.management.ThreadMXBean]
    assert(bean.isThreadAllocatedMemorySupported)
    val enabled = bean.isThreadAllocatedMemoryEnabled
    if !enabled then bean.setThreadAllocatedMemoryEnabled(true)
    try
      for rows <- Vector(8, 256) do
        val array = NDArray.fromArrayCopy(Shape(rows, 256), Array.fill(rows * 256)(1.25))
        for view <- Vector(array, array.transpose, array.slice(1, 0 until 256 by 2).reverse(0)) do
          var i = 0
          while i < 100 do { assert(view.allFinite); i += 1 }
          val thread = Thread.currentThread().threadId()
          val before = bean.getThreadAllocatedBytes(thread)
          var finite = true
          i = 0
          while i < 100 do { finite = finite && view.allFinite; i += 1 }
          val allocated = bean.getThreadAllocatedBytes(thread) - before
          assert(finite)
          println(
            s"Ravel allFinite rows=$rows cells=${view.size} allocated=$allocated for 100 scans"
          )
          assert(allocated <= 8192, s"finite scan allocated $allocated bytes")
    finally if !enabled then bean.setThreadAllocatedMemoryEnabled(false)
