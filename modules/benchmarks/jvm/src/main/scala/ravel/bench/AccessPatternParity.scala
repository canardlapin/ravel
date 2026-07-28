package ravel.bench

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import ravel.NDArray

object AccessPatternParity:
  private final case class Config(
      output: Path = Paths.get("target/access-patterns/ravel-signatures.json"),
      sides: Vector[Int] = Vector(256, 1024)
  )

  private final case class Signature(
      name: String,
      side: Int,
      logicalWorkUnits: Long,
      workUnit: String,
      resultSize: Int,
      resultLayout: String,
      sum: Double,
      weightedSum: Double
  )

  def main(args: Array[String]): Unit =
    val config = parseArgs(args)
    val signatures = config.sides.flatMap(signaturesFor)
    val parent = config.output.toAbsolutePath.getParent
    if parent != null then
      val _ = Files.createDirectories(parent)
    val _ = Files.writeString(
      config.output,
      render(signatures),
      StandardCharsets.UTF_8
    )
    println(s"Wrote ${signatures.size} signatures to ${config.output}")

  private def parseArgs(args: Array[String]): Config =
    var config = Config()
    var index = 0
    while index < args.length do
      args(index) match
        case "--out" =>
          index += 1
          require(index < args.length, "--out requires a path")
          config = config.copy(output = Paths.get(args(index)))
        case "--side" =>
          index += 1
          require(index < args.length, "--side requires a comma-separated list")
          val sides = args(index).split(",").toVector.map(_.toInt)
          require(sides.nonEmpty, "--side requires at least one size")
          config = config.copy(sides = sides)
        case argument =>
          throw new IllegalArgumentException(s"unknown argument: $argument")
      index += 1
    config

  private def signaturesFor(side: Int): Vector[Signature] =
    val fixture = new AccessPatternFixture(side)
    val full = Math.multiplyExact(side, side).toLong
    val half = full / 2L
    Vector(
      arraySignature(
        "contiguous_add",
        side,
        full,
        "element",
        AccessPatternWorkloads.contiguousAdd(fixture)
      ),
      arraySignature(
        "inner_stride_add",
        side,
        half,
        "element",
        AccessPatternWorkloads.innerStrideAdd(fixture)
      ),
      arraySignature(
        "outer_stride_add",
        side,
        half,
        "element",
        AccessPatternWorkloads.outerStrideAdd(fixture)
      ),
      arraySignature(
        "reverse_add",
        side,
        full,
        "element",
        AccessPatternWorkloads.reverseAdd(fixture)
      ),
      arraySignature(
        "transpose_add",
        side,
        full,
        "element",
        AccessPatternWorkloads.transposeAdd(fixture)
      ),
      arraySignature(
        "broadcast_row_add",
        side,
        full,
        "element",
        AccessPatternWorkloads.broadcastRowAdd(fixture)
      ),
      scalarSignature(
        "full_sum_contiguous",
        side,
        full,
        "element",
        AccessPatternWorkloads.fullSumContiguous(fixture)
      ),
      scalarSignature(
        "full_sum_inner_stride",
        side,
        half,
        "element",
        AccessPatternWorkloads.fullSumInnerStride(fixture)
      ),
      arraySignature(
        "axis0_sum",
        side,
        full,
        "element",
        AccessPatternWorkloads.axis0Sum(fixture)
      ),
      arraySignature(
        "axis1_sum",
        side,
        full,
        "element",
        AccessPatternWorkloads.axis1Sum(fixture)
      ),
      arraySignature(
        "copy_inner_stride",
        side,
        half,
        "element",
        AccessPatternWorkloads.copyInnerStride(fixture)
      ),
      arraySignature(
        "copy_transpose",
        side,
        full,
        "element",
        AccessPatternWorkloads.copyTranspose(fixture)
      ),
      scalarSignature(
        "scalar_read_row_major",
        side,
        full,
        "element access",
        AccessPatternWorkloads.scalarReadRowMajor(fixture)
      ),
      scalarSignature(
        "scalar_read_column_major",
        side,
        full,
        "element access",
        AccessPatternWorkloads.scalarReadColumnMajor(fixture)
      ),
      arraySignature(
        "view_inner_stride_create",
        side,
        1L,
        "view",
        AccessPatternWorkloads.viewInnerStrideCreate(fixture)
      ),
      arraySignature(
        "view_transpose_create",
        side,
        1L,
        "view",
        AccessPatternWorkloads.viewTransposeCreate(fixture)
      )
    )

  private def scalarSignature(
      name: String,
      side: Int,
      logicalWorkUnits: Long,
      workUnit: String,
      value: Double
  ): Signature =
    Signature(
      name,
      side,
      logicalWorkUnits,
      workUnit,
      1,
      "scalar",
      value,
      value
    )

  private def arraySignature(
      name: String,
      side: Int,
      logicalWorkUnits: Long,
      workUnit: String,
      array: NDArray[Double, ?]
  ): Signature =
    val iterator = array.elementsIterator
    var logicalIndex = 1L
    var sum = 0.0
    var weightedSum = 0.0
    while iterator.hasNext do
      val value = iterator.next()
      sum += value
      weightedSum += value * logicalIndex.toDouble
      logicalIndex += 1L
    Signature(
      name,
      side,
      logicalWorkUnits,
      workUnit,
      array.size,
      if array.isContiguous then "c_contiguous" else "strided_view",
      sum,
      weightedSum
    )

  private def render(signatures: Vector[Signature]): String =
    val results = signatures.map(renderSignature).mkString(",\n")
    s"""{
       |  "schema": "ravel.access-pattern.signatures.v1",
       |  "metadata": {
       |    "library": "ravel",
       |    "scala_version": "${escape(scala.util.Properties.versionNumberString)}",
       |    "java_version": "${escape(System.getProperty("java.version"))}",
       |    "dtype": "float64"
       |  },
       |  "results": [
       |$results
       |  ]
       |}
       |""".stripMargin

  private def renderSignature(signature: Signature): String =
    s"""    {
       |      "case": "${escape(signature.name)}",
       |      "side": ${signature.side},
       |      "logical_work_units": ${signature.logicalWorkUnits},
       |      "work_unit": "${escape(signature.workUnit)}",
       |      "result_size": ${signature.resultSize},
       |      "result_layout": "${escape(signature.resultLayout)}",
       |      "sum": ${signature.sum},
       |      "weighted_sum": ${signature.weightedSum}
       |    }""".stripMargin

  private def escape(value: String): String =
    val builder = new StringBuilder
    value.foreach {
      case '"' => builder.append("\\\"")
      case '\\' => builder.append("\\\\")
      case '\n' => builder.append("\\n")
      case '\r' => builder.append("\\r")
      case '\t' => builder.append("\\t")
      case char => builder.append(char)
    }
    builder.result()
