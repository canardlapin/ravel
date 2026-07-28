package ravel.bench

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import ravel.NDArray

/** Emits semantic signatures for every operation-matrix benchmark case.
  *
  * This is intentionally independent of JMH so CI can run the full semantic matrix quickly. The
  * existing cross-platform reduction law suite remains the raw-bit authority for Ravel's exact
  * 128-value block-pairwise schedule.
  */
object OperationMatrixParity:
  private final case class Config(
      output: Path = Paths.get("target/operation-matrix/ravel-signatures.json"),
      sides: Vector[Int] = Vector(32, 64)
  )

  private final case class Signature(
      name: String,
      family: String,
      side: Int,
      inputDType: String,
      resultDType: String,
      inputLayout: String,
      logicalWorkUnits: Long,
      workUnit: String,
      comparison: String,
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
    println(s"Wrote ${signatures.size} operation-matrix signatures to ${config.output}")

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
    val cases = OperationMatrixCases(new OperationMatrixFixture(side))
    val names = cases.map(_.name)
    require(names.distinct.size == names.size, s"duplicate operation-matrix case at side $side")
    cases.map { operation =>
      signature(operation, side, operation.signature())
    }

  private def signature(operation: OperationMatrixCase, side: Int, value: Any): Signature =
    value match
      case array: NDArray[?, ?] =>
        val iterator = array.elementsIterator
        var logicalIndex = 1L
        var sum = 0.0
        var weightedSum = 0.0
        while iterator.hasNext do
          val scalar = toDouble(iterator.next())
          sum += scalar
          weightedSum += scalar * logicalIndex.toDouble
          logicalIndex += 1L
        Signature(
          operation.name,
          operation.family,
          side,
          operation.inputDType,
          operation.resultDType,
          operation.inputLayout,
          operation.logicalWorkUnits,
          operation.workUnit,
          operation.comparison,
          array.size,
          if array.isContiguous then "c_contiguous" else "strided_view",
          sum,
          weightedSum
        )
      case scalar =>
        val number = toDouble(scalar)
        Signature(
          operation.name,
          operation.family,
          side,
          operation.inputDType,
          operation.resultDType,
          operation.inputLayout,
          operation.logicalWorkUnits,
          operation.workUnit,
          operation.comparison,
          1,
          "scalar",
          number,
          number
        )

  private def toDouble(value: Any): Double =
    value match
      case boolean: Boolean => if boolean then 1.0 else 0.0
      case byte: Byte => byte.toDouble
      case short: Short => short.toDouble
      case int: Int => int.toDouble
      case long: Long => long.toDouble
      case float: Float => float.toDouble
      case double: Double => double
      case other =>
        throw new IllegalArgumentException(
          s"unsupported operation-matrix result element: ${other.getClass.getName}"
        )

  private def render(signatures: Vector[Signature]): String =
    val results = signatures.map(renderSignature).mkString(",\n")
    s"""{
       |  "schema": "ravel.operation-matrix.signatures.v1",
       |  "metadata": {
       |    "library": "ravel",
       |    "scala_version": "${escape(scala.util.Properties.versionNumberString)}",
       |    "java_version": "${escape(System.getProperty("java.version"))}",
       |    "java_vendor": "${escape(System.getProperty("java.vendor"))}",
       |    "os_name": "${escape(System.getProperty("os.name"))}",
       |    "os_arch": "${escape(System.getProperty("os.arch"))}",
       |    "available_processors": ${Runtime.getRuntime.availableProcessors()},
       |    "exact_reduction_receipt": "ReductionLawsSuite raw-bit reference checks on JVM and Scala.js"
       |  },
       |  "results": [
       |$results
       |  ]
       |}
       |""".stripMargin

  private def renderSignature(signature: Signature): String =
    s"""    {
       |      "case": "${escape(signature.name)}",
       |      "family": "${escape(signature.family)}",
       |      "side": ${signature.side},
       |      "input_dtype": "${escape(signature.inputDType)}",
       |      "result_dtype": "${escape(signature.resultDType)}",
       |      "input_layout": "${escape(signature.inputLayout)}",
       |      "logical_work_units": ${signature.logicalWorkUnits},
       |      "work_unit": "${escape(signature.workUnit)}",
       |      "comparison": "${escape(signature.comparison)}",
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
