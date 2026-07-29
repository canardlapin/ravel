package ravel.bench

import ravel.*
import ravel.DType.given
import ravel.internal.{ProbeApi, Storage}

private[bench] trait CourtPlatformContract:
  def runtimeName: String
  def runtimeVersion: String
  def vmName: String
  def osName: String
  def osArch: String
  def availableProcessors: Int
  def timerName: String
  def isScalaJs: Boolean
  def explicitGcAvailable: Boolean
  def nowNanos(): Double
  def releaseCaseMemory(): Unit
  def commandLineArgs(default: Array[String]): Array[String]
  def writeText(path: String, content: String): Unit

private[bench] enum CourtTimingScope:
  case AllPlatforms
  case JvmOnly

  def label: String =
    this match
      case AllPlatforms => "all_platforms"
      case JvmOnly => "jvm_only"

private[bench] final case class CourtConfig(
    output: Option[String] = None,
    sides: Vector[Int] = Vector(64, 256, 1024),
    warmupMillis: Int = 200,
    sampleMillis: Int = 250,
    samples: Int = 7,
    caseFilter: Option[String] = None,
    signaturesOnly: Boolean = false,
    progress: Boolean = false
):
  def validate(): CourtConfig =
    require(sides.nonEmpty, "--sides requires at least one side")
    require(sides.distinct.size == sides.size, s"--sides contains duplicates: $sides")
    sides.foreach { side =>
      require(side > 0 && side % 2 == 0, s"side must be a positive even integer, got $side")
    }
    require(warmupMillis >= 0, s"--warmup-ms must be non-negative, got $warmupMillis")
    require(sampleMillis > 0, s"--sample-ms must be positive, got $sampleMillis")
    require(samples > 0, s"--samples must be positive, got $samples")
    require(caseFilter.forall(_.nonEmpty), "--case-filter must not be empty")
    this

private[bench] final case class CourtCase(
    name: String,
    family: String,
    inputDType: String,
    resultDType: String,
    inputLayout: String,
    logicalWorkUnits: Long,
    comparison: String,
    timingScope: CourtTimingScope,
    run: () => Any,
    signature: () => Any
)

private[bench] final case class CourtCaseDefinition(
    name: String,
    family: String,
    inputDType: String,
    resultDType: String,
    inputLayout: String,
    comparison: String,
    timingScope: CourtTimingScope,
    maximumSemanticSide: Option[Int],
    prepare: Int => CourtCase
):
  def includes(side: Int): Boolean =
    maximumSemanticSide.forall(side <= _)

private[bench] final case class CourtSignature(
    resultSize: Int,
    resultLayout: String,
    sum: Double,
    weightedSum: Double,
    nanCount: Int,
    positiveInfinityCount: Int,
    negativeInfinityCount: Int,
    negativeZeroCount: Int
)

private[bench] final case class CourtTiming(
    batchSize: Int,
    medianNanos: Double,
    minimumNanos: Double,
    maximumNanos: Double,
    p25Nanos: Double,
    p75Nanos: Double,
    relativeStandardDeviation: Double,
    samplesNanos: Vector[Double]
)

private[bench] final case class CourtCaseMetadata(
    name: String,
    family: String,
    inputDType: String,
    resultDType: String,
    inputLayout: String,
    logicalWorkUnits: Long,
    comparison: String,
    timingScope: CourtTimingScope
)

private[bench] object CourtCaseMetadata:
  def apply(operation: CourtCase): CourtCaseMetadata =
    CourtCaseMetadata(
      operation.name,
      operation.family,
      operation.inputDType,
      operation.resultDType,
      operation.inputLayout,
      operation.logicalWorkUnits,
      operation.comparison,
      operation.timingScope
    )

private[bench] final case class CourtResult(
    operation: CourtCaseMetadata,
    side: Int,
    signature: CourtSignature,
    timingStatus: String,
    timing: Option[CourtTiming]
)

/** Reproducible public-operation court shared by the JVM and Scala.js benchmark artifacts.
  *
  * JMH remains the authority for JVM performance. This runner exists to give full-linked Node an
  * equivalent public-API court and to verify that both platforms execute the same semantic cases.
  */
object CrossRuntimeCourt:
  private var timingSink: Any = ()

  def main(args: Array[String]): Unit =
    val config = parseArgs(CrossRuntimeCourtPlatform.commandLineArgs(args)).validate()
    val results = execute(config, CrossRuntimeCourtPlatform)
    val rendered = render(config, CrossRuntimeCourtPlatform, results)
    val output = config.output.getOrElse(
      s"target/cross-runtime-court/${CrossRuntimeCourtPlatform.runtimeName}.json"
    )
    CrossRuntimeCourtPlatform.writeText(output, rendered)
    println(
      s"Wrote ${results.size} cross-runtime court rows to $output " +
        s"(${if config.signaturesOnly then "signatures only" else "timed"})"
    )

  private[bench] def execute(
      config: CourtConfig,
      platform: CourtPlatformContract
  ): Vector[CourtResult] =
    val validated = config.validate()
    val results = Vector.newBuilder[CourtResult]
    validated.sides.foreach { side =>
      val definitions =
        selectedDefinitions(CrossRuntimeCourtCases.definitions, validated.caseFilter)
          .filter(_.includes(side))
      val names = definitions.map(_.name)
      require(names.nonEmpty, s"no cases selected at side $side")
      require(names.distinct.size == names.size, s"duplicate cross-runtime case at side $side")
      definitions.foreach { definition =>
        platform.releaseCaseMemory()
        val prepareStarted = platform.nowNanos()
        val operation = definition.prepare(side)
        val prepared = platform.nowNanos()
        if validated.progress then
          println(
            f"[court] side=$side case=${definition.name} " +
              f"prepare_ms=${(prepared - prepareStarted) / 1000000.0}%.3f"
          )
        val signatureValue = operation.signature()
        val operated = platform.nowNanos()
        if validated.progress then
          println(
            f"[court] side=$side case=${definition.name} " +
              f"operation_ms=${(operated - prepared) / 1000000.0}%.3f"
          )
        val signature = signatureOf(signatureValue)
        val signed = platform.nowNanos()
        if validated.progress then
          println(
            f"[court] side=$side case=${definition.name} " +
              f"signature_ms=${(signed - operated) / 1000000.0}%.3f"
          )
        val timingAllowed =
          !validated.signaturesOnly &&
            (operation.timingScope == CourtTimingScope.AllPlatforms || !platform.isScalaJs)
        val timing =
          if timingAllowed then Some(measure(definition.prepare(side).run, validated, platform))
          else None
        val status =
          if validated.signaturesOnly then "not_requested"
          else if timingAllowed then "measured"
          else "correctness_only"
        results += CourtResult(CourtCaseMetadata(operation), side, signature, status, timing)
      }
    }
    platform.releaseCaseMemory()
    results.result()

  private def selectedDefinitions(
      cases: Vector[CourtCaseDefinition],
      filter: Option[String]
  ): Vector[CourtCaseDefinition] =
    filter match
      case Some(pattern) => cases.filter(_.name.contains(pattern))
      case None => cases

  private[bench] def signatureOf(value: Any): CourtSignature =
    value match
      case array: NDArray[?, ?] =>
        if array.isContiguous then contiguousSignature(array)
        else iteratorSignature(array)
      case scalar =>
        val number = toDouble(scalar)
        CourtSignature(
          1,
          "scalar",
          if number.isFinite then number else 0.0,
          if number.isFinite then number else 0.0,
          if number.isNaN then 1 else 0,
          if number == Double.PositiveInfinity then 1 else 0,
          if number == Double.NegativeInfinity then 1 else 0,
          if isNegativeZero(number) then 1 else 0
        )

  private def contiguousSignature(array: NDArray[?, ?]): CourtSignature =
    var logicalIndex = 1
    var sum = 0.0
    var weightedSum = 0.0
    var nanCount = 0
    var positiveInfinityCount = 0
    var negativeInfinityCount = 0
    var negativeZeroCount = 0
    inline def record(scalar: Double): Unit =
      if scalar.isNaN then nanCount += 1
      else if scalar == Double.PositiveInfinity then positiveInfinityCount += 1
      else if scalar == Double.NegativeInfinity then negativeInfinityCount += 1
      else
        if isNegativeZero(scalar) then negativeZeroCount += 1
        sum += scalar
        weightedSum += scalar * logicalIndex.toDouble
      logicalIndex += 1
    inline def scan(inline read: Int => Double): Unit =
      var index = 0
      while index < array.size do
        record(read(index))
        index += 1

    val offset = array.layout.offset
    array.dtype.tag match
      case DType.BooleanTag =>
        val storage = array.storage.asInstanceOf[Storage[Boolean]]
        scan(index => if ProbeApi.getBoolean(storage, offset + index) then 1.0 else 0.0)
      case DType.ByteTag =>
        val storage = array.storage.asInstanceOf[Storage[Byte]]
        scan(index => ProbeApi.getByte(storage, offset + index).toDouble)
      case DType.ShortTag =>
        val storage = array.storage.asInstanceOf[Storage[Short]]
        scan(index => ProbeApi.getShort(storage, offset + index).toDouble)
      case DType.IntTag =>
        val storage = array.storage.asInstanceOf[Storage[Int]]
        scan(index => ProbeApi.getInt(storage, offset + index).toDouble)
      case DType.LongTag =>
        val storage = array.storage.asInstanceOf[Storage[Long]]
        scan(index => ProbeApi.getLong(storage, offset + index).toDouble)
      case DType.FloatTag =>
        val storage = array.storage.asInstanceOf[Storage[Float]]
        scan(index => ProbeApi.getFloat(storage, offset + index).toDouble)
      case DType.DoubleTag =>
        val storage = array.storage.asInstanceOf[Storage[Double]]
        scan(index => ProbeApi.getDouble(storage, offset + index))
      case tag => throw new MatchError(tag)
    CourtSignature(
      array.size,
      "c_contiguous",
      sum,
      weightedSum,
      nanCount,
      positiveInfinityCount,
      negativeInfinityCount,
      negativeZeroCount
    )

  private def iteratorSignature(array: NDArray[?, ?]): CourtSignature =
    val iterator = array.elementsIterator
    var logicalIndex = 1
    var sum = 0.0
    var weightedSum = 0.0
    var nanCount = 0
    var positiveInfinityCount = 0
    var negativeInfinityCount = 0
    var negativeZeroCount = 0
    while iterator.hasNext do
      val scalar = toDouble(iterator.next())
      if scalar.isNaN then nanCount += 1
      else if scalar == Double.PositiveInfinity then positiveInfinityCount += 1
      else if scalar == Double.NegativeInfinity then negativeInfinityCount += 1
      else
        if isNegativeZero(scalar) then negativeZeroCount += 1
        sum += scalar
        weightedSum += scalar * logicalIndex.toDouble
      logicalIndex += 1
    CourtSignature(
      array.size,
      if array.isContiguous then "c_contiguous" else "strided_view",
      sum,
      weightedSum,
      nanCount,
      positiveInfinityCount,
      negativeInfinityCount,
      negativeZeroCount
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
          s"unsupported cross-runtime result element: ${other.getClass.getName}"
        )

  private def isNegativeZero(value: Double): Boolean =
    value == 0.0 && 1.0 / value == Double.NegativeInfinity

  private def measure(
      operation: () => Any,
      config: CourtConfig,
      platform: CourtPlatformContract
  ): CourtTiming =
    val sampleTarget = config.sampleMillis.toDouble * 1000000.0
    val calibrationTarget = math.max(1000000.0, math.min(sampleTarget / 8.0, 20000000.0))
    var batchSize = 1
    var calibration = timedBatch(operation, batchSize, platform)
    while calibration._1 < calibrationTarget && batchSize < (1 << 26) do
      batchSize *= 2
      calibration = timedBatch(operation, batchSize, platform)
    timingSink = calibration._2

    if config.warmupMillis > 0 then
      val warmupTarget = config.warmupMillis.toDouble * 1000000.0
      val started = platform.nowNanos()
      var now = started
      while now - started < warmupTarget do
        val warmed = timedBatch(operation, batchSize, platform)
        timingSink = warmed._2
        now = platform.nowNanos()

    val samples = Vector.tabulate(config.samples) { _ =>
      var iterations = 0
      var elapsed = 0.0
      var last: Any = ()
      while elapsed < sampleTarget do
        val measured = timedBatch(operation, batchSize, platform)
        elapsed += measured._1
        last = measured._2
        iterations += batchSize
      timingSink = last
      elapsed / iterations.toDouble
    }
    val sorted = samples.sorted
    val mean = samples.sum / samples.size.toDouble
    val variance =
      samples.iterator.map { value =>
        val centered = value - mean
        centered * centered
      }.sum / samples.size.toDouble
    require(timingSink != null, "timed operation unexpectedly returned null")
    CourtTiming(
      batchSize,
      percentile(sorted, 0.5),
      sorted.head,
      sorted.last,
      percentile(sorted, 0.25),
      percentile(sorted, 0.75),
      if mean == 0.0 then 0.0 else math.sqrt(variance) / mean,
      samples
    )

  private def timedBatch(
      operation: () => Any,
      batchSize: Int,
      platform: CourtPlatformContract
  ): (Double, Any) =
    val started = platform.nowNanos()
    var index = 0
    var last: Any = ()
    while index < batchSize do
      last = operation()
      index += 1
    (platform.nowNanos() - started, last)

  private def percentile(sorted: Vector[Double], probability: Double): Double =
    if sorted.size == 1 then sorted.head
    else
      val position = probability * (sorted.size - 1).toDouble
      val lower = position.toInt
      val upper = math.min(lower + 1, sorted.size - 1)
      val fraction = position - lower.toDouble
      sorted(lower) * (1.0 - fraction) + sorted(upper) * fraction

  private[bench] def render(
      config: CourtConfig,
      platform: CourtPlatformContract,
      results: Vector[CourtResult]
  ): String =
    val rows = results.map(renderResult).mkString(",\n")
    val selectedFilter = config.caseFilter.fold("null")(value => s"\"${escape(value)}\"")
    s"""{
       |  "schema": "ravel.cross-runtime-court.v1",
       |  "metadata": {
       |    "library": "ravel",
       |    "runtime_name": "${escape(platform.runtimeName)}",
       |    "runtime_version": "${escape(platform.runtimeVersion)}",
       |    "vm_name": "${escape(platform.vmName)}",
       |    "scala_version": "${escape(CrossRuntimeCourtBuild.scalaVersion)}",
       |    "os_name": "${escape(platform.osName)}",
       |    "os_arch": "${escape(platform.osArch)}",
       |    "available_processors": ${platform.availableProcessors},
       |    "timer": "${escape(platform.timerName)}",
       |    "explicit_gc_available": ${platform.explicitGcAvailable},
       |    "mode": "${if config.signaturesOnly then "signatures_only" else "timed"}",
       |    "sides": [${config.sides.mkString(", ")}],
       |    "warmup_ms": ${config.warmupMillis},
       |    "sample_ms": ${config.sampleMillis},
       |    "samples": ${config.samples},
       |    "case_filter": $selectedFilter,
       |    "allocation_evidence": "external profiler required; use JMH gc profiler on JVM and Node --trace-gc for Scala.js",
       |    "jvm_timing_authority": "OperationMatrixBenchmarks JMH"
       |  },
       |  "results": [
       |$rows
       |  ]
       |}
       |""".stripMargin

  private def renderResult(result: CourtResult): String =
    val operation = result.operation
    val signature = result.signature
    val timingFields = result.timing match
      case Some(timing) =>
        val samples = timing.samplesNanos.map(renderNumber).mkString(", ")
        s""",
           |      "score_unit": "ns/op",
           |      "batch_size": ${timing.batchSize},
           |      "median_ns": ${renderNumber(timing.medianNanos)},
           |      "minimum_ns": ${renderNumber(timing.minimumNanos)},
           |      "maximum_ns": ${renderNumber(timing.maximumNanos)},
           |      "p25_ns": ${renderNumber(timing.p25Nanos)},
           |      "p75_ns": ${renderNumber(timing.p75Nanos)},
           |      "relative_standard_deviation": ${renderNumber(timing.relativeStandardDeviation)},
           |      "samples_ns": [$samples]""".stripMargin
      case None => ""
    s"""    {
       |      "case": "${escape(operation.name)}",
       |      "family": "${escape(operation.family)}",
       |      "side": ${result.side},
       |      "input_dtype": "${escape(operation.inputDType)}",
       |      "result_dtype": "${escape(operation.resultDType)}",
       |      "input_layout": "${escape(operation.inputLayout)}",
       |      "logical_work_units": ${operation.logicalWorkUnits},
       |      "work_unit": "element",
       |      "comparison": "${escape(operation.comparison)}",
       |      "timing_scope": "${operation.timingScope.label}",
       |      "timing_status": "${escape(result.timingStatus)}",
       |      "result_size": ${signature.resultSize},
       |      "result_layout": "${escape(signature.resultLayout)}",
       |      "sum": ${renderNumber(signature.sum)},
       |      "weighted_sum": ${renderNumber(signature.weightedSum)},
       |      "nan_count": ${signature.nanCount},
       |      "positive_infinity_count": ${signature.positiveInfinityCount},
       |      "negative_infinity_count": ${signature.negativeInfinityCount},
       |      "negative_zero_count": ${signature.negativeZeroCount}$timingFields
       |    }""".stripMargin

  private def renderNumber(value: Double): String =
    require(value.isFinite, s"court JSON cannot render non-finite number: $value")
    value.toString

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

  private[bench] def parseArgs(args: Array[String]): CourtConfig =
    var config = CourtConfig()
    var index = 0
    while index < args.length do
      args(index) match
        case "--out" =>
          index += 1
          require(index < args.length, "--out requires a path")
          config = config.copy(output = Some(args(index)))
        case "--sides" =>
          index += 1
          require(index < args.length, "--sides requires a comma-separated list")
          config = config.copy(sides = args(index).split(",").toVector.map(_.toInt))
        case "--warmup-ms" =>
          index += 1
          require(index < args.length, "--warmup-ms requires an integer")
          config = config.copy(warmupMillis = args(index).toInt)
        case "--sample-ms" =>
          index += 1
          require(index < args.length, "--sample-ms requires an integer")
          config = config.copy(sampleMillis = args(index).toInt)
        case "--samples" =>
          index += 1
          require(index < args.length, "--samples requires an integer")
          config = config.copy(samples = args(index).toInt)
        case "--case-filter" =>
          index += 1
          require(index < args.length, "--case-filter requires a substring")
          config = config.copy(caseFilter = Some(args(index)))
        case "--signatures-only" =>
          config = config.copy(signaturesOnly = true)
        case "--progress" =>
          config = config.copy(progress = true)
        case argument =>
          throw new IllegalArgumentException(s"unknown argument: $argument")
      index += 1
    config

private[bench] object CrossRuntimeCourtCases:
  val definitions: Vector[CourtCaseDefinition] =
    Vector(
      valueCase(
        "contiguous_subtract_double",
        "binary",
        "float64",
        "float64",
        "contiguous",
        fullWork
      )(fixture => fixture.doubleLeft - fixture.doubleRight),
      valueCase(
        "inner_stride_multiply_double",
        "binary",
        "float64",
        "float64",
        "inner_stride",
        halfWork
      )(fixture => fixture.innerDoubleLeft * fixture.innerDoubleRight),
      valueCase(
        "outer_stride_divide_double",
        "binary",
        "float64",
        "float64",
        "outer_stride",
        halfWork
      )(fixture => fixture.outerDoubleLeft / fixture.outerDoublePositive),
      valueCase(
        "reverse_minimum_double",
        "binary",
        "float64",
        "float64",
        "reversed",
        fullWork
      )(fixture => fixture.reversedDouble.minimum(fixture.doubleRight)),
      valueCase(
        "transpose_maximum_double",
        "binary",
        "float64",
        "float64",
        "transposed",
        fullWork
      )(fixture => fixture.transposedDouble.maximum(fixture.transposedRight)),
      valueCase(
        "broadcast_subtract_double",
        "binary",
        "float64",
        "float64",
        "broadcast",
        fullWork
      )(fixture => fixture.doubleLeft - fixture.doubleRow),
      valueCase(
        "scalar_add_double",
        "scalar",
        "float64",
        "float64",
        "contiguous",
        fullWork
      )(fixture => fixture.doubleLeft + 1.25),
      exactCase(
        "greater_equal_scalar_double",
        "comparison",
        "float64",
        "bool",
        "contiguous",
        fullWork
      )(fixture => fixture.doubleLeft >= 0.0),
      exactCase(
        "is_nan_double",
        "predicate",
        "float64",
        "bool",
        "contiguous",
        fullWork
      )(_.doubleNonFinite.isNaN),
      exactCase(
        "is_finite_double",
        "predicate",
        "float64",
        "bool",
        "contiguous",
        fullWork
      )(_.doubleNonFinite.isFinite),
      exactCase(
        "cast_double_int",
        "cast",
        "float64",
        "int32",
        "contiguous",
        fullWork
      )(_.doubleLeft.cast[Int]),
      valueCase(
        "cast_float_double",
        "cast",
        "float32",
        "float64",
        "contiguous",
        fullWork
      )(_.floatLeft.cast[Double]),
      valueCase("abs_double", "unary", "float64", "float64", "contiguous", fullWork)(
        _.doubleLeft.abs
      ),
      valueCase("sin_double", "unary", "float64", "float64", "contiguous", fullWork)(
        _.doubleLeft.sin
      ),
      valueCase(
        "full_mean_double",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        fullWork
      )(_.doubleLeft.mean),
      valueCase(
        "axis0_mean_double",
        "reduction",
        "float64",
        "float64",
        "contiguous",
        fullWork
      )(_.doubleLeft.mean(axis = 0)),
      valueCase(
        "full_sum_float",
        "reduction",
        "float32",
        "float32",
        "contiguous",
        fullWork
      )(_.floatLeft.sum),
      exactCase(
        "contiguous_add_int",
        "binary",
        "int32",
        "int32",
        "contiguous",
        fullWork
      )(fixture => fixture.intLeft + fixture.intRight),
      exactCase(
        "scalar_quot_int",
        "scalar",
        "int32",
        "int32",
        "contiguous",
        fullWork
      )(_.intPositive.quot(3)),
      exactCase(
        "full_sum_int",
        "reduction",
        "int32",
        "int32",
        "contiguous",
        fullWork
      )(_.intLeft.sum),
      inPlaceCase("inplace_multiply_double", "float64", "contiguous", fullWork)(fixture =>
        fixture.mutableMultiplyDouble.multiplyInPlace(-1.0)
        fixture.mutableMultiplyDouble(fixture.side - 1, fixture.side - 1)
      ) { fixture =>
        fixture.mutableMultiplyDouble.multiplyInPlace(-1.0)
        fixture.mutableMultiplyDouble.freezeCopy()
      },
      exactCase(
        "contiguous_add_long",
        "binary",
        "int64",
        "int64",
        "contiguous",
        fullWork,
        CourtTimingScope.JvmOnly,
        Some(64)
      )(fixture => fixture.longLeft + fixture.longRight),
      exactCase(
        "full_sum_long",
        "reduction",
        "int64",
        "int64",
        "contiguous",
        fullWork,
        CourtTimingScope.JvmOnly,
        Some(64)
      )(_.longLeft.sum)
    )

  def apply(side: Int): Vector[CourtCase] =
    definitions.filter(_.includes(side)).map(_.prepare(side))

  private def fullWork(side: Int): Long = side.toLong * side.toLong
  private def halfWork(side: Int): Long = side.toLong * side.toLong / 2L

  private def valueCase(
      name: String,
      family: String,
      inputDType: String,
      resultDType: String,
      inputLayout: String,
      work: Int => Long,
      timingScope: CourtTimingScope = CourtTimingScope.AllPlatforms,
      maximumSemanticSide: Option[Int] = None
  )(body: CourtFixture => Any): CourtCaseDefinition =
    definition(
      name,
      family,
      inputDType,
      resultDType,
      inputLayout,
      "floating",
      timingScope,
      maximumSemanticSide,
      work
    )(body, body)

  private def exactCase(
      name: String,
      family: String,
      inputDType: String,
      resultDType: String,
      inputLayout: String,
      work: Int => Long,
      timingScope: CourtTimingScope = CourtTimingScope.AllPlatforms,
      maximumSemanticSide: Option[Int] = None
  )(body: CourtFixture => Any): CourtCaseDefinition =
    definition(
      name,
      family,
      inputDType,
      resultDType,
      inputLayout,
      "exact",
      timingScope,
      maximumSemanticSide,
      work
    )(body, body)

  private def inPlaceCase(
      name: String,
      dtype: String,
      inputLayout: String,
      work: Int => Long
  )(body: CourtFixture => Any)(signatureBody: CourtFixture => Any): CourtCaseDefinition =
    definition(
      name,
      "in-place",
      dtype,
      dtype,
      inputLayout,
      "floating",
      CourtTimingScope.AllPlatforms,
      None,
      work
    )(body, signatureBody)

  private def definition(
      name: String,
      family: String,
      inputDType: String,
      resultDType: String,
      inputLayout: String,
      comparison: String,
      timingScope: CourtTimingScope,
      maximumSemanticSide: Option[Int],
      work: Int => Long
  )(
      body: CourtFixture => Any,
      signatureBody: CourtFixture => Any
  ): CourtCaseDefinition =
    CourtCaseDefinition(
      name,
      family,
      inputDType,
      resultDType,
      inputLayout,
      comparison,
      timingScope,
      maximumSemanticSide,
      side =>
        val fixture = new CourtFixture(side)
        CourtCase(
          name,
          family,
          inputDType,
          resultDType,
          inputLayout,
          work(side),
          comparison,
          timingScope,
          () => body(fixture),
          () => signatureBody(fixture)
        )
    )

private final class CourtFixture(val side: Int):
  require(side > 0 && side % 2 == 0, s"side must be a positive even integer, got $side")
  require(
    side.toLong * side.toLong <= Int.MaxValue.toLong,
    s"side squared exceeds the supported array size: $side"
  )

  lazy val doubleLeft: Array2[Double] =
    tabulateDouble { (row, column) =>
      ((row * 131 + column * 17) % 251 - 125).toDouble / 16.0
    }
  lazy val doubleRight: Array2[Double] =
    tabulateDouble { (row, column) =>
      ((row * 43 + column * 19) % 257 - 128).toDouble / 32.0
    }
  lazy val doublePositive: Array2[Double] =
    tabulateDouble { (row, column) =>
      ((row * 23 + column * 11) % 97 + 1).toDouble / 17.0
    }
  lazy val doubleNonFinite: Array2[Double] =
    tabulateDouble { (row, column) =>
      val index = row * side + column
      if index % 257 == 0 then Double.NaN
      else if index % 263 == 0 then Double.PositiveInfinity
      else doubleFixtureValue(row, column)
    }
  lazy val doubleRow: Array1[Double] =
    tabulateDoubleRow { column =>
      ((column * 29) % 127 - 63).toDouble / 8.0
    }

  lazy val floatLeft: Array2[Float] =
    tabulateFloat { (row, column) =>
      doubleFixtureValue(row, column).toFloat
    }

  lazy val intLeft: Array2[Int] =
    tabulateInt((row, column) => (row * 17 + column * 5) % 23 - 11)
  lazy val intRight: Array2[Int] =
    tabulateInt((row, column) => (row * 7 + column * 3) % 11 - 5)
  lazy val intPositive: Array2[Int] =
    tabulateInt((row, column) => (row * 3 + column * 5) % 7 + 1)

  lazy val longLeft: Array2[Long] =
    tabulateLong((row, column) => ((row * 31 + column * 7) % 41 - 20).toLong)
  lazy val longRight: Array2[Long] =
    tabulateLong((row, column) => ((row * 5 + column * 11) % 19 - 9).toLong)

  lazy val innerDoubleLeft: Array2[Double] =
    doubleLeft.slice(axis = 1, Slice(0, side, 2))
  lazy val innerDoubleRight: Array2[Double] =
    doubleRight.slice(axis = 1, Slice(1, side, 2))
  lazy val outerDoubleLeft: Array2[Double] =
    doubleLeft.slice(axis = 0, Slice(0, side, 2))
  lazy val outerDoublePositive: Array2[Double] =
    doublePositive.slice(axis = 0, Slice(1, side, 2))
  lazy val reversedDouble: Array2[Double] = doubleLeft.reverse(axis = 1)
  lazy val transposedDouble: Array2[Double] = doubleLeft.transpose
  lazy val transposedRight: Array2[Double] = doubleRight.transpose
  lazy val mutableMultiplyDouble: MutableNDArray[Double, Rank[2]] = doubleLeft.mutableCopy

  private def tabulateDouble(f: (Int, Int) => Double): Array2[Double] =
    val result = NDArray.zeros[Double](side, side)
    var row = 0
    var index = 0
    while row < side do
      var column = 0
      while column < side do
        ProbeApi.setDouble(result.storage, index, f(row, column))
        index += 1
        column += 1
      row += 1
    result

  private def tabulateDoubleRow(f: Int => Double): Array1[Double] =
    val result = NDArray.zeros[Double](side)
    var index = 0
    while index < side do
      ProbeApi.setDouble(result.storage, index, f(index))
      index += 1
    result

  private def tabulateFloat(f: (Int, Int) => Float): Array2[Float] =
    val result = NDArray.zeros[Float](side, side)
    var row = 0
    var index = 0
    while row < side do
      var column = 0
      while column < side do
        ProbeApi.setFloat(result.storage, index, f(row, column))
        index += 1
        column += 1
      row += 1
    result

  private def tabulateInt(f: (Int, Int) => Int): Array2[Int] =
    val result = NDArray.zeros[Int](side, side)
    var row = 0
    var index = 0
    while row < side do
      var column = 0
      while column < side do
        ProbeApi.setInt(result.storage, index, f(row, column))
        index += 1
        column += 1
      row += 1
    result

  private def tabulateLong(f: (Int, Int) => Long): Array2[Long] =
    val result = NDArray.zeros[Long](side, side)
    var row = 0
    var index = 0
    while row < side do
      var column = 0
      while column < side do
        ProbeApi.setLong(result.storage, index, f(row, column))
        index += 1
        column += 1
      row += 1
    result

  private def doubleFixtureValue(row: Int, column: Int): Double =
    ((row * 131 + column * 17) % 251 - 125).toDouble / 16.0
