package ravel.bench

final class CrossRuntimeCourtSuite extends munit.FunSuite:
  private object SignaturePlatform extends CourtPlatformContract:
    override val runtimeName: String = "test"
    override val runtimeVersion: String = "1"
    override val vmName: String = "test-vm"
    override val osName: String = "test-os"
    override val osArch: String = "test-arch"
    override val availableProcessors: Int = 1
    override val timerName: String = "test-clock"
    override val isScalaJs: Boolean = false
    override val explicitGcAvailable: Boolean = false
    override def nowNanos(): Double = 0.0
    override def releaseCaseMemory(): Unit = ()
    override def commandLineArgs(default: Array[String]): Array[String] = default
    override def writeText(_path: String, _content: String): Unit = ()

  test("registry spans public operation, dtype, and layout dimensions") {
    val cases = CrossRuntimeCourtCases(4)
    assertEquals(cases.size, 23)
    assertEquals(cases.map(_.name).distinct.size, cases.size)
    assertEquals(
      cases.map(_.family).toSet,
      Set("binary", "scalar", "unary", "comparison", "predicate", "cast", "reduction", "in-place")
    )
    assertEquals(cases.map(_.inputDType).toSet, Set("float64", "float32", "int32", "int64"))
    assertEquals(
      cases.map(_.inputLayout).toSet,
      Set("contiguous", "inner_stride", "outer_stride", "reversed", "transposed", "broadcast")
    )
    assertEquals(
      cases.filter(_.timingScope == CourtTimingScope.JvmOnly).map(_.inputDType).toSet,
      Set("int64")
    )
    assertEquals(CrossRuntimeCourtCases(256).size, 21)
    assertEquals(
      CrossRuntimeCourtCases(256).map(_.inputDType).toSet,
      Set("float64", "float32", "int32")
    )
  }

  test("default court uses three distinct generality sizes") {
    val config = CourtConfig().validate()
    assertEquals(config.sides, Vector(64, 256, 1024))
    intercept[IllegalArgumentException](CourtConfig(sides = Vector(3)).validate())
    intercept[IllegalArgumentException](CourtConfig(sides = Vector(4, 4)).validate())
  }

  test("semantic signatures are deterministic for every shared case") {
    val first = CrossRuntimeCourtCases(4).map { operation =>
      operation.name -> CrossRuntimeCourt.signatureOf(operation.signature())
    }
    val second = CrossRuntimeCourtCases(4).map { operation =>
      operation.name -> CrossRuntimeCourt.signatureOf(operation.signature())
    }
    assertEquals(first, second)
  }

  test("non-finite predicates carry exact finite signatures") {
    val results = CrossRuntimeCourt.execute(
      CourtConfig(sides = Vector(4), caseFilter = Some("is_"), signaturesOnly = true),
      SignaturePlatform
    )
    assertEquals(results.map(_.operation.name).toSet, Set("is_nan_double", "is_finite_double"))
    results.foreach { result =>
      assertEquals(result.signature.nanCount, 0)
      assertEquals(result.signature.positiveInfinityCount, 0)
      assertEquals(result.signature.negativeInfinityCount, 0)
      assertEquals(result.timingStatus, "not_requested")
      assertEquals(result.timing, None)
    }
  }

  test("signature-only JSON is machine-readable in shape and contains no timing score") {
    val config = CourtConfig(sides = Vector(4), signaturesOnly = true)
    val results = CrossRuntimeCourt.execute(config, SignaturePlatform)
    val rendered = CrossRuntimeCourt.render(config, SignaturePlatform, results)
    assert(rendered.contains("\"schema\": \"ravel.cross-runtime-court.v1\""))
    assert(rendered.contains("\"mode\": \"signatures_only\""))
    assert(rendered.contains("\"case\": \"contiguous_subtract_double\""))
    assert(rendered.contains("\"negative_zero_count\""))
    assert(!rendered.contains("\"median_ns\""))
  }

  test("argument parser rejects unknown or incomplete options") {
    assertEquals(
      CrossRuntimeCourt.parseArgs(
        Array("--sides", "4,8,16", "--samples", "3", "--signatures-only", "--progress")
      ),
      CourtConfig(
        sides = Vector(4, 8, 16),
        samples = 3,
        signaturesOnly = true,
        progress = true
      )
    )
    intercept[IllegalArgumentException](CrossRuntimeCourt.parseArgs(Array("--unknown")))
    intercept[IllegalArgumentException](CrossRuntimeCourt.parseArgs(Array("--out")))
  }
