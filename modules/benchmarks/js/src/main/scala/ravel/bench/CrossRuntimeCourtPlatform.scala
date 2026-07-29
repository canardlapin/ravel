package ravel.bench

import scala.scalajs.js

private[bench] object CrossRuntimeCourtPlatform extends CourtPlatformContract:
  private val process = js.Dynamic.global.process

  override val runtimeName: String = "node"
  override val runtimeVersion: String = process.version.asInstanceOf[String]
  override val vmName: String =
    s"V8 ${process.versions.v8.asInstanceOf[String]}"
  override val osName: String = process.platform.asInstanceOf[String]
  override val osArch: String = process.arch.asInstanceOf[String]
  override val availableProcessors: Int =
    js.Dynamic.global.require("node:os").cpus().length.asInstanceOf[Int]
  override val timerName: String = "performance.now"
  override val isScalaJs: Boolean = true
  override val explicitGcAvailable: Boolean =
    js.typeOf(js.Dynamic.global.selectDynamic("gc")) == "function"

  override def nowNanos(): Double =
    js.Dynamic.global.performance.now().asInstanceOf[Double] * 1000000.0

  override def releaseCaseMemory(): Unit =
    if explicitGcAvailable then
      js.Dynamic.global.selectDynamic("gc").asInstanceOf[js.Function0[Unit]]()

  override def commandLineArgs(_default: Array[String]): Array[String] =
    val arguments = process.argv.asInstanceOf[js.Array[String]]
    Array.tabulate(math.max(0, arguments.length - 2))(index => arguments(index + 2))

  override def writeText(path: String, content: String): Unit =
    val fileSystem = js.Dynamic.global.require("node:fs")
    val paths = js.Dynamic.global.require("node:path")
    val parent = paths.dirname(path)
    val _ = fileSystem.mkdirSync(parent, js.Dynamic.literal(recursive = true))
    val _ = fileSystem.writeFileSync(path, content, "utf8")
