package ravel.bench

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

private[bench] object CrossRuntimeCourtPlatform extends CourtPlatformContract:
  override val runtimeName: String = "jvm"
  override val runtimeVersion: String = System.getProperty("java.version")
  override val vmName: String = System.getProperty("java.vm.name")
  override val osName: String = System.getProperty("os.name")
  override val osArch: String = System.getProperty("os.arch")
  override val availableProcessors: Int = Runtime.getRuntime.availableProcessors()
  override val timerName: String = "System.nanoTime"
  override val isScalaJs: Boolean = false
  override val explicitGcAvailable: Boolean = false

  override def nowNanos(): Double =
    System.nanoTime().toDouble

  override def releaseCaseMemory(): Unit = ()

  override def commandLineArgs(default: Array[String]): Array[String] =
    default

  override def writeText(path: String, content: String): Unit =
    val output = Paths.get(path)
    val parent = output.toAbsolutePath.getParent
    if parent != null then
      val _ = Files.createDirectories(parent)
    val _ = Files.writeString(output, content, StandardCharsets.UTF_8)
