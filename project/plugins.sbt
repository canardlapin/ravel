addSbtPlugin("org.scala-js" % "sbt-scalajs" % "1.22.0")
addSbtPlugin("org.portable-scala" % "sbt-scalajs-crossproject" % "1.3.2")
addSbtPlugin("pl.project13.scala" % "sbt-jmh" % "0.4.8")
addSbtPlugin("com.github.sbt" % "sbt-ci-release" % "1.11.2")
addSbtPlugin("org.scalameta" % "sbt-scalafmt" % "2.5.4")
addSbtPlugin("com.typesafe" % "sbt-mima-plugin" % "1.1.4")
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "2.3.1")

libraryDependencies +=
  "io.github.gmkumar2005" %% "scala-js-env-playwright" % "0.1.18"
