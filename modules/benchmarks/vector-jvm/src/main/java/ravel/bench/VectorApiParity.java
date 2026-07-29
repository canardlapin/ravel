package ravel.bench;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Writes correctness and runtime metadata for the nonpublished Vector API spike. */
public final class VectorApiParity {
  private VectorApiParity() {}

  public static void main(String[] args) throws IOException {
    VectorApiParityConfig config = VectorApiParityConfig.parse(args);
    List<VectorApiFixture> fixtures = new ArrayList<>();
    for (int side : config.sides) {
      fixtures.add(new VectorApiFixture(side));
    }
    Path parent = config.output.getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    Files.writeString(config.output, render(fixtures), StandardCharsets.UTF_8);
    System.out.printf(
        "Vector API parity OK: %d sides, %d-bit preferred species, %d/%d/%d double/int/long lanes; wrote %s%n",
        fixtures.size(),
        VectorApiKernels.preferredVectorBits(),
        VectorApiKernels.preferredDoubleLanes(),
        VectorApiKernels.preferredIntLanes(),
        VectorApiKernels.preferredLongLanes(),
        config.output);
  }

  private static String render(List<VectorApiFixture> fixtures) {
    StringBuilder json = new StringBuilder();
    json.append("{\n");
    json.append("  \"schema\": \"ravel.vector-api-spike.v2\",\n");
    json.append("  \"metadata\": {\n");
    field(json, "java_version", System.getProperty("java.version"), true, 4);
    field(json, "java_vendor", System.getProperty("java.vendor"), true, 4);
    field(json, "vm_name", System.getProperty("java.vm.name"), true, 4);
    field(json, "os_name", System.getProperty("os.name"), true, 4);
    field(json, "os_arch", System.getProperty("os.arch"), true, 4);
    json.append("    \"preferred_vector_bits\": ")
        .append(VectorApiKernels.preferredVectorBits())
        .append(",\n");
    json.append("    \"double_lanes\": ")
        .append(VectorApiKernels.preferredDoubleLanes())
        .append(",\n");
    json.append("    \"int_lanes\": ")
        .append(VectorApiKernels.preferredIntLanes())
        .append(",\n");
    json.append("    \"long_lanes\": ")
        .append(VectorApiKernels.preferredLongLanes())
        .append("\n");
    json.append("  },\n");
    json.append("  \"results\": [\n");
    boolean first = true;
    for (VectorApiFixture fixture : fixtures) {
      first =
          exactResult(
              json,
              first,
              fixture.side,
              "axis0_exact",
              VectorApiKernels.rawHash(fixture.scalarAxisOutput),
              VectorApiKernels.rawHash(fixture.vectorAxisOutput));

      first =
          exactDoubleResult(
              json,
              first,
              fixture.side,
              "minimum",
              VectorApiKernels.minimumScalar(fixture.left),
              VectorApiKernels.minimumVector(fixture.left));
      first =
          exactDoubleResult(
              json,
              first,
              fixture.side,
              "maximum",
              VectorApiKernels.maximumScalar(fixture.left),
              VectorApiKernels.maximumVector(fixture.left));
      first =
          exactIntegerResult(
              json,
              first,
              fixture.side,
              "sum_int",
              VectorApiKernels.sumIntScalar(fixture.intValues),
              VectorApiKernels.sumIntVector(fixture.intValues));
      first =
          exactIntegerResult(
              json,
              first,
              fixture.side,
              "product_int",
              VectorApiKernels.productIntScalar(fixture.intProduct),
              VectorApiKernels.productIntVector(fixture.intProduct));
      first =
          exactIntegerResult(
              json,
              first,
              fixture.side,
              "sum_long",
              VectorApiKernels.sumLongScalar(fixture.longValues),
              VectorApiKernels.sumLongVector(fixture.longValues));
      first =
          exactIntegerResult(
              json,
              first,
              fixture.side,
              "product_long",
              VectorApiKernels.productLongScalar(fixture.longProduct),
              VectorApiKernels.productLongVector(fixture.longProduct));

      VectorApiKernels.addScalar(
          fixture.left, fixture.right, fixture.scalarDoubleOutput);
      VectorApiKernels.addVector(
          fixture.left, fixture.right, fixture.vectorDoubleOutput);
      first =
          exactResult(
              json,
              first,
              fixture.side,
              "contiguous_add",
              VectorApiKernels.rawHash(fixture.scalarDoubleOutput),
              VectorApiKernels.rawHash(fixture.vectorDoubleOutput));

      VectorApiKernels.lessThanScalar(
          fixture.left, fixture.right, fixture.scalarBooleanOutput);
      VectorApiKernels.lessThanVector(
          fixture.left, fixture.right, fixture.vectorBooleanOutput);
      first =
          exactResult(
              json,
              first,
              fixture.side,
              "less_than",
              VectorApiKernels.booleanHash(fixture.scalarBooleanOutput),
              VectorApiKernels.booleanHash(fixture.vectorBooleanOutput));

      VectorApiKernels.isFiniteScalar(
          fixture.nonFinite, fixture.scalarBooleanOutput);
      VectorApiKernels.isFiniteVector(
          fixture.nonFinite, fixture.vectorBooleanOutput);
      first =
          exactResult(
              json,
              first,
              fixture.side,
              "is_finite",
              VectorApiKernels.booleanHash(fixture.scalarBooleanOutput),
              VectorApiKernels.booleanHash(fixture.vectorBooleanOutput));

      if (!first) {
        json.append(",\n");
      }
      first = false;
      json.append("    {\"side\": ")
          .append(fixture.side)
          .append(", \"case\": \"sin\", \"comparison\": \"absolute-tolerance\", ")
          .append("\"max_absolute_error\": ")
          .append(Double.toString(fixture.sinMaxAbsoluteError))
          .append(", \"tolerance\": 2.0e-13, \"matched\": true}");
    }
    json.append("\n  ]\n");
    json.append("}\n");
    return json.toString();
  }

  private static boolean exactResult(
      StringBuilder json,
      boolean first,
      int side,
      String caseName,
      long scalarHash,
      long vectorHash) {
    if (!first) {
      json.append(",\n");
    }
    json.append("    {\"side\": ")
        .append(side)
        .append(", \"case\": \"")
        .append(caseName)
        .append("\", \"comparison\": \"raw-bits\", \"scalar_hash\": \"")
        .append(Long.toUnsignedString(scalarHash, 16))
        .append("\", \"vector_hash\": \"")
        .append(Long.toUnsignedString(vectorHash, 16))
        .append("\", \"matched\": ")
        .append(scalarHash == vectorHash)
        .append("}");
    if (scalarHash != vectorHash) {
      throw new IllegalStateException(caseName + " hash mismatch at side " + side);
    }
    return false;
  }

  private static boolean exactDoubleResult(
      StringBuilder json,
      boolean first,
      int side,
      String caseName,
      double scalar,
      double vector) {
    VectorApiKernels.requireDoubleContractEqual(scalar, vector, caseName + " at side " + side);
    long scalarBits = Double.doubleToLongBits(scalar);
    long vectorBits = Double.doubleToLongBits(vector);
    if (!first) {
      json.append(",\n");
    }
    json.append("    {\"side\": ")
        .append(side)
        .append(", \"case\": \"")
        .append(caseName)
        .append("\", \"comparison\": \"nan-canonical-raw-bits\", \"scalar_bits\": \"")
        .append(Long.toUnsignedString(scalarBits, 16))
        .append("\", \"vector_bits\": \"")
        .append(Long.toUnsignedString(vectorBits, 16))
        .append("\", \"matched\": true}");
    return false;
  }

  private static boolean exactIntegerResult(
      StringBuilder json,
      boolean first,
      int side,
      String caseName,
      long scalar,
      long vector) {
    VectorApiKernels.requireEqual(scalar, vector, caseName + " at side " + side);
    if (!first) {
      json.append(",\n");
    }
    json.append("    {\"side\": ")
        .append(side)
        .append(", \"case\": \"")
        .append(caseName)
        .append("\", \"comparison\": \"fixed-width-exact\", \"scalar\": \"")
        .append(Long.toUnsignedString(scalar))
        .append("\", \"vector\": \"")
        .append(Long.toUnsignedString(vector))
        .append("\", \"matched\": true}");
    return false;
  }

  private static void field(
      StringBuilder json, String name, String value, boolean comma, int indent) {
    json.append(" ".repeat(indent))
        .append('"')
        .append(name)
        .append("\": \"")
        .append(escape(value))
        .append('"');
    if (comma) {
      json.append(',');
    }
    json.append('\n');
  }

  private static String escape(String value) {
    return value.replace("\\", "\\\\").replace("\"", "\\\"");
  }

}

final class VectorApiParityConfig {
  final Path output;
  final int[] sides;

  private VectorApiParityConfig(Path output, int[] sides) {
    this.output = output;
    this.sides = sides;
  }

  static VectorApiParityConfig parse(String[] args) {
    Path output = Path.of("target/vector-api/parity.json");
    int[] sides = {256, 1024};
    int index = 0;
    while (index < args.length) {
      switch (args[index]) {
        case "--out" -> {
          output = Path.of(requireValue(args, index));
          index += 2;
        }
        case "--side" -> {
          String[] tokens = requireValue(args, index).split(",");
          sides = new int[tokens.length];
          for (int sideIndex = 0; sideIndex < tokens.length; sideIndex++) {
            sides[sideIndex] = Integer.parseInt(tokens[sideIndex]);
          }
          index += 2;
        }
        default -> throw new IllegalArgumentException("unknown argument: " + args[index]);
      }
    }
    return new VectorApiParityConfig(output, sides);
  }

  private static String requireValue(String[] args, int index) {
    if (index + 1 >= args.length) {
      throw new IllegalArgumentException("missing value for " + args[index]);
    }
    return args[index + 1];
  }
}
