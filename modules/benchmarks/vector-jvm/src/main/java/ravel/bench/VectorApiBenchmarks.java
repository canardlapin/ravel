package ravel.bench;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Nonpublished Vector API controls.
 *
 * <p>Every vector lane represents an independent element or reduction fiber. The exact axis-0
 * kernel retains Ravel's 128-row block order and scalar merge tree; it never performs a horizontal
 * vector reduction.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 7, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 2, jvmArgsAppend = "--add-modules=jdk.incubator.vector")
@State(Scope.Thread)
public class VectorApiBenchmarks {
  @Param({"256", "1024"})
  public int side;

  private VectorApiFixture fixture;

  @Setup
  public void setup() {
    fixture = new VectorApiFixture(side);
  }

  @Benchmark
  public double[] scalar_axis0_exact() {
    VectorApiKernels.axis0ExactScalar(
        fixture.left,
        side,
        side,
        fixture.scalarAxisBlockValues,
        fixture.scalarAxisPartials,
        fixture.scalarAxisMerge,
        fixture.scalarAxisOutput);
    return fixture.scalarAxisOutput;
  }

  @Benchmark
  public double[] vector_axis0_exact() {
    VectorApiKernels.axis0ExactVector(
        fixture.left,
        side,
        side,
        fixture.vectorAxisBlockValues,
        fixture.vectorAxisPartials,
        fixture.vectorAxisMerge,
        fixture.vectorAxisOutput);
    return fixture.vectorAxisOutput;
  }

  @Benchmark
  public double[] scalar_contiguous_add() {
    VectorApiKernels.addScalar(fixture.left, fixture.right, fixture.scalarDoubleOutput);
    return fixture.scalarDoubleOutput;
  }

  @Benchmark
  public double[] vector_contiguous_add() {
    VectorApiKernels.addVector(fixture.left, fixture.right, fixture.vectorDoubleOutput);
    return fixture.vectorDoubleOutput;
  }

  @Benchmark
  public boolean[] scalar_less_than() {
    VectorApiKernels.lessThanScalar(fixture.left, fixture.right, fixture.scalarBooleanOutput);
    return fixture.scalarBooleanOutput;
  }

  @Benchmark
  public boolean[] vector_less_than() {
    VectorApiKernels.lessThanVector(fixture.left, fixture.right, fixture.vectorBooleanOutput);
    return fixture.vectorBooleanOutput;
  }

  @Benchmark
  public boolean[] scalar_is_finite() {
    VectorApiKernels.isFiniteScalar(fixture.nonFinite, fixture.scalarBooleanOutput);
    return fixture.scalarBooleanOutput;
  }

  @Benchmark
  public boolean[] vector_is_finite() {
    VectorApiKernels.isFiniteVector(fixture.nonFinite, fixture.vectorBooleanOutput);
    return fixture.vectorBooleanOutput;
  }

  @Benchmark
  public double[] scalar_sin() {
    VectorApiKernels.sinScalar(fixture.angle, fixture.scalarDoubleOutput);
    return fixture.scalarDoubleOutput;
  }

  @Benchmark
  public double[] vector_sin() {
    VectorApiKernels.sinVector(fixture.angle, fixture.vectorDoubleOutput);
    return fixture.vectorDoubleOutput;
  }
}

final class VectorApiFixture {
  final int side;
  final int size;
  final int blockCount;
  final double[] left;
  final double[] right;
  final double[] nonFinite;
  final double[] angle;

  final double[] scalarAxisBlockValues;
  final double[] scalarAxisPartials;
  final double[] scalarAxisMerge;
  final double[] scalarAxisOutput;
  final double[] vectorAxisBlockValues;
  final double[] vectorAxisPartials;
  final double[] vectorAxisMerge;
  final double[] vectorAxisOutput;

  final double[] scalarDoubleOutput;
  final double[] vectorDoubleOutput;
  final boolean[] scalarBooleanOutput;
  final boolean[] vectorBooleanOutput;
  double sinMaxAbsoluteError;

  VectorApiFixture(int side) {
    if (side <= 0) {
      throw new IllegalArgumentException("side must be positive");
    }
    this.side = side;
    this.size = Math.multiplyExact(side, side);
    this.blockCount = (side + VectorApiKernels.PAIRWISE_BLOCK_SIZE - 1)
        / VectorApiKernels.PAIRWISE_BLOCK_SIZE;
    this.left = new double[size];
    this.right = new double[size];
    this.nonFinite = new double[size];
    this.angle = new double[size];
    for (int index = 0; index < size; index++) {
      int row = index / side;
      int column = index - row * side;
      double base = Math.sin(row * 0.037 + column * 0.013);
      left[index] = base * (column % 11 == 0 ? 1.0e7 : 1.0);
      right[index] = Math.cos(row * 0.019 - column * 0.023) * 3.0;
      nonFinite[index] =
          index % 257 == 0
              ? Double.NaN
              : index % 263 == 0 ? Double.POSITIVE_INFINITY : base;
      angle[index] = ((row * 131 + column * 17) % 2048 - 1024) / 128.0;
    }

    scalarAxisBlockValues = new double[blockCount * side];
    scalarAxisPartials = new double[side];
    scalarAxisMerge = new double[Math.max(blockCount, 1)];
    scalarAxisOutput = new double[side];
    vectorAxisBlockValues = new double[blockCount * side];
    vectorAxisPartials = new double[side];
    vectorAxisMerge = new double[Math.max(blockCount, 1)];
    vectorAxisOutput = new double[side];

    scalarDoubleOutput = new double[size];
    vectorDoubleOutput = new double[size];
    scalarBooleanOutput = new boolean[size];
    vectorBooleanOutput = new boolean[size];
    validate();
  }

  private void validate() {
    VectorApiKernels.axis0ExactScalar(
        left,
        side,
        side,
        scalarAxisBlockValues,
        scalarAxisPartials,
        scalarAxisMerge,
        scalarAxisOutput);
    VectorApiKernels.axis0ExactVector(
        left,
        side,
        side,
        vectorAxisBlockValues,
        vectorAxisPartials,
        vectorAxisMerge,
        vectorAxisOutput);
    VectorApiKernels.requireRawEqual(
        scalarAxisOutput, vectorAxisOutput, "axis-0 exact schedule");

    VectorApiKernels.addScalar(left, right, scalarDoubleOutput);
    VectorApiKernels.addVector(left, right, vectorDoubleOutput);
    VectorApiKernels.requireRawEqual(
        scalarDoubleOutput, vectorDoubleOutput, "contiguous add");

    VectorApiKernels.lessThanScalar(left, right, scalarBooleanOutput);
    VectorApiKernels.lessThanVector(left, right, vectorBooleanOutput);
    VectorApiKernels.requireEqual(
        scalarBooleanOutput, vectorBooleanOutput, "less-than comparison");

    VectorApiKernels.isFiniteScalar(nonFinite, scalarBooleanOutput);
    VectorApiKernels.isFiniteVector(nonFinite, vectorBooleanOutput);
    VectorApiKernels.requireEqual(
        scalarBooleanOutput, vectorBooleanOutput, "is-finite predicate");

    VectorApiKernels.sinScalar(angle, scalarDoubleOutput);
    VectorApiKernels.sinVector(angle, vectorDoubleOutput);
    sinMaxAbsoluteError =
        VectorApiKernels.requireClose(
            scalarDoubleOutput, vectorDoubleOutput, 2.0e-13, "sin control");
  }
}

final class VectorApiKernels {
  static final int PAIRWISE_BLOCK_SIZE = 128;

  private VectorApiKernels() {}

  static int preferredVectorBits() {
    return DoubleVector.SPECIES_PREFERRED.vectorBitSize();
  }

  static int preferredDoubleLanes() {
    return DoubleVector.SPECIES_PREFERRED.length();
  }

  static void axis0ExactScalar(
      double[] source,
      int rows,
      int columns,
      double[] blockValues,
      double[] partials,
      double[] mergeScratch,
      double[] output) {
    int block = 0;
    int rowStart = 0;
    while (rowStart < rows) {
      int rowUntil = Math.min(rowStart + PAIRWISE_BLOCK_SIZE, rows);
      Arrays.fill(partials, 0.0);
      int row = rowStart;
      while (row < rowUntil) {
        int physical = row * columns;
        int column = 0;
        while (column < columns) {
          partials[column] += source[physical + column];
          column++;
        }
        row++;
      }
      System.arraycopy(partials, 0, blockValues, block * columns, columns);
      block++;
      rowStart = rowUntil;
    }
    mergeColumns(blockValues, block, columns, mergeScratch, output);
  }

  static void axis0ExactVector(
      double[] source,
      int rows,
      int columns,
      double[] blockValues,
      double[] partials,
      double[] mergeScratch,
      double[] output) {
    VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;
    int vectorBound = species.loopBound(columns);
    int vectorLength = species.length();
    int block = 0;
    int rowStart = 0;
    while (rowStart < rows) {
      int rowUntil = Math.min(rowStart + PAIRWISE_BLOCK_SIZE, rows);
      Arrays.fill(partials, 0.0);
      int row = rowStart;
      while (row < rowUntil) {
        int physical = row * columns;
        int column = 0;
        while (column < vectorBound) {
          DoubleVector sums = DoubleVector.fromArray(species, partials, column);
          DoubleVector values = DoubleVector.fromArray(species, source, physical + column);
          sums.add(values).intoArray(partials, column);
          column += vectorLength;
        }
        while (column < columns) {
          partials[column] += source[physical + column];
          column++;
        }
        row++;
      }
      System.arraycopy(partials, 0, blockValues, block * columns, columns);
      block++;
      rowStart = rowUntil;
    }
    mergeColumns(blockValues, block, columns, mergeScratch, output);
  }

  private static void mergeColumns(
      double[] blockValues,
      int blockCount,
      int columns,
      double[] mergeScratch,
      double[] output) {
    int column = 0;
    while (column < columns) {
      int block = 0;
      while (block < blockCount) {
        mergeScratch[block] = blockValues[block * columns + column];
        block++;
      }
      output[column] = merge(mergeScratch, blockCount);
      column++;
    }
  }

  private static double merge(double[] scratch, int initialCount) {
    int count = initialCount;
    while (count > 1) {
      int read = 0;
      int write = 0;
      while (read + 1 < count) {
        scratch[write] = scratch[read] + scratch[read + 1];
        read += 2;
        write++;
      }
      if (read < count) {
        scratch[write] = scratch[read];
        write++;
      }
      count = write;
    }
    return initialCount == 0 ? 0.0 : scratch[0];
  }

  static void addScalar(double[] left, double[] right, double[] output) {
    int index = 0;
    while (index < output.length) {
      output[index] = left[index] + right[index];
      index++;
    }
  }

  static void addVector(double[] left, double[] right, double[] output) {
    VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;
    int bound = species.loopBound(output.length);
    int length = species.length();
    int index = 0;
    while (index < bound) {
      DoubleVector x = DoubleVector.fromArray(species, left, index);
      DoubleVector y = DoubleVector.fromArray(species, right, index);
      x.add(y).intoArray(output, index);
      index += length;
    }
    while (index < output.length) {
      output[index] = left[index] + right[index];
      index++;
    }
  }

  static void lessThanScalar(double[] left, double[] right, boolean[] output) {
    int index = 0;
    while (index < output.length) {
      output[index] = left[index] < right[index];
      index++;
    }
  }

  static void lessThanVector(double[] left, double[] right, boolean[] output) {
    VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;
    int bound = species.loopBound(output.length);
    int length = species.length();
    int index = 0;
    while (index < bound) {
      DoubleVector x = DoubleVector.fromArray(species, left, index);
      DoubleVector y = DoubleVector.fromArray(species, right, index);
      VectorMask<Double> mask = x.compare(VectorOperators.LT, y);
      mask.intoArray(output, index);
      index += length;
    }
    while (index < output.length) {
      output[index] = left[index] < right[index];
      index++;
    }
  }

  static void isFiniteScalar(double[] input, boolean[] output) {
    int index = 0;
    while (index < output.length) {
      output[index] = Double.isFinite(input[index]);
      index++;
    }
  }

  static void isFiniteVector(double[] input, boolean[] output) {
    VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;
    int bound = species.loopBound(output.length);
    int length = species.length();
    int index = 0;
    while (index < bound) {
      DoubleVector values = DoubleVector.fromArray(species, input, index);
      values.test(VectorOperators.IS_FINITE).intoArray(output, index);
      index += length;
    }
    while (index < output.length) {
      output[index] = Double.isFinite(input[index]);
      index++;
    }
  }

  static void sinScalar(double[] input, double[] output) {
    int index = 0;
    while (index < output.length) {
      output[index] = Math.sin(input[index]);
      index++;
    }
  }

  static void sinVector(double[] input, double[] output) {
    VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;
    int bound = species.loopBound(output.length);
    int length = species.length();
    int index = 0;
    while (index < bound) {
      DoubleVector values = DoubleVector.fromArray(species, input, index);
      values.lanewise(VectorOperators.SIN).intoArray(output, index);
      index += length;
    }
    while (index < output.length) {
      output[index] = Math.sin(input[index]);
      index++;
    }
  }

  static void requireRawEqual(double[] expected, double[] actual, String label) {
    for (int index = 0; index < expected.length; index++) {
      if (Double.doubleToRawLongBits(expected[index])
          != Double.doubleToRawLongBits(actual[index])) {
        throw new IllegalStateException(label + " differs at index " + index);
      }
    }
  }

  static void requireEqual(boolean[] expected, boolean[] actual, String label) {
    if (!Arrays.equals(expected, actual)) {
      throw new IllegalStateException(label + " differs");
    }
  }

  static double requireClose(
      double[] expected, double[] actual, double tolerance, String label) {
    double maxAbsoluteError = 0.0;
    for (int index = 0; index < expected.length; index++) {
      double error = Math.abs(expected[index] - actual[index]);
      if (!Double.isFinite(error) || error > tolerance) {
        throw new IllegalStateException(
            label + " differs at index " + index + " by " + error);
      }
      maxAbsoluteError = Math.max(maxAbsoluteError, error);
    }
    return maxAbsoluteError;
  }

  static long rawHash(double[] values) {
    long hash = 0xcbf29ce484222325L;
    for (double value : values) {
      hash ^= Double.doubleToRawLongBits(value);
      hash *= 0x100000001b3L;
    }
    return hash;
  }

  static long booleanHash(boolean[] values) {
    long hash = 0xcbf29ce484222325L;
    for (boolean value : values) {
      hash ^= value ? 1L : 0L;
      hash *= 0x100000001b3L;
    }
    return hash;
  }
}
