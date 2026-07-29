package ravel.bench;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
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
  public double scalar_minimum() {
    return VectorApiKernels.minimumScalar(fixture.left);
  }

  @Benchmark
  public double vector_minimum() {
    return VectorApiKernels.minimumVector(fixture.left);
  }

  @Benchmark
  public double scalar_maximum() {
    return VectorApiKernels.maximumScalar(fixture.left);
  }

  @Benchmark
  public double vector_maximum() {
    return VectorApiKernels.maximumVector(fixture.left);
  }

  @Benchmark
  public int scalar_sum_int() {
    return VectorApiKernels.sumIntScalar(fixture.intValues);
  }

  @Benchmark
  public int vector_sum_int() {
    return VectorApiKernels.sumIntVector(fixture.intValues);
  }

  @Benchmark
  public int scalar_product_int() {
    return VectorApiKernels.productIntScalar(fixture.intProduct);
  }

  @Benchmark
  public int vector_product_int() {
    return VectorApiKernels.productIntVector(fixture.intProduct);
  }

  @Benchmark
  public long scalar_sum_long() {
    return VectorApiKernels.sumLongScalar(fixture.longValues);
  }

  @Benchmark
  public long vector_sum_long() {
    return VectorApiKernels.sumLongVector(fixture.longValues);
  }

  @Benchmark
  public long scalar_product_long() {
    return VectorApiKernels.productLongScalar(fixture.longProduct);
  }

  @Benchmark
  public long vector_product_long() {
    return VectorApiKernels.productLongVector(fixture.longProduct);
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
  final int[] intValues;
  final int[] intProduct;
  final long[] longValues;
  final long[] longProduct;

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
    this.intValues = new int[size];
    this.intProduct = new int[size];
    this.longValues = new long[size];
    this.longProduct = new long[size];
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
      intValues[index] =
          index % 257 == 0
              ? Integer.MAX_VALUE
              : index % 263 == 0 ? Integer.MIN_VALUE : row * 17 + column * 5 - 11;
      intProduct[index] = (row + column) % 3 == 0 ? -1 : 1;
      longValues[index] =
          index % 257 == 0
              ? Long.MAX_VALUE
              : index % 263 == 0
                  ? Long.MIN_VALUE
                  : (long) row * 1_000_003L + (long) column * 97L - 41L;
      longProduct[index] = (row + column) % 5 == 0 ? -1L : 1L;
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

    VectorApiKernels.requireDoubleContractEqual(
        VectorApiKernels.minimumScalar(left),
        VectorApiKernels.minimumVector(left),
        "minimum");
    VectorApiKernels.requireDoubleContractEqual(
        VectorApiKernels.maximumScalar(left),
        VectorApiKernels.maximumVector(left),
        "maximum");
    VectorApiKernels.requireEqual(
        VectorApiKernels.sumIntScalar(intValues),
        VectorApiKernels.sumIntVector(intValues),
        "Int sum");
    VectorApiKernels.requireEqual(
        VectorApiKernels.productIntScalar(intProduct),
        VectorApiKernels.productIntVector(intProduct),
        "Int product");
    VectorApiKernels.requireEqual(
        VectorApiKernels.sumLongScalar(longValues),
        VectorApiKernels.sumLongVector(longValues),
        "Long sum");
    VectorApiKernels.requireEqual(
        VectorApiKernels.productLongScalar(longProduct),
        VectorApiKernels.productLongVector(longProduct),
        "Long product");
    VectorApiKernels.validateAdversarialReductions();

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

  static int preferredIntLanes() {
    return IntVector.SPECIES_PREFERRED.length();
  }

  static int preferredLongLanes() {
    return LongVector.SPECIES_PREFERRED.length();
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

  static double minimumScalar(double[] input) {
    return extremumScalar(input, true);
  }

  static double maximumScalar(double[] input) {
    return extremumScalar(input, false);
  }

  private static double extremumScalar(double[] input, boolean minimum) {
    if (input.length == 0) {
      throw new IllegalArgumentException("extremum requires nonempty input");
    }
    if (input.length < 4) {
      double result = input[0];
      int index = 1;
      while (index < input.length) {
        result = minimum ? Math.min(result, input[index]) : Math.max(result, input[index]);
        index++;
      }
      return result;
    }

    double value0 = input[0];
    double value1 = input[1];
    double value2 = input[2];
    double value3 = input[3];
    int index = 4;
    while (index + 3 < input.length) {
      value0 = minimum ? Math.min(value0, input[index]) : Math.max(value0, input[index]);
      value1 =
          minimum ? Math.min(value1, input[index + 1]) : Math.max(value1, input[index + 1]);
      value2 =
          minimum ? Math.min(value2, input[index + 2]) : Math.max(value2, input[index + 2]);
      value3 =
          minimum ? Math.min(value3, input[index + 3]) : Math.max(value3, input[index + 3]);
      index += 4;
    }
    double left = minimum ? Math.min(value0, value1) : Math.max(value0, value1);
    double right = minimum ? Math.min(value2, value3) : Math.max(value2, value3);
    double result = minimum ? Math.min(left, right) : Math.max(left, right);
    while (index < input.length) {
      result = minimum ? Math.min(result, input[index]) : Math.max(result, input[index]);
      index++;
    }
    return result;
  }

  static double minimumVector(double[] input) {
    return extremumVector(input, true);
  }

  static double maximumVector(double[] input) {
    return extremumVector(input, false);
  }

  private static double extremumVector(double[] input, boolean minimum) {
    if (input.length == 0) {
      throw new IllegalArgumentException("extremum requires nonempty input");
    }
    VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;
    int length = species.length();
    int step = length * 4;
    double identity = minimum ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
    DoubleVector value0 = DoubleVector.broadcast(species, identity);
    DoubleVector value1 = value0;
    DoubleVector value2 = value0;
    DoubleVector value3 = value0;
    int index = 0;
    while (index + step <= input.length) {
      DoubleVector input0 = DoubleVector.fromArray(species, input, index);
      DoubleVector input1 = DoubleVector.fromArray(species, input, index + length);
      DoubleVector input2 = DoubleVector.fromArray(species, input, index + length * 2);
      DoubleVector input3 = DoubleVector.fromArray(species, input, index + length * 3);
      if (minimum) {
        value0 = value0.min(input0);
        value1 = value1.min(input1);
        value2 = value2.min(input2);
        value3 = value3.min(input3);
      } else {
        value0 = value0.max(input0);
        value1 = value1.max(input1);
        value2 = value2.max(input2);
        value3 = value3.max(input3);
      }
      index += step;
    }
    DoubleVector left = minimum ? value0.min(value1) : value0.max(value1);
    DoubleVector right = minimum ? value2.min(value3) : value2.max(value3);
    DoubleVector lanes = minimum ? left.min(right) : left.max(right);
    double result =
        lanes.reduceLanes(minimum ? VectorOperators.MIN : VectorOperators.MAX);
    while (index < input.length) {
      result = minimum ? Math.min(result, input[index]) : Math.max(result, input[index]);
      index++;
    }
    return result;
  }

  static int sumIntScalar(int[] input) {
    int value0 = 0;
    int value1 = 0;
    int value2 = 0;
    int value3 = 0;
    int value4 = 0;
    int value5 = 0;
    int value6 = 0;
    int value7 = 0;
    int index = 0;
    while (index + 7 < input.length) {
      value0 += input[index];
      value1 += input[index + 1];
      value2 += input[index + 2];
      value3 += input[index + 3];
      value4 += input[index + 4];
      value5 += input[index + 5];
      value6 += input[index + 6];
      value7 += input[index + 7];
      index += 8;
    }
    int result =
        ((value0 + value1) + (value2 + value3))
            + ((value4 + value5) + (value6 + value7));
    while (index < input.length) {
      result += input[index];
      index++;
    }
    return result;
  }

  static int sumIntVector(int[] input) {
    VectorSpecies<Integer> species = IntVector.SPECIES_PREFERRED;
    int length = species.length();
    int step = length * 4;
    IntVector value0 = IntVector.zero(species);
    IntVector value1 = value0;
    IntVector value2 = value0;
    IntVector value3 = value0;
    int index = 0;
    while (index + step <= input.length) {
      value0 = value0.add(IntVector.fromArray(species, input, index));
      value1 = value1.add(IntVector.fromArray(species, input, index + length));
      value2 = value2.add(IntVector.fromArray(species, input, index + length * 2));
      value3 = value3.add(IntVector.fromArray(species, input, index + length * 3));
      index += step;
    }
    int result =
        value0.add(value1).add(value2.add(value3)).reduceLanes(VectorOperators.ADD);
    while (index < input.length) {
      result += input[index];
      index++;
    }
    return result;
  }

  static int productIntScalar(int[] input) {
    int value0 = 1;
    int value1 = 1;
    int value2 = 1;
    int value3 = 1;
    int value4 = 1;
    int value5 = 1;
    int value6 = 1;
    int value7 = 1;
    int index = 0;
    while (index + 7 < input.length) {
      value0 *= input[index];
      value1 *= input[index + 1];
      value2 *= input[index + 2];
      value3 *= input[index + 3];
      value4 *= input[index + 4];
      value5 *= input[index + 5];
      value6 *= input[index + 6];
      value7 *= input[index + 7];
      index += 8;
    }
    int result =
        ((value0 * value1) * (value2 * value3))
            * ((value4 * value5) * (value6 * value7));
    while (index < input.length) {
      result *= input[index];
      index++;
    }
    return result;
  }

  static int productIntVector(int[] input) {
    VectorSpecies<Integer> species = IntVector.SPECIES_PREFERRED;
    int length = species.length();
    int step = length * 4;
    IntVector value0 = IntVector.broadcast(species, 1);
    IntVector value1 = value0;
    IntVector value2 = value0;
    IntVector value3 = value0;
    int index = 0;
    while (index + step <= input.length) {
      value0 = value0.mul(IntVector.fromArray(species, input, index));
      value1 = value1.mul(IntVector.fromArray(species, input, index + length));
      value2 = value2.mul(IntVector.fromArray(species, input, index + length * 2));
      value3 = value3.mul(IntVector.fromArray(species, input, index + length * 3));
      index += step;
    }
    int result =
        value0.mul(value1).mul(value2.mul(value3)).reduceLanes(VectorOperators.MUL);
    while (index < input.length) {
      result *= input[index];
      index++;
    }
    return result;
  }

  static long sumLongScalar(long[] input) {
    long value0 = 0L;
    long value1 = 0L;
    long value2 = 0L;
    long value3 = 0L;
    long value4 = 0L;
    long value5 = 0L;
    long value6 = 0L;
    long value7 = 0L;
    int index = 0;
    while (index + 7 < input.length) {
      value0 += input[index];
      value1 += input[index + 1];
      value2 += input[index + 2];
      value3 += input[index + 3];
      value4 += input[index + 4];
      value5 += input[index + 5];
      value6 += input[index + 6];
      value7 += input[index + 7];
      index += 8;
    }
    long result =
        ((value0 + value1) + (value2 + value3))
            + ((value4 + value5) + (value6 + value7));
    while (index < input.length) {
      result += input[index];
      index++;
    }
    return result;
  }

  static long sumLongVector(long[] input) {
    VectorSpecies<Long> species = LongVector.SPECIES_PREFERRED;
    int length = species.length();
    int step = length * 4;
    LongVector value0 = LongVector.zero(species);
    LongVector value1 = value0;
    LongVector value2 = value0;
    LongVector value3 = value0;
    int index = 0;
    while (index + step <= input.length) {
      value0 = value0.add(LongVector.fromArray(species, input, index));
      value1 = value1.add(LongVector.fromArray(species, input, index + length));
      value2 = value2.add(LongVector.fromArray(species, input, index + length * 2));
      value3 = value3.add(LongVector.fromArray(species, input, index + length * 3));
      index += step;
    }
    long result =
        value0.add(value1).add(value2.add(value3)).reduceLanes(VectorOperators.ADD);
    while (index < input.length) {
      result += input[index];
      index++;
    }
    return result;
  }

  static long productLongScalar(long[] input) {
    long value0 = 1L;
    long value1 = 1L;
    long value2 = 1L;
    long value3 = 1L;
    long value4 = 1L;
    long value5 = 1L;
    long value6 = 1L;
    long value7 = 1L;
    int index = 0;
    while (index + 7 < input.length) {
      value0 *= input[index];
      value1 *= input[index + 1];
      value2 *= input[index + 2];
      value3 *= input[index + 3];
      value4 *= input[index + 4];
      value5 *= input[index + 5];
      value6 *= input[index + 6];
      value7 *= input[index + 7];
      index += 8;
    }
    long result =
        ((value0 * value1) * (value2 * value3))
            * ((value4 * value5) * (value6 * value7));
    while (index < input.length) {
      result *= input[index];
      index++;
    }
    return result;
  }

  static long productLongVector(long[] input) {
    VectorSpecies<Long> species = LongVector.SPECIES_PREFERRED;
    int length = species.length();
    int step = length * 4;
    LongVector value0 = LongVector.broadcast(species, 1L);
    LongVector value1 = value0;
    LongVector value2 = value0;
    LongVector value3 = value0;
    int index = 0;
    while (index + step <= input.length) {
      value0 = value0.mul(LongVector.fromArray(species, input, index));
      value1 = value1.mul(LongVector.fromArray(species, input, index + length));
      value2 = value2.mul(LongVector.fromArray(species, input, index + length * 2));
      value3 = value3.mul(LongVector.fromArray(species, input, index + length * 3));
      index += step;
    }
    long result =
        value0.mul(value1).mul(value2.mul(value3)).reduceLanes(VectorOperators.MUL);
    while (index < input.length) {
      result *= input[index];
      index++;
    }
    return result;
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

  static void validateAdversarialReductions() {
    double payloadNaN = Double.longBitsToDouble(0x7ff8_0000_0000_0042L);
    double[][] extremaCases = {
      {0.0},
      {-0.0},
      {0.0, -0.0},
      {-0.0, 0.0},
      {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY},
      {3.0, Double.NaN, -7.0},
      {payloadNaN, 1.0, -1.0},
      {4.0, -3.0, 2.0, -1.0, 0.0, -0.0, 8.0}
    };
    for (int index = 0; index < extremaCases.length; index++) {
      double[] values = extremaCases[index];
      requireDoubleContractEqual(
          minimumScalar(values), minimumVector(values), "adversarial minimum " + index);
      requireDoubleContractEqual(
          maximumScalar(values), maximumVector(values), "adversarial maximum " + index);
    }

    int[] intTail = new int[preferredIntLanes() * 4 + 3];
    for (int index = 0; index < intTail.length; index++) {
      intTail[index] =
          index % 5 == 0 ? Integer.MAX_VALUE : index % 3 == 0 ? Integer.MIN_VALUE : index * 17 + 3;
    }
    requireEqual(sumIntScalar(intTail), sumIntVector(intTail), "adversarial Int sum tail");
    requireEqual(
        productIntScalar(intTail), productIntVector(intTail), "adversarial Int product tail");
    requireEqual(sumIntScalar(new int[0]), sumIntVector(new int[0]), "empty Int sum");
    requireEqual(productIntScalar(new int[0]), productIntVector(new int[0]), "empty Int product");

    long[] longTail = new long[preferredLongLanes() * 4 + 3];
    for (int index = 0; index < longTail.length; index++) {
      longTail[index] =
          index % 5 == 0
              ? Long.MAX_VALUE
              : index % 3 == 0 ? Long.MIN_VALUE : (long) index * 1_000_003L + 5L;
    }
    requireEqual(sumLongScalar(longTail), sumLongVector(longTail), "adversarial Long sum tail");
    requireEqual(
        productLongScalar(longTail), productLongVector(longTail), "adversarial Long product tail");
    requireEqual(sumLongScalar(new long[0]), sumLongVector(new long[0]), "empty Long sum");
    requireEqual(
        productLongScalar(new long[0]), productLongVector(new long[0]), "empty Long product");
  }

  static void requireDoubleContractEqual(double expected, double actual, String label) {
    boolean equal =
        Double.isNaN(expected)
            ? Double.isNaN(actual)
            : Double.doubleToRawLongBits(expected) == Double.doubleToRawLongBits(actual);
    if (!equal) {
      throw new IllegalStateException(
          label
              + " differs: expected 0x"
              + Long.toUnsignedString(Double.doubleToRawLongBits(expected), 16)
              + ", actual 0x"
              + Long.toUnsignedString(Double.doubleToRawLongBits(actual), 16));
    }
  }

  static void requireEqual(int expected, int actual, String label) {
    if (expected != actual) {
      throw new IllegalStateException(
          label + " differs: expected " + expected + ", actual " + actual);
    }
  }

  static void requireEqual(long expected, long actual, String label) {
    if (expected != actual) {
      throw new IllegalStateException(
          label + " differs: expected " + expected + ", actual " + actual);
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
