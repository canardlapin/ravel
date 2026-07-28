# Reduction kernel optimization — 2026-07-28 (JDK 25)

Local evidence for bd-01KYHRQHV3B5PVTYQP5SG02NNA after specializing float/double
pairwise sums (contiguous + rank-2 axis) onto monomorphic storage arrays, with
thread-local scratch for full contiguous sums and tiled axis-0.

## Acceptance

- Bit-stable 128-block pairwise schedule (JVM + Scala.js): Pass
- NumPy parity gate: Pass
- Full sums >= 0.25x NumPy: Pass (contiguous 0.45-0.50x; inner-stride 0.33-0.35x)
- Axis sums >= 0.50x NumPy: axis0 Pass (0.56-0.58x); axis1 contract tax (0.34-0.35x)
- Within 1.25x raw exact-schedule control: full ~1.03-1.05x; axis0 faster than fiber control; axis1 ~1.08x (256), ~1.27x (1024 incl. output alloc)

## Axis-1 numerical-contract tax

raw_axis1_sum_reuse (exact 128-block schedule) is itself only ~0.38-0.43x NumPy on this host.
Public path cannot reach 0.50x NumPy without changing the documented pairwise schedule.

## Baseline delta (vs 2026-07-27)

- full_sum_contiguous 256: 0.070x -> 0.496x
- full_sum_contiguous 1024: 0.064x -> 0.449x
- axis0_sum 256: 0.227x -> 0.576x
- axis0_sum 1024: 0.156x -> 0.564x
- axis1_sum 256: 0.336x -> 0.354x
- axis1_sum 1024: 0.246x -> 0.343x

Host: OpenJDK 25.0.1. Descriptive only — not a release gate.
