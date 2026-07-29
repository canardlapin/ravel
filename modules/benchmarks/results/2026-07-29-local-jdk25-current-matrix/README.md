# Current public-operation matrix

This is the broad detector used to select the next optimization targets. It is
not an optimization receipt by itself.

The run used Homebrew OpenJDK 25.0.1 on macOS AArch64 and NumPy 2.4.3 under
Python 3.14.3. Ravel JMH used two forks, five 500 ms warmup iterations, and
seven 500 ms measurement iterations. Semantic parity passed for all 158 public
rows at sides 256 and 1024 before the timing report was generated.

The side-combined family geometric means were:

| Family | Ravel speed versus NumPy |
|---|---:|
| binary | 0.893x |
| cast | 0.912x |
| comparison | 0.530x |
| in-place | 0.812x |
| predicate | 0.277x |
| reduction | 0.701x |
| scalar | 0.909x |
| unary | 0.830x |

These aggregates identify families to investigate. They do not prove a common
cause, and individual rows with wide confidence intervals require a focused
same-runtime control before they support a change.

The matrix exposed reversed scalar mutation, predicates, full extrema, and
fixed-width reductions as the remaining large public gaps. The general
packed-layout optimization was developed and measured separately in
[`../2026-07-29-local-jdk25-frontier`](../2026-07-29-local-jdk25-frontier/).
Explicit Vector API controls live outside `ravel-core` and have separate
receipts.

Files:

- `ravel-signatures.json`: exact semantic signatures for the public rows;
- `numpy.json`: NumPy timings and signatures;
- `ravel-jmh.json`: complete Ravel JMH output; and
- `comparison.md`: parity-gated comparison, including every individual row.
