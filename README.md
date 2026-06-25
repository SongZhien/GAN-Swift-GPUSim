Swift-Sim is a modular and hybrid GPU simulation framework.

This repository is an updated version based on the original Swift-Sim project:
https://github.com/xurongxiang/Swift-GPUSim

Compared with the original implementation, this version focuses on improving
the simulator's functionality, architectural coverage, correctness, and runtime
efficiency. The major updates include:

1. Added support for Tensor Core simulation, enabling more accurate modeling of
   workloads that rely on tensor operations.

2. Fixed multiple instruction-merging issues that could affect the correctness
   of instruction execution and performance estimation.

3. Added support for several previously unsupported instructions, improving the
   simulator's coverage of real GPU workloads.

4. Implemented support for newer GPU architectures, allowing Swift-Sim to model
   more recent hardware platforms.

5. Fixed inaccurate multi-level cache simulation behavior in some scenarios,
   improving the reliability of memory hierarchy modeling.

6. Fixed memory leak issues, improving simulator stability during long-running
   experiments.

With these updates, both simulation speed and accuracy have been improved over
the previous version.
