# Sparkit

## Overview

**Sparkit** is a C++20 library for sparse linear algebra, also providing a
stable C ABI for integration into other languages. It provides multiple
storage formats, format conversion, and efficient read/write operations.
There are multiple decomposition, preconditioning, and solver techniques
included, with high-performance sequential, multi-threaded, multi-process,
and accelerated operation and built-in tools for automated performance
tuning.

The name is a nod to Yousef Saad's **SPARSKIT2**, which is a primary
reference for storage formats, format conversions, and sparse matrix
operations. Many algorithms in Sparkit are modern C++ reimplementations of
ideas found in SPARSKIT2 and the netlib ecosystem.

## Supported Element Types

Sparkit supports many element types including floating-point, fixed-point,
complex, and generic field-like types such as dual numbers and
element-wise vectors. The default value type (`double`) and size type
(`std::ptrdiff_t`) are configurable at build time via CMake.

## Key References

These are the primary reference materials for algorithms and formats:

- **SPARSKIT2** (Yousef Saad): Storage formats, format conversions, sparse
  BLAS, ILU preconditioners, matrix generation.
  https://www-users.cse.umn.edu/~saad/software/SPARSKIT/
- **Templates for the Solution of Linear Systems** (Barrett et al.):
  Iterative solvers (CG, GMRES, BiCGSTAB, etc.) and preconditioners.
  https://www.netlib.org/templates/templates.pdf
- **Matrix Market** (NIST): Exchange format for sparse matrices (`.mtx`).
  https://math.nist.gov/MatrixMarket/
- **Sparse BLAS standard**: Handle-based API for sparse matrix operations.
  https://math.nist.gov/spblas/
- **CSparse/CXSparse** (Tim Davis): Concise reference implementations of
  sparse direct methods (LU, Cholesky, QR). From "Direct Methods for
  Sparse Linear Systems" (SIAM).
- **SuiteSparse** (Tim Davis et al.): AMD, COLAMD, CHOLMOD, UMFPACK, KLU,
  SPQR. https://people.engr.tamu.edu/davis/suitesparse.html
- **Netlib sparse collection**: https://www.netlib.org/sparse/
- **Netlib linalg collection**: https://www.netlib.org/linalg/

## Architecture

### Namespace Layout

```
sparkit::config          -- Compile-time configuration (value_type, size_type, version info)
sparkit::data::detail    -- Storage formats, indices, shapes (implementation detail)
sparkit::data            -- Public facade re-exporting detail types
sparkit::utility         -- Shared utilities and exception types
```

All implementation types live in `detail` namespaces. Public headers
(`sparkit.hpp`, `data.hpp`) re-export the user-facing surface.

### Design Patterns

- **PImpl (Pointer to Implementation)**: Used for storage format classes
  (`Coordinate_sparsity`, `Compressed_row_sparsity`) to maintain ABI
  stability across the shared library boundary.
- **CSR as hub format**: Following SPARSKIT2 convention, CSR is the central
  format. Most format conversions route through CSR.
- **ADL-based JSON serialization**: `to_json`/`from_json` free functions in
  the same namespace as the type, compatible with `nlohmann::json`.

### Coding Conventions

- C++20 standard
- `#pragma once` for include guards
- Return types on their own line for non-trivial signatures
- `size_type` aliased from `config::size_type` (`std::ptrdiff_t`)
- Header comment sections: Standard headers, External headers, Sparkit headers
- Classes use `Capitalized_snake_case`
- Namespace closing comments: `// end of namespace sparkit::data::detail`

### Build System

- **CMake 3.12+** with Ninja Multi-Config generator
- **CMakePresets.json** for standardized configuration
- Strict warnings: `-Wall -Wextra -pedantic -Werror`
- Sanitizer builds: ASan + UBSan via `sanitizing-cache.cmake`
- Dependencies fetched via CMake: `nlohmann/json`, Google Test
- `cmake_utilities` git submodule for shared CMake infrastructure

### Storage Formats (Current and Planned)

Following SPARSKIT2's taxonomy, with CSR as the hub:

| Format                   | Abbreviation | Status                              |
|--------------------------|--------------|-------------------------------------|
| Coordinate               | COO          | Implemented (`Coordinate_sparsity`) |
| Compressed Sparse Row    | CSR          | Stub (`Compressed_row_sparsity`)    |
| Compressed Sparse Column | CSC          | Planned                             |
| Modified Sparse Row      | MSR          | Planned                             |
| Block Sparse Row         | BSR          | Planned                             |
| ELLPACK/ITPACK           | ELL          | Planned                             |
| Diagonal                 | DIA          | Planned                             |
| Jagged Diagonal          | JAD          | Planned                             |

### Matrix-Free Interface

Krylov solvers only need the operation `y = A*x`. The matrix-free
interface will allow users to supply a callable instead of an explicit
matrix, enabling operator-based approaches for problems where forming A
is impractical or expensive.
