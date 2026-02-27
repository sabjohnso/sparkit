#pragma once

namespace sparkit::data::detail {

  /**
   * @brief Specifies which eigenvalues to compute.
   *
   * Symmetric solvers (Lanczos) support all targets.
   * General solvers (Arnoldi) support largest/smallest_magnitude
   * and largest/smallest_real.
   */
  enum class Eigen_target {
    largest_magnitude,
    smallest_magnitude,
    largest_algebraic,
    smallest_algebraic,
    largest_real,
    smallest_real,
  };

} // end of namespace sparkit::data::detail
