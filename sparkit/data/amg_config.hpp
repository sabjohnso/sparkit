#pragma once

//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>

namespace sparkit::data::detail {

  // Configuration for algebraic multigrid (smoothed aggregation).

  template <typename T>
  struct Amg_config {
    T strength_threshold{0.25};
    T jacobi_weight{T{2} / T{3}};
    config::size_type pre_smoothing_steps{1};
    config::size_type post_smoothing_steps{1};
    config::size_type max_levels{25};
    config::size_type coarsest_size{100};
    T prolongation_smoothing_weight{T{4} / T{3}};
  };

  // Single level in the AMG hierarchy.

  template <typename T>
  struct Amg_level {
    Compressed_row_matrix<T> A;
    std::vector<T> inv_diag;
  };

  // Transfer operators between adjacent levels.

  template <typename T>
  struct Amg_transfer {
    Compressed_row_matrix<T> P; // Prolongation (fine → coarse)
    Compressed_row_matrix<T> R; // Restriction  (coarse → fine) = P^T
  };

  // Complete AMG hierarchy built by amg_setup.

  template <typename T>
  struct Amg_hierarchy {
    std::vector<Amg_level<T>> levels;
    std::vector<Amg_transfer<T>> transfers;
    Compressed_row_matrix<T> coarse_factor;
    Amg_config<T> config;
  };

} // end of namespace sparkit::data::detail
