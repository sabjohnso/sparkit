#pragma once

//
// ... sparkit header files
//
#include <sparkit/data/conditioning.hpp>
#include <sparkit/data/diagnostics.hpp>
#include <sparkit/data/Entry.hpp>
#include <sparkit/data/Shape.hpp>

namespace sparkit::data {
  using ::sparkit::data::detail::Entry;
  using ::sparkit::data::detail::Shape;

  // conditioning
  using ::sparkit::data::detail::estimate_condition_1;
  using ::sparkit::data::detail::estimate_eigenvalue_bounds;
  using ::sparkit::data::detail::estimate_norm_1_inverse;
  using ::sparkit::data::detail::estimate_spectral_radius;

  // diagnostics
  using ::sparkit::data::detail::column_dominance_ratios;
  using ::sparkit::data::detail::is_column_diagonally_dominant;
  using ::sparkit::data::detail::is_numerically_symmetric;
  using ::sparkit::data::detail::is_positive_definite;
  using ::sparkit::data::detail::is_row_diagonally_dominant;
  using ::sparkit::data::detail::is_strictly_column_diagonally_dominant;
  using ::sparkit::data::detail::is_strictly_row_diagonally_dominant;
  using ::sparkit::data::detail::is_structurally_symmetric;
  using ::sparkit::data::detail::row_dominance_ratios;
  using ::sparkit::data::detail::spy;
  using ::sparkit::data::detail::spy_svg;

} // end of namespace sparkit::data
