#pragma once

//
// ... sparkit header files
//
#include <sparkit/data/Coordinate_sparsity.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>

namespace sparkit::data::detail {

  Compressed_row_sparsity
  to_compressed_row(Coordinate_sparsity const& coo);

} // end of namespace sparkit::data::detail
