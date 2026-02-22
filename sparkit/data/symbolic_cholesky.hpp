#pragma once

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_sparsity.hpp>

namespace sparkit::data::detail {

  Compressed_row_sparsity
  symbolic_cholesky(Compressed_row_sparsity const& sp);

} // end of namespace sparkit::data::detail
