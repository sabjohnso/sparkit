#include <sparkit/data/conversions.hpp>

namespace sparkit::data::detail {

  Compressed_row_sparsity
  to_compressed_row(Coordinate_sparsity const& coo)
  {
    auto idx = coo.indices();
    return Compressed_row_sparsity(coo.shape(), begin(idx), end(idx));
  }

} // end of namespace sparkit::data::detail
