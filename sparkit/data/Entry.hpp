#pragma once

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Index.hpp>

namespace sparkit::data::detail {

  template<typename T = config::value_type>
  struct Entry
  {
    Index index;
    T value;
  };

} // end of namespace sparkit::data::detail
