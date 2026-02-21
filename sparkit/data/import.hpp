#pragma once

//
// ... Standard header files
//
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

//
// ... External header files
//
#include <nlohmann/json.hpp>

namespace sparkit::data::detail {

  using size_type = std::ptrdiff_t;

  using nlohmann::json;

  using std::unordered_set;
  using std::vector;

  using std::logic_error;

} // end of namespace sparkit::data::detail
