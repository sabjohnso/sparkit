#pragma once

//
// ... Standard header files
//
#include <cstdint>
#include <cassert>
#include <unordered_set>
#include <vector>
#include <stdexcept>

//
// ... External header files
//
#include <nlohmann/json.hpp>

namespace sparkit::data::detail
{

  using size_type = std::ptrdiff_t;

  using nlohmann::json;

  using std::vector;
  using std::unordered_set;

  using std::logic_error;


} // end of namespace sparkit::data::detail
