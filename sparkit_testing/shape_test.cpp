//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>

//
// ... External header files
//
#include <nlohmann/json.hpp>

//
// ... sparkit header files
//
#include <sparkit/sparkit.hpp>

namespace sparkit::testing {
  using nlohmann::json;

  TEST_CASE("shape - to_json", "[shape]")
  {
    Shape shape{3, 4};
    CHECK(json({3, 4}) == json(shape));
  }

  TEST_CASE("shape - default_construction", "[shape]"){
    Shape shape{};
    CHECK(0 == shape.row());
    CHECK(0 == shape.column());
  }

} // end of namespace sparkit::testing
