//
// ... Test header files
//
#include <gtest/gtest.h>

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

  TEST(shape, to_json)
  {
    Shape shape{3, 4};
    EXPECT_EQ(json({3, 4}), json(shape));
  }

  TEST(shape, default_construction){
    Shape shape{};
    EXPECT_EQ(0, shape.row());
    EXPECT_EQ(0, shape.column());
  }

} // end of namespace sparkit::testing
