//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>

//
// ... sparkit header files
//
#include <sparkit/data/eigen_target.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Eigen_target;

  TEST_CASE("eigen_target - all values are distinct", "[eigen_target]") {
    auto lm = static_cast<int>(Eigen_target::largest_magnitude);
    auto sm = static_cast<int>(Eigen_target::smallest_magnitude);
    auto la = static_cast<int>(Eigen_target::largest_algebraic);
    auto sa = static_cast<int>(Eigen_target::smallest_algebraic);
    auto lr = static_cast<int>(Eigen_target::largest_real);
    auto sr = static_cast<int>(Eigen_target::smallest_real);

    CHECK(lm != sm);
    CHECK(lm != la);
    CHECK(lm != sa);
    CHECK(lm != lr);
    CHECK(lm != sr);
    CHECK(sm != la);
    CHECK(sm != sa);
    CHECK(sm != lr);
    CHECK(sm != sr);
    CHECK(la != sa);
    CHECK(la != lr);
    CHECK(la != sr);
    CHECK(sa != lr);
    CHECK(sa != sr);
    CHECK(lr != sr);
  }

  TEST_CASE("eigen_target - usable in switch", "[eigen_target]") {
    auto target = Eigen_target::largest_magnitude;

    bool matched = false;
    switch (target) {
      case Eigen_target::largest_magnitude:
        matched = true;
        break;
      case Eigen_target::smallest_magnitude:
      case Eigen_target::largest_algebraic:
      case Eigen_target::smallest_algebraic:
      case Eigen_target::largest_real:
      case Eigen_target::smallest_real:
        break;
    }

    CHECK(matched);
  }

} // end of namespace sparkit::testing
