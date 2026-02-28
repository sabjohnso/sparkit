//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <cmath>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/amg_prolongation.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::multiply;
  using sparkit::data::detail::smooth_prolongation;
  using sparkit::data::detail::tentative_prolongation;

  using size_type = sparkit::config::size_type;

  static Compressed_row_matrix<double>
  make_matrix(Shape shape, std::vector<Entry<double>> const& entries) {
    std::vector<Index> indices;
    indices.reserve(entries.size());
    for (auto const& e : entries) {
      indices.push_back(e.index);
    }

    Compressed_row_sparsity sp{shape, indices.begin(), indices.end()};

    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    std::vector<double> vals(static_cast<std::size_t>(sp.size()), 0.0);

    for (auto const& e : entries) {
      auto row = e.index.row();
      auto col = e.index.column();
      for (auto p = rp[row]; p < rp[row + 1]; ++p) {
        if (ci[p] == col) {
          vals[static_cast<std::size_t>(p)] = e.value;
          break;
        }
      }
    }

    return Compressed_row_matrix<double>{std::move(sp), std::move(vals)};
  }

  // ================================================================
  // Tentative prolongation tests
  // ================================================================

  TEST_CASE("tentative prolongation - shape", "[amg_prolongation]") {
    // 6 nodes, 2 aggregates: {0,1,2} and {3,4,5}
    std::vector<size_type> agg = {0, 0, 0, 1, 1, 1};
    size_type n_agg = 2;
    size_type n = 6;

    auto P = tentative_prolongation<double>(agg, n_agg, n);

    CHECK(P.shape().row() == 6);
    CHECK(P.shape().column() == 2);
    CHECK(P.size() == 6); // one entry per row
  }

  TEST_CASE(
    "tentative prolongation - values normalized", "[amg_prolongation]") {
    // 4 nodes, 2 aggregates: {0,1} and {2,3}
    std::vector<size_type> agg = {0, 0, 1, 1};
    size_type n_agg = 2;
    size_type n = 4;

    auto P = tentative_prolongation<double>(agg, n_agg, n);

    // Each entry should be 1/sqrt(aggregate_size)
    // Aggregate 0 has 2 nodes: value = 1/sqrt(2)
    CHECK(P(0, 0) == Catch::Approx(1.0 / std::sqrt(2.0)));
    CHECK(P(1, 0) == Catch::Approx(1.0 / std::sqrt(2.0)));
    CHECK(P(2, 1) == Catch::Approx(1.0 / std::sqrt(2.0)));
    CHECK(P(3, 1) == Catch::Approx(1.0 / std::sqrt(2.0)));

    // Off-aggregate entries are zero
    CHECK(P(0, 1) == Catch::Approx(0.0));
    CHECK(P(2, 0) == Catch::Approx(0.0));
  }

  TEST_CASE(
    "tentative prolongation - preserves constant vector",
    "[amg_prolongation]") {
    // P * ones_coarse should produce a constant vector (not necessarily 1).
    std::vector<size_type> agg = {0, 0, 1, 1, 1};
    size_type n_agg = 2;
    size_type n = 5;

    auto P = tentative_prolongation<double>(agg, n_agg, n);

    // P * [1, 1]^T
    std::vector<double> ones_coarse = {1.0, 1.0};
    auto result = multiply(P, std::span<double const>{ones_coarse});

    // Within each aggregate, all nodes should get the same value
    CHECK(result[0] == Catch::Approx(result[1]));
    CHECK(result[2] == Catch::Approx(result[3]));
    CHECK(result[3] == Catch::Approx(result[4]));
  }

  // ================================================================
  // Smoothed prolongation tests
  // ================================================================

  TEST_CASE(
    "smooth prolongation - omega=0 equals tentative", "[amg_prolongation]") {
    // With omega=0, P = (I - 0*D^{-1}*A)*P_tent = P_tent.
    std::vector<size_type> agg = {0, 0, 1, 1};
    size_type n_agg = 2;
    size_type n = 4;

    auto A = make_matrix(
      Shape{n, n},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, -1.0},
       Entry<double>{Index{1, 0}, -1.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{1, 2}, -1.0},
       Entry<double>{Index{2, 1}, -1.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{2, 3}, -1.0},
       Entry<double>{Index{3, 2}, -1.0},
       Entry<double>{Index{3, 3}, 4.0}});

    auto P_tent = tentative_prolongation<double>(agg, n_agg, n);
    auto P_smooth = smooth_prolongation(A, P_tent, 0.0);

    CHECK(P_smooth.shape().row() == P_tent.shape().row());
    CHECK(P_smooth.shape().column() == P_tent.shape().column());

    for (size_type i = 0; i < n; ++i) {
      for (size_type j = 0; j < n_agg; ++j) {
        CHECK(P_smooth(i, j) == Catch::Approx(P_tent(i, j)));
      }
    }
  }

  TEST_CASE(
    "smooth prolongation - has more fill than tentative",
    "[amg_prolongation]") {
    // Smoothing should produce fill-in: non-zeros where P_tent had zeros.
    std::vector<size_type> agg = {0, 0, 1, 1};
    size_type n_agg = 2;
    size_type n = 4;

    auto A = make_matrix(
      Shape{n, n},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, -1.0},
       Entry<double>{Index{1, 0}, -1.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{1, 2}, -1.0},
       Entry<double>{Index{2, 1}, -1.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{2, 3}, -1.0},
       Entry<double>{Index{3, 2}, -1.0},
       Entry<double>{Index{3, 3}, 4.0}});

    auto P_tent = tentative_prolongation<double>(agg, n_agg, n);
    auto P_smooth = smooth_prolongation(A, P_tent, 4.0 / 3.0);

    // P_tent has 4 entries (1 per row). Smoothing adds fill.
    CHECK(P_smooth.size() > P_tent.size());
  }

  TEST_CASE("smooth prolongation - shape preserved", "[amg_prolongation]") {
    std::vector<size_type> agg = {0, 0, 0, 1, 1, 1};
    size_type n_agg = 2;
    size_type n = 6;

    // Build a 6x6 tridiag
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < n; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 4.0});
      if (i + 1 < n) {
        entries.push_back(Entry<double>{Index{i, i + 1}, -1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, -1.0});
      }
    }
    auto A = make_matrix(Shape{n, n}, entries);

    auto P_tent = tentative_prolongation<double>(agg, n_agg, n);
    auto P_smooth = smooth_prolongation(A, P_tent, 4.0 / 3.0);

    CHECK(P_smooth.shape().row() == n);
    CHECK(P_smooth.shape().column() == n_agg);
  }

} // end of namespace sparkit::testing
