//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <cmath>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/amg_cycle.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Amg_config;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::amg_setup;

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

  static Compressed_row_matrix<double>
  make_tridiag(size_type n) {
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < n; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 4.0});
      if (i + 1 < n) {
        entries.push_back(Entry<double>{Index{i, i + 1}, -1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, -1.0});
      }
    }
    return make_matrix(Shape{n, n}, entries);
  }

  static Compressed_row_matrix<double>
  make_grid(size_type grid) {
    size_type const n = grid * grid;
    std::vector<Entry<double>> entries;
    for (size_type r = 0; r < grid; ++r) {
      for (size_type c = 0; c < grid; ++c) {
        auto node = r * grid + c;
        size_type degree = 0;
        if (c > 0) {
          entries.push_back(Entry<double>{Index{node, node - 1}, -1.0});
          ++degree;
        }
        if (c + 1 < grid) {
          entries.push_back(Entry<double>{Index{node, node + 1}, -1.0});
          ++degree;
        }
        if (r > 0) {
          entries.push_back(Entry<double>{Index{node, node - grid}, -1.0});
          ++degree;
        }
        if (r + 1 < grid) {
          entries.push_back(Entry<double>{Index{node, node + grid}, -1.0});
          ++degree;
        }
        entries.push_back(
          Entry<double>{Index{node, node}, static_cast<double>(degree) + 1.0});
      }
    }
    return make_matrix(Shape{n, n}, entries);
  }

  // ================================================================
  // AMG setup tests
  // ================================================================

  TEST_CASE("amg setup - 2-level hierarchy", "[amg_setup]") {
    // Small tridiag that should produce exactly 2 levels.
    auto A = make_tridiag(8);

    Amg_config<double> cfg;
    cfg.coarsest_size = 4;
    cfg.max_levels = 10;

    auto h = amg_setup(A, cfg);

    CHECK(h.levels.size() >= 2);
    CHECK(h.transfers.size() == h.levels.size() - 1);

    // Finest level is the original matrix
    CHECK(h.levels[0].A.shape().row() == 8);
    CHECK(h.levels[0].A.shape().column() == 8);

    // Coarsest level is smaller
    auto last = h.levels.size() - 1;
    CHECK(h.levels[last].A.shape().row() < 8);
  }

  TEST_CASE("amg setup - coarsening ratio", "[amg_setup]") {
    auto A = make_grid(4);

    Amg_config<double> cfg;
    cfg.coarsest_size = 4;

    auto h = amg_setup(A, cfg);

    // Each level should be strictly smaller than the previous
    for (std::size_t l = 1; l < h.levels.size(); ++l) {
      CHECK(h.levels[l].A.shape().row() < h.levels[l - 1].A.shape().row());
    }
  }

  TEST_CASE("amg setup - Galerkin operator is square", "[amg_setup]") {
    auto A = make_grid(4);

    Amg_config<double> cfg;
    cfg.coarsest_size = 4;

    auto h = amg_setup(A, cfg);

    for (auto const& level : h.levels) {
      CHECK(level.A.shape().row() == level.A.shape().column());
    }
  }

  TEST_CASE("amg setup - max_levels=1 produces single level", "[amg_setup]") {
    auto A = make_tridiag(8);

    Amg_config<double> cfg;
    cfg.max_levels = 1;

    auto h = amg_setup(A, cfg);

    CHECK(h.levels.size() == 1);
    CHECK(h.transfers.empty());
  }

  TEST_CASE("amg setup - transfer dimensions match", "[amg_setup]") {
    auto A = make_grid(4);

    Amg_config<double> cfg;
    cfg.coarsest_size = 4;

    auto h = amg_setup(A, cfg);

    for (std::size_t l = 0; l < h.transfers.size(); ++l) {
      auto fine_n = h.levels[l].A.shape().row();
      auto coarse_n = h.levels[l + 1].A.shape().row();

      // P: fine × coarse
      CHECK(h.transfers[l].P.shape().row() == fine_n);
      CHECK(h.transfers[l].P.shape().column() == coarse_n);

      // R: coarse × fine
      CHECK(h.transfers[l].R.shape().row() == coarse_n);
      CHECK(h.transfers[l].R.shape().column() == fine_n);
    }
  }

} // end of namespace sparkit::testing
