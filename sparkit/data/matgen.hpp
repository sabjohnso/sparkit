#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/Entry.hpp>

namespace sparkit::data::detail {

  // Build a CSR matrix from a list of (index, value) entries.
  //
  // This is the foundation function that replaces the duplicated
  // make_matrix helpers scattered across 29+ test files.
  template <typename T = config::value_type>
  Compressed_row_matrix<T>
  make_matrix(Shape shape, std::vector<Entry<T>> const& entries) {
    std::vector<Index> indices;
    indices.reserve(entries.size());
    for (auto const& e : entries) {
      indices.push_back(e.index);
    }

    Compressed_row_sparsity sp{shape, indices.begin(), indices.end()};

    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    std::vector<T> vals(static_cast<std::size_t>(sp.size()), T{0});

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

    return Compressed_row_matrix<T>{std::move(sp), std::move(vals)};
  }

  // Build an n x n diagonal matrix from a vector of diagonal values.
  template <typename T = config::value_type>
  Compressed_row_matrix<T>
  diagonal_matrix(std::vector<T> const& diag) {
    auto n = static_cast<config::size_type>(diag.size());

    std::vector<Entry<T>> entries;
    entries.reserve(diag.size());
    for (config::size_type i = 0; i < n; ++i) {
      entries.push_back(
        Entry<T>{Index{i, i}, diag[static_cast<std::size_t>(i)]});
    }

    return make_matrix(Shape{n, n}, entries);
  }

  // Build an n x n tridiagonal matrix with given sub-diagonal,
  // diagonal, and super-diagonal values.
  template <typename T = config::value_type>
  Compressed_row_matrix<T>
  tridiagonal_matrix(config::size_type n, T sub, T diag, T super) {
    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(3 * n - 2));

    for (config::size_type i = 0; i < n; ++i) {
      if (i > 0) { entries.push_back(Entry<T>{Index{i, i - 1}, sub}); }
      entries.push_back(Entry<T>{Index{i, i}, diag});
      if (i + 1 < n) { entries.push_back(Entry<T>{Index{i, i + 1}, super}); }
    }

    return make_matrix(Shape{n, n}, entries);
  }

  // Build an n x n arrow matrix: dense first row/column plus diagonal.
  //
  // The diagonal entries all have value `diag`, and the off-diagonal
  // entries in the first row and first column have value `arrow`.
  template <typename T = config::value_type>
  Compressed_row_matrix<T>
  arrow_matrix(config::size_type n, T diag, T arrow) {
    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(3 * n - 2));

    // First row
    entries.push_back(Entry<T>{Index{0, 0}, diag});
    for (config::size_type j = 1; j < n; ++j) {
      entries.push_back(Entry<T>{Index{0, j}, arrow});
    }

    // Remaining rows: arrow in column 0, diagonal
    for (config::size_type i = 1; i < n; ++i) {
      entries.push_back(Entry<T>{Index{i, 0}, arrow});
      entries.push_back(Entry<T>{Index{i, i}, diag});
    }

    return make_matrix(Shape{n, n}, entries);
  }

  // 5-point finite-difference Laplacian on an nx x ny grid.
  //
  // Produces the standard discrete Laplacian where the diagonal entry
  // equals the number of grid neighbors (2, 3, or 4) and off-diagonal
  // entries connecting neighbors are -1. The resulting matrix is SPD.
  //
  // Analog of SPARSKIT2 GEN57PT for the 5-point case.
  template <typename T = config::value_type>
  Compressed_row_matrix<T>
  poisson_2d(config::size_type nx, config::size_type ny) {
    auto const n = nx * ny;

    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(5 * n));

    for (config::size_type r = 0; r < ny; ++r) {
      for (config::size_type c = 0; c < nx; ++c) {
        auto node = r * nx + c;
        config::size_type degree = 0;

        if (c > 0) {
          entries.push_back(Entry<T>{Index{node, node - 1}, T{-1}});
          ++degree;
        }
        if (c + 1 < nx) {
          entries.push_back(Entry<T>{Index{node, node + 1}, T{-1}});
          ++degree;
        }
        if (r > 0) {
          entries.push_back(Entry<T>{Index{node, node - nx}, T{-1}});
          ++degree;
        }
        if (r + 1 < ny) {
          entries.push_back(Entry<T>{Index{node, node + nx}, T{-1}});
          ++degree;
        }

        entries.push_back(Entry<T>{Index{node, node}, static_cast<T>(degree)});
      }
    }

    return make_matrix(Shape{n, n}, entries);
  }

  // Square grid shortcut for poisson_2d.
  template <typename T = config::value_type>
  Compressed_row_matrix<T>
  poisson_2d(config::size_type grid) {
    return poisson_2d<T>(grid, grid);
  }

  // 2D convection-diffusion operator on an nx x ny grid.
  //
  // Combines diffusion (5-point Laplacian scaled by `diffusion`)
  // with first-order upwind convection in x and y directions.
  // With zero convection, this reduces to diffusion * poisson_2d.
  //
  // Analog of SPARSKIT2 GEN57PT with convection terms.
  template <typename T = config::value_type>
  Compressed_row_matrix<T>
  convection_diffusion_2d(
    config::size_type nx,
    config::size_type ny,
    T diffusion,
    T convection_x,
    T convection_y) {
    auto const n = nx * ny;

    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(5 * n));

    for (config::size_type r = 0; r < ny; ++r) {
      for (config::size_type c = 0; c < nx; ++c) {
        auto node = r * nx + c;
        T diag_val{0};

        // West neighbor (c-1)
        if (c > 0) {
          auto val = -diffusion - convection_x;
          entries.push_back(Entry<T>{Index{node, node - 1}, val});
          diag_val -= val;
        }

        // East neighbor (c+1)
        if (c + 1 < nx) {
          auto val = -diffusion;
          entries.push_back(Entry<T>{Index{node, node + 1}, val});
          diag_val -= val;
        }

        // South neighbor (r-1)
        if (r > 0) {
          auto val = -diffusion - convection_y;
          entries.push_back(Entry<T>{Index{node, node - nx}, val});
          diag_val -= val;
        }

        // North neighbor (r+1)
        if (r + 1 < ny) {
          auto val = -diffusion;
          entries.push_back(Entry<T>{Index{node, node + nx}, val});
          diag_val -= val;
        }

        entries.push_back(Entry<T>{Index{node, node}, diag_val});
      }
    }

    return make_matrix(Shape{n, n}, entries);
  }

  // Generate a random sparse n x n matrix with diagonal dominance.
  //
  // Each row has up to `nnz_per_row` entries (including the diagonal).
  // Off-diagonal entries are drawn from U(-1, 1), then the diagonal is
  // set to the absolute row sum + 1 to ensure strict diagonal dominance
  // (and hence nonsingularity).
  //
  // Uses an explicit seed for reproducibility.
  // Analog of SPARSKIT2 MATRF2.
  template <typename T = config::value_type>
  Compressed_row_matrix<T>
  random_sparse(
    config::size_type n, config::size_type nnz_per_row, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> val_dist(T{-1}, T{1});
    std::uniform_int_distribution<config::size_type> col_dist(0, n - 1);

    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(n * nnz_per_row));

    for (config::size_type i = 0; i < n; ++i) {
      T off_diag_sum{0};

      // Generate off-diagonal entries
      std::vector<config::size_type> cols_used;
      cols_used.push_back(i); // reserve diagonal

      for (config::size_type k = 1; k < nnz_per_row; ++k) {
        auto col = col_dist(gen);
        // Skip duplicates and diagonal
        if (
          std::find(cols_used.begin(), cols_used.end(), col) !=
          cols_used.end()) {
          continue;
        }
        cols_used.push_back(col);

        auto val = val_dist(gen);
        entries.push_back(Entry<T>{Index{i, col}, val});
        off_diag_sum += std::abs(val);
      }

      // Diagonal: strictly dominant
      entries.push_back(Entry<T>{Index{i, i}, off_diag_sum + T{1}});
    }

    return make_matrix(Shape{n, n}, entries);
  }

  // Generate a system A, x_exact, b where A * x_exact = b.
  //
  // Builds a diagonally-dominant SPD tridiagonal system, sets
  // x_exact = [1, 2, ..., n], and computes b = A * x_exact.
  // Useful for verifying solver correctness.
  template <typename T = config::value_type>
  Compressed_row_matrix<T>
  manufactured_solution(
    config::size_type n, std::vector<T>& x_exact, std::vector<T>& b) {
    auto A = tridiagonal_matrix<T>(n, T{-1}, T{4}, T{-1});

    x_exact.resize(static_cast<std::size_t>(n));
    for (config::size_type i = 0; i < n; ++i) {
      x_exact[static_cast<std::size_t>(i)] = static_cast<T>(i + 1);
    }

    // Compute b = A * x_exact
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    b.resize(static_cast<std::size_t>(n));
    for (config::size_type i = 0; i < n; ++i) {
      T sum{0};
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        sum += vals[j] * x_exact[static_cast<std::size_t>(ci[j])];
      }
      b[static_cast<std::size_t>(i)] = sum;
    }

    return A;
  }

} // end of namespace sparkit::data::detail
