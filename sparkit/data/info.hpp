#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>

namespace sparkit::data::detail {

  // Neumaier's improved compensated summation.
  // Unlike plain Kahan, handles the case where the next term is
  // larger in magnitude than the running sum.
  //
  // Reference: Neumaier, A. (1974). "Rundungsfehleranalyse einiger
  // Verfahren zur Summation endlicher Summen."
  template<typename T>
  class Neumaier_sum final
  {
  public:
    void
    add(T val)
    {
      auto t = sum_ + val;
      if (std::abs(sum_) >= std::abs(val)) {
        compensation_ += (sum_ - t) + val;
      } else {
        compensation_ += (val - t) + sum_;
      }
      sum_ = t;
    }

    T
    result() const
    {
      return sum_ + compensation_;
    }

  private:
    T sum_{0};
    T compensation_{0};
  };

  template<typename T>
  std::vector<T>
  row_norms_1(Compressed_row_matrix<T> const& A)
  {
    auto rows = A.shape().row();
    auto rp = A.row_ptr();
    auto vals = A.values();

    std::vector<T> norms(static_cast<std::size_t>(rows), T{0});
    for (config::size_type i = 0; i < rows; ++i) {
      Neumaier_sum<T> acc;
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        acc.add(std::abs(vals[j]));
      }
      norms[static_cast<std::size_t>(i)] = acc.result();
    }
    return norms;
  }

  template<typename T>
  std::vector<T>
  row_norms_inf(Compressed_row_matrix<T> const& A)
  {
    auto rows = A.shape().row();
    auto rp = A.row_ptr();
    auto vals = A.values();

    std::vector<T> norms(static_cast<std::size_t>(rows), T{0});
    for (config::size_type i = 0; i < rows; ++i) {
      T mx{0};
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        mx = std::max(mx, std::abs(vals[j]));
      }
      norms[static_cast<std::size_t>(i)] = mx;
    }
    return norms;
  }

  template<typename T>
  std::vector<T>
  column_norms_1(Compressed_row_matrix<T> const& A)
  {
    auto rows = A.shape().row();
    auto cols = A.shape().column();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    auto n = static_cast<std::size_t>(cols);
    std::vector<Neumaier_sum<T>> accumulators(n);
    for (config::size_type i = 0; i < rows; ++i) {
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        accumulators[static_cast<std::size_t>(ci[j])].add(std::abs(vals[j]));
      }
    }

    std::vector<T> norms(n);
    for (std::size_t c = 0; c < n; ++c) {
      norms[c] = accumulators[c].result();
    }
    return norms;
  }

  template<typename T>
  std::vector<T>
  column_norms_inf(Compressed_row_matrix<T> const& A)
  {
    auto rows = A.shape().row();
    auto cols = A.shape().column();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<T> norms(static_cast<std::size_t>(cols), T{0});
    for (config::size_type i = 0; i < rows; ++i) {
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        auto c = static_cast<std::size_t>(ci[j]);
        norms[c] = std::max(norms[c], std::abs(vals[j]));
      }
    }
    return norms;
  }

  // Scaled Frobenius norm: avoids overflow/underflow in squaring
  // by dividing all values by the maximum absolute value first.
  // Uses Neumaier summation for the sum of scaled squares.
  template<typename T>
  T
  frobenius_norm(Compressed_row_matrix<T> const& A)
  {
    auto vals = A.values();

    if (vals.empty()) {
      return T{0};
    }

    // Find scale = max |a_ij|
    T scale{0};
    for (auto v : vals) {
      scale = std::max(scale, std::abs(v));
    }

    if (scale == T{0}) {
      return T{0};
    }

    // Sum (a_ij / scale)^2 with compensated summation
    Neumaier_sum<T> acc;
    for (auto v : vals) {
      auto scaled = v / scale;
      acc.add(scaled * scaled);
    }

    return scale * std::sqrt(acc.result());
  }

  template<typename T>
  T
  norm_1(Compressed_row_matrix<T> const& A)
  {
    auto cn = column_norms_1(A);
    return *std::max_element(cn.begin(), cn.end());
  }

  template<typename T>
  T
  norm_inf(Compressed_row_matrix<T> const& A)
  {
    auto rn = row_norms_1(A);
    return *std::max_element(rn.begin(), rn.end());
  }

  template<typename T>
  std::pair<config::size_type, config::size_type>
  bandwidth(Compressed_row_matrix<T> const& A)
  {
    auto rows = A.shape().row();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();

    config::size_type lower = 0;
    config::size_type upper = 0;

    for (config::size_type i = 0; i < rows; ++i) {
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        auto col = ci[j];
        if (i >= col) {
          lower = std::max(lower, i - col);
        }
        if (col >= i) {
          upper = std::max(upper, col - i);
        }
      }
    }
    return {lower, upper};
  }

  template<typename T>
  config::size_type
  diagonal_occupancy(Compressed_row_matrix<T> const& A)
  {
    auto rows = A.shape().row();
    auto cols = A.shape().column();
    auto diag_len = std::min(rows, cols);
    auto rp = A.row_ptr();
    auto ci = A.col_ind();

    config::size_type count = 0;
    for (config::size_type i = 0; i < diag_len; ++i) {
      auto begin = ci.begin() + rp[i];
      auto end = ci.begin() + rp[i + 1];
      if (std::binary_search(begin, end, i)) {
        ++count;
      }
    }
    return count;
  }

  template<typename T>
  std::vector<bool>
  diagonal_positions(Compressed_row_matrix<T> const& A)
  {
    auto rows = A.shape().row();
    auto cols = A.shape().column();
    auto diag_len = std::min(rows, cols);
    auto rp = A.row_ptr();
    auto ci = A.col_ind();

    std::vector<bool> pos(static_cast<std::size_t>(diag_len), false);
    for (config::size_type i = 0; i < diag_len; ++i) {
      auto begin = ci.begin() + rp[i];
      auto end = ci.begin() + rp[i + 1];
      if (std::binary_search(begin, end, i)) {
        pos[static_cast<std::size_t>(i)] = true;
      }
    }
    return pos;
  }

  template<typename T>
  std::pair<config::size_type, config::size_type>
  detect_block_size(Compressed_row_matrix<T> const& A)
  {
    auto rows = A.shape().row();
    auto cols = A.shape().column();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();

    if (A.size() == 0) {
      return {1, 1};
    }

    // Try decreasing block sizes from the largest dividing both dimensions.
    for (auto br = rows; br >= 2; --br) {
      if (rows % br != 0) continue;

      for (auto bc = cols; bc >= 2; --bc) {
        if (cols % bc != 0) continue;

        // Check: for every nonzero (i,j), all entries in its br×bc block
        // must also be structurally present.
        bool valid = true;

        for (config::size_type i = 0; i < rows && valid; ++i) {
          for (auto jj = rp[i]; jj < rp[i + 1] && valid; ++jj) {
            auto col = ci[jj];
            auto block_row_start = (i / br) * br;
            auto block_col_start = (col / bc) * bc;

            for (auto bi = block_row_start; bi < block_row_start + br && valid; ++bi) {
              for (auto bj = block_col_start; bj < block_col_start + bc && valid; ++bj) {
                auto row_begin = ci.begin() + rp[bi];
                auto row_end = ci.begin() + rp[bi + 1];
                if (!std::binary_search(row_begin, row_end, bj)) {
                  valid = false;
                }
              }
            }
          }
        }

        if (valid) {
          return {br, bc};
        }
      }
    }

    return {1, 1};
  }

} // end of namespace sparkit::data::detail
