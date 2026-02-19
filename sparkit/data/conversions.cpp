#include <sparkit/data/conversions.hpp>

namespace sparkit::data::detail {

  Compressed_row_sparsity
  to_compressed_row(Coordinate_sparsity const& coo)
  {
    auto idx = coo.indices();
    return Compressed_row_sparsity(coo.shape(), begin(idx), end(idx));
  }

  Compressed_column_sparsity
  to_compressed_column(Compressed_row_sparsity const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto nrow = csr.shape().row();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csr.size()));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        indices.push_back(Index{row, ci[j]});
      }
    }

    return Compressed_column_sparsity(csr.shape(), indices.begin(), indices.end());
  }

  Compressed_row_sparsity
  to_compressed_row(Compressed_column_sparsity const& csc)
  {
    auto cp = csc.col_ptr();
    auto ri = csc.row_ind();
    auto ncol = csc.shape().column();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csc.size()));

    for (size_type col = 0; col < ncol; ++col) {
      for (auto j = cp[col]; j < cp[col + 1]; ++j) {
        indices.push_back(Index{ri[j], col});
      }
    }

    return Compressed_row_sparsity(csc.shape(), indices.begin(), indices.end());
  }

  Modified_sparse_row_sparsity
  to_modified_sparse_row(Compressed_row_sparsity const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto nrow = csr.shape().row();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csr.size()));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        indices.push_back(Index{row, ci[j]});
      }
    }

    return Modified_sparse_row_sparsity(csr.shape(), indices.begin(), indices.end());
  }

  Compressed_row_sparsity
  to_compressed_row(Modified_sparse_row_sparsity const& msr)
  {
    auto shape = msr.shape();
    auto nrow = shape.row();
    auto diag_len = msr.diagonal_length();
    auto od_rp = msr.off_diagonal_row_ptr();
    auto od_ci = msr.off_diagonal_col_ind();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(msr.size()));

    for (size_type row = 0; row < nrow; ++row) {
      if (row < diag_len && msr.has_diagonal(row)) {
        indices.push_back(Index{row, row});
      }
      for (auto j = od_rp[row]; j < od_rp[row + 1]; ++j) {
        indices.push_back(Index{row, od_ci[j]});
      }
    }

    return Compressed_row_sparsity(shape, indices.begin(), indices.end());
  }

  Diagonal_sparsity
  to_diagonal(Compressed_row_sparsity const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto nrow = csr.shape().row();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csr.size()));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        indices.push_back(Index{row, ci[j]});
      }
    }

    return Diagonal_sparsity(csr.shape(), indices.begin(), indices.end());
  }

  Compressed_row_sparsity
  to_compressed_row(Diagonal_sparsity const& dia)
  {
    auto shape = dia.shape();
    auto nrow = shape.row();
    auto ncol = shape.column();
    auto off = dia.offsets();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(dia.size()));

    for (auto offset : off) {
      size_type start_row, start_col, len;
      if (offset >= 0) {
        start_row = 0;
        start_col = offset;
        len = std::min(nrow, ncol - offset);
      } else {
        start_row = -offset;
        start_col = 0;
        len = std::min(nrow + offset, ncol);
      }
      for (size_type k = 0; k < len; ++k) {
        indices.push_back(Index{start_row + k, start_col + k});
      }
    }

    return Compressed_row_sparsity(shape, indices.begin(), indices.end());
  }

  Ellpack_sparsity
  to_ellpack(Compressed_row_sparsity const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto nrow = csr.shape().row();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csr.size()));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        indices.push_back(Index{row, ci[j]});
      }
    }

    return Ellpack_sparsity(csr.shape(), indices.begin(), indices.end());
  }

  Compressed_row_sparsity
  to_compressed_row(Ellpack_sparsity const& ell)
  {
    auto shape = ell.shape();
    auto nrow = shape.row();
    auto max_nnz = ell.max_nnz_per_row();
    auto ci = ell.col_ind();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(ell.size()));

    for (size_type row = 0; row < nrow; ++row) {
      auto base = static_cast<std::size_t>(row * max_nnz);
      for (size_type k = 0; k < max_nnz; ++k) {
        auto col = ci[base + static_cast<std::size_t>(k)];
        if (col == -1) break;
        indices.push_back(Index{row, col});
      }
    }

    return Compressed_row_sparsity(shape, indices.begin(), indices.end());
  }

  Block_sparse_row_sparsity
  to_block_sparse_row(Compressed_row_sparsity const& csr,
                      size_type block_rows, size_type block_cols)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto nrow = csr.shape().row();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csr.size()));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        indices.push_back(Index{row, ci[j]});
      }
    }

    return Block_sparse_row_sparsity(
      csr.shape(), block_rows, block_cols, indices.begin(), indices.end());
  }

  Compressed_row_sparsity
  to_compressed_row(Block_sparse_row_sparsity const& bsr)
  {
    auto shape = bsr.shape();
    auto nrow = shape.row();
    auto ncol = shape.column();
    auto br = bsr.block_rows();
    auto bc = bsr.block_cols();
    auto rp = bsr.row_ptr();
    auto ci = bsr.col_ind();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(bsr.size()));

    auto nbr = bsr.num_block_rows();
    for (size_type block_row = 0; block_row < nbr; ++block_row) {
      for (auto j = rp[block_row]; j < rp[block_row + 1]; ++j) {
        auto block_col = ci[j];
        for (size_type lr = 0; lr < br; ++lr) {
          for (size_type lc = 0; lc < bc; ++lc) {
            auto row = block_row * br + lr;
            auto col = block_col * bc + lc;
            if (row < nrow && col < ncol) {
              indices.push_back(Index{row, col});
            }
          }
        }
      }
    }

    return Compressed_row_sparsity(shape, indices.begin(), indices.end());
  }

  Jagged_diagonal_sparsity
  to_jagged_diagonal(Compressed_row_sparsity const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto nrow = csr.shape().row();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csr.size()));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        indices.push_back(Index{row, ci[j]});
      }
    }

    return Jagged_diagonal_sparsity(csr.shape(), indices.begin(), indices.end());
  }

  Compressed_row_sparsity
  to_compressed_row(Jagged_diagonal_sparsity const& jad)
  {
    auto shape = jad.shape();
    auto pm = jad.perm();
    auto jd = jad.jdiag();
    auto ci = jad.col_ind();
    auto num_jdiags = std::ssize(jd) - 1;

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(jad.size()));

    for (size_type k = 0; k < num_jdiags; ++k) {
      auto width = jd[k + 1] - jd[k];
      for (size_type i = 0; i < width; ++i) {
        auto orig_row = pm[i];
        indices.push_back(Index{orig_row, ci[jd[k] + i]});
      }
    }

    return Compressed_row_sparsity(shape, indices.begin(), indices.end());
  }

  Compressed_row_sparsity
  to_compressed_row(Symmetric_compressed_row_sparsity const& scsr)
  {
    auto rp = scsr.row_ptr();
    auto ci = scsr.col_ind();
    auto shape = scsr.shape();
    auto nrow = shape.row();

    // Expand lower triangle to full: emit (i,j) and (j,i) for off-diagonal
    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(scsr.size() * 2));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        auto col = ci[j];
        indices.push_back(Index{row, col});
        if (row != col) {
          indices.push_back(Index{col, row});
        }
      }
    }

    return Compressed_row_sparsity(shape, indices.begin(), indices.end());
  }

  Symmetric_compressed_row_sparsity
  to_symmetric_compressed_row(Compressed_row_sparsity const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto shape = csr.shape();
    auto nrow = shape.row();

    // Filter to lower triangle (row >= col)
    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csr.size()));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        if (row >= ci[j]) {
          indices.push_back(Index{row, ci[j]});
        }
      }
    }

    return Symmetric_compressed_row_sparsity(
      shape, indices.begin(), indices.end());
  }

  Symmetric_compressed_row_sparsity
  to_symmetric_compressed_row(Symmetric_coordinate_sparsity const& scoo)
  {
    auto idx = scoo.indices();
    return Symmetric_compressed_row_sparsity(
      scoo.shape(), begin(idx), end(idx));
  }

  Compressed_row_sparsity
  to_compressed_row(Symmetric_coordinate_sparsity const& scoo)
  {
    // Route through sCSR as intermediate
    auto scsr = to_symmetric_compressed_row(scoo);
    return to_compressed_row(scsr);
  }

  Symmetric_block_sparse_row_sparsity
  to_symmetric_block_sparse_row(
    Compressed_row_sparsity const& csr,
    size_type block_rows, size_type block_cols)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto nrow = csr.shape().row();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csr.size()));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        indices.push_back(Index{row, ci[j]});
      }
    }

    return Symmetric_block_sparse_row_sparsity(
      csr.shape(), block_rows, block_cols,
      indices.begin(), indices.end());
  }

  Compressed_row_sparsity
  to_compressed_row(Symmetric_block_sparse_row_sparsity const& sbsr)
  {
    auto shape = sbsr.shape();
    auto nrow = shape.row();
    auto ncol = shape.column();
    auto br = sbsr.block_rows();
    auto bc = sbsr.block_cols();
    auto rp = sbsr.row_ptr();
    auto ci = sbsr.col_ind();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(sbsr.size() * 2));

    auto nbr = sbsr.num_block_rows();
    for (size_type block_row = 0; block_row < nbr; ++block_row) {
      for (auto j = rp[block_row]; j < rp[block_row + 1]; ++j) {
        auto block_col = ci[j];
        for (size_type lr = 0; lr < br; ++lr) {
          for (size_type lc = 0; lc < bc; ++lc) {
            auto row = block_row * br + lr;
            auto col = block_col * bc + lc;
            if (row < nrow && col < ncol) {
              indices.push_back(Index{row, col});
              if (block_row != block_col) {
                indices.push_back(Index{col, row});
              }
            }
          }
        }
      }
    }

    return Compressed_row_sparsity(shape, indices.begin(), indices.end());
  }

} // end of namespace sparkit::data::detail
