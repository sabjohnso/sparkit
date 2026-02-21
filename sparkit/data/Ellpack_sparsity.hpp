#pragma once

//
// ... Standard header files
//
#include <initializer_list>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Index.hpp>
#include <sparkit/data/Shape.hpp>

namespace sparkit::data::detail {

  /**
   * @brief Immutable ELLPACK/ITPACK (ELL) sparsity pattern.
   *
   * Stores column indices in a padded row-major array of dimensions
   * nrow x max_nnz_per_row. Unused slots are padded with sentinel -1.
   *
   * @note This class uses the PImpl idiom for ABI stability.
   *
   * @see Compressed_row_sparsity
   */
  class Ellpack_sparsity final {
  public:
    using size_type = config::size_type;

    Ellpack_sparsity(Shape shape, std::initializer_list<Index> const& input);

    template <typename Iter>
    Ellpack_sparsity(Shape shape, Iter first, Iter last)
        : Ellpack_sparsity(shape, std::vector<Index>(first, last)) {}

    Ellpack_sparsity(Ellpack_sparsity const& input);
    Ellpack_sparsity(Ellpack_sparsity&& input);

    Ellpack_sparsity&
    operator=(Ellpack_sparsity const& input);

    Ellpack_sparsity&
    operator=(Ellpack_sparsity&& input);

    ~Ellpack_sparsity();

    size_type
    size() const;

    Shape
    shape() const;

    /**
     * @brief Return the maximum number of nonzeros per row.
     */
    size_type
    max_nnz_per_row() const;

    /**
     * @brief Return the padded column-index array (row-major, nrow x max_nnz).
     *
     * Sentinel value is -1.
     */
    std::span<size_type const>
    col_ind() const;

  private:
    Ellpack_sparsity(Shape shape, std::vector<Index> indices);

    class Impl;
    Impl* pimpl;

  }; // end of class Ellpack_sparsity

} // end of namespace sparkit::data::detail
