#pragma once

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/import.hpp>

namespace sparkit::data::detail {

  /**
   * @brief A type describing a matrix index: the number of rows and columns
   */
  class Index final {
  public:
    using size_type = config::size_type;

    Index(size_type row, size_type column);

    size_type
    row() const;

    size_type
    column() const;

    friend bool
    operator==(const Index& index1, const Index& index2);

  private:
    size_type row_{};
    size_type column_{};

  }; // end of class Index

  void
  to_json(json& j, Index const& index);

  void
  from_json(const json& j, Index& index);

  struct IndexHash {
    size_type
    operator()(Index index) const {
      return (index.row() << 32) ^ index.column();
    }
  };

} // namespace sparkit::data::detail
