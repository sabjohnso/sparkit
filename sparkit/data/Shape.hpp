#pragma once

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/import.hpp>

namespace sparkit::data::detail {

  /**
   * @brief A type describing a matrix shape: the number of rows and columns
   */
  class Shape final {
  public:
    using size_type = config::size_type;

    Shape(size_type row, size_type column);
    Shape(const Shape& input) = default;
    Shape&
    operator=(const Shape& input) = default;
    Shape(Shape&& input) = default;
    Shape&
    operator=(Shape&& input) = default;
    ~Shape() = default;
    Shape() = default;

    size_type
    row() const;

    size_type
    column() const;

    friend bool
    operator==(const Shape& shape1, const Shape& shape2);

  private:
    size_type row_{};
    size_type column_{};

  }; // end of class Shape

  void
  to_json(json& j, Shape const& shape);

  void
  from_json(const json& j, Shape& shape);

} // namespace sparkit::data::detail
