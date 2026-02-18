#include <sparkit/data/Shape.hpp>


namespace sparkit::data::detail
{

  Shape::Shape(size_type row, size_type column)
      : row_(row)
      , column_(column)
    {
      if (row_ <= 1 || column_ <= 1) {
        throw logic_error("invalid matrix shape");
      }
    }

  config::size_type
  Shape::row() const { return row_; }

  config::size_type
  Shape::column() const { return column_; }

  bool
  operator==(const Shape& shape1, const Shape& shape2){
    return shape1.row_ == shape2.row_ && shape1.column_ == shape2.column_;
  }

  void
  to_json(json& j, Shape const& shape)
  {
    j = {shape.row(), shape.column()};
  }

  void
  from_json(const json& j, Shape& shape){
    shape = Shape(j[0], j[1]);
  }

} // end of namespace sparkit::data::detail
