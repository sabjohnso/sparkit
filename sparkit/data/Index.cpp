#include <sparkit/data/Index.hpp>

namespace sparkit::data::detail{

  Index::Index(size_type row, size_type column)
      : row_(row)
      , column_(column)
    {
      if (row_ < 0 || column_ < 0) {
        throw logic_error("invalid matrix index");
      }
    }

  config::size_type
  Index::row() const { return row_; }

  config::size_type
  Index::column() const { return column_; }

  bool
  operator==(const Index& index1, const Index& index2){
    return index1.row_ == index2.row_ && index1.column_ == index2.column_;
  }

  void
  to_json(json& j, Index const& index)
  {
    j = {index.row(), index.column()};
  }

  void
  from_json(const json& j, Index& index){
    index = Index(j[0], j[1]);
  }
} // end of namespace sparkit::data::detail
