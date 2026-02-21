#pragma once

//
// ... Standard header files
//
#include <vector>

//
// ... External header files
//
#include <nlohmann/json.hpp>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/Coordinate_matrix.hpp>
#include <sparkit/data/Coordinate_sparsity.hpp>
#include <sparkit/data/Entry.hpp>

// adl_serializer specializations for non-default-constructible types.
// Index has no default constructor, so get<Index>() and
// get<std::vector<Index>>() require this specialization.
// Entry<T> contains Index, so it also needs specialization.

namespace nlohmann {

  template <>
  struct adl_serializer<sparkit::data::detail::Index> {
    static sparkit::data::detail::Index
    from_json(json const& j) {
      return sparkit::data::detail::Index{
          j[0].get<sparkit::config::size_type>(),
          j[1].get<sparkit::config::size_type>()};
    }

    static void
    to_json(json& j, sparkit::data::detail::Index const& idx) {
      j = {idx.row(), idx.column()};
    }
  };

  template <typename T>
  struct adl_serializer<sparkit::data::detail::Entry<T>> {
    static sparkit::data::detail::Entry<T>
    from_json(json const& j) {
      return sparkit::data::detail::Entry<T>{
          j.at("index").get<sparkit::data::detail::Index>(),
          j.at("value").get<T>()};
    }

    static void
    to_json(json& j, sparkit::data::detail::Entry<T> const& e) {
      j = {{"index", e.index}, {"value", e.value}};
    }
  };

} // end of namespace nlohmann

namespace sparkit::data::detail {

  // -- Coordinate_sparsity --

  inline nlohmann::json
  coordinate_sparsity_to_json(Coordinate_sparsity const& sp) {
    nlohmann::json j;
    j["shape"] = sp.shape();
    j["indices"] = sp.indices();
    return j;
  }

  inline Coordinate_sparsity
  coordinate_sparsity_from_json(nlohmann::json const& j) {
    auto shape = j.at("shape").get<Shape>();
    auto indices = j.at("indices").get<std::vector<Index>>();
    return Coordinate_sparsity{shape, indices.begin(), indices.end()};
  }

  // -- Compressed_row_sparsity --

  inline nlohmann::json
  compressed_row_sparsity_to_json(Compressed_row_sparsity const& sp) {
    nlohmann::json j;
    j["shape"] = sp.shape();

    auto rp = sp.row_ptr();
    j["row_ptr"] = std::vector<config::size_type>(rp.begin(), rp.end());

    auto ci = sp.col_ind();
    j["col_ind"] = std::vector<config::size_type>(ci.begin(), ci.end());

    return j;
  }

  inline Compressed_row_sparsity
  compressed_row_sparsity_from_json(nlohmann::json const& j) {
    auto shape = j.at("shape").get<Shape>();
    auto col_ind = j.at("col_ind").get<std::vector<config::size_type>>();
    auto row_ptr = j.at("row_ptr").get<std::vector<config::size_type>>();

    std::vector<Index> indices;
    indices.reserve(col_ind.size());

    auto nrow = shape.row();
    for (config::size_type row = 0; row < nrow; ++row) {
      for (auto k = row_ptr[static_cast<std::size_t>(row)];
           k < row_ptr[static_cast<std::size_t>(row + 1)]; ++k) {
        indices.push_back(Index{row, col_ind[static_cast<std::size_t>(k)]});
      }
    }

    return Compressed_row_sparsity{shape, indices.begin(), indices.end()};
  }

  // -- Coordinate_matrix<T> --

  template <typename T>
  nlohmann::json
  coordinate_matrix_to_json(Coordinate_matrix<T> const& mat) {
    nlohmann::json j;
    j["shape"] = mat.shape();

    auto entries = mat.entries();
    nlohmann::json arr = nlohmann::json::array();
    for (auto const& [index, value] : entries) {
      arr.push_back(Entry<T>{index, value});
    }
    j["entries"] = arr;

    return j;
  }

  template <typename T>
  Coordinate_matrix<T>
  coordinate_matrix_from_json(nlohmann::json const& j) {
    auto shape = j.at("shape").get<Shape>();
    auto entries = j.at("entries").get<std::vector<Entry<T>>>();
    return Coordinate_matrix<T>{shape, entries.begin(), entries.end()};
  }

  // -- Compressed_row_matrix<T> --

  template <typename T>
  nlohmann::json
  compressed_row_matrix_to_json(Compressed_row_matrix<T> const& mat) {
    nlohmann::json j;
    j["shape"] = mat.shape();

    auto rp = mat.row_ptr();
    j["row_ptr"] = std::vector<config::size_type>(rp.begin(), rp.end());

    auto ci = mat.col_ind();
    j["col_ind"] = std::vector<config::size_type>(ci.begin(), ci.end());

    auto sv = mat.values();
    j["values"] = std::vector<T>(sv.begin(), sv.end());

    return j;
  }

  template <typename T>
  Compressed_row_matrix<T>
  compressed_row_matrix_from_json(nlohmann::json const& j) {
    auto shape = j.at("shape").get<Shape>();
    auto row_ptr = j.at("row_ptr").get<std::vector<config::size_type>>();
    auto col_ind = j.at("col_ind").get<std::vector<config::size_type>>();
    auto values = j.at("values").get<std::vector<T>>();

    std::vector<Index> indices;
    indices.reserve(col_ind.size());

    auto nrow = shape.row();
    for (config::size_type row = 0; row < nrow; ++row) {
      for (auto k = row_ptr[static_cast<std::size_t>(row)];
           k < row_ptr[static_cast<std::size_t>(row + 1)]; ++k) {
        indices.push_back(Index{row, col_ind[static_cast<std::size_t>(k)]});
      }
    }

    Compressed_row_sparsity sparsity{shape, indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(values)};
  }

} // end of namespace sparkit::data::detail
