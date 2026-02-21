//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

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
#include <sparkit/data/json_serialization.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Coordinate_sparsity;
  using sparkit::data::detail::Coordinate_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Compressed_row_matrix;
  using nlohmann::json;

  // -- Entry --

  TEST_CASE("json_serialization - entry_to_json", "[json_serialization]")
  {
    Entry<double> e{Index{2, 3}, 4.5};
    json j = e;

    CHECK(j["index"][0] == 2);
    CHECK(j["index"][1] == 3);
    CHECK(j["value"] == Catch::Approx(4.5));
  }

  TEST_CASE("json_serialization - entry_from_json", "[json_serialization]")
  {
    json j = {{"index", {1, 4}}, {"value", 7.0}};
    auto e = j.get<Entry<double>>();

    CHECK(e.index.row() == 1);
    CHECK(e.index.column() == 4);
    CHECK(e.value == Catch::Approx(7.0));
  }

  TEST_CASE("json_serialization - entry_round_trip", "[json_serialization]")
  {
    Entry<double> original{Index{5, 8}, 3.14};
    json j = original;
    auto result = j.get<Entry<double>>();

    CHECK(result.index == original.index);
    CHECK(result.value == Catch::Approx(original.value));
  }

  // -- Coordinate_sparsity --

  TEST_CASE("json_serialization - coordinate_sparsity_to_json", "[json_serialization]")
  {
    Coordinate_sparsity sp{Shape{3, 4}, {
      Index{0, 1}, Index{1, 2}, Index{2, 0}
    }};

    json j = sparkit::data::detail::coordinate_sparsity_to_json(sp);

    CHECK(j["shape"][0] == 3);
    CHECK(j["shape"][1] == 4);
    CHECK(j["indices"].size() == 3);
  }

  TEST_CASE("json_serialization - coordinate_sparsity_round_trip", "[json_serialization]")
  {
    Coordinate_sparsity original{Shape{4, 5}, {
      Index{0, 0}, Index{1, 3}, Index{2, 1}, Index{3, 4}
    }};

    json j = sparkit::data::detail::coordinate_sparsity_to_json(original);
    auto result = sparkit::data::detail::coordinate_sparsity_from_json(j);

    CHECK(result.shape() == original.shape());
    CHECK(result.size() == original.size());

    auto orig_indices = original.indices();
    auto res_indices = result.indices();
    REQUIRE(orig_indices.size() == res_indices.size());

    // Sort both for comparison since COO order may vary
    auto by_row_col = [](auto const& a, auto const& b) {
      return a.row() < b.row()
        || (a.row() == b.row() && a.column() < b.column());
    };
    std::sort(orig_indices.begin(), orig_indices.end(), by_row_col);
    std::sort(res_indices.begin(), res_indices.end(), by_row_col);

    for (std::size_t i = 0; i < orig_indices.size(); ++i) {
      CHECK(orig_indices[i] == res_indices[i]);
    }
  }

  // -- Compressed_row_sparsity --

  TEST_CASE("json_serialization - compressed_row_sparsity_to_json", "[json_serialization]")
  {
    Compressed_row_sparsity sp{Shape{3, 4}, {
      Index{0, 1}, Index{1, 2}, Index{2, 0}
    }};

    json j = sparkit::data::detail::compressed_row_sparsity_to_json(sp);

    CHECK(j["shape"][0] == 3);
    CHECK(j["shape"][1] == 4);
    CHECK(j["row_ptr"].size() == 4);  // nrow + 1
    CHECK(j["col_ind"].size() == 3);
  }

  TEST_CASE("json_serialization - compressed_row_sparsity_round_trip", "[json_serialization]")
  {
    Compressed_row_sparsity original{Shape{3, 4}, {
      Index{0, 1}, Index{0, 3}, Index{1, 2}, Index{2, 0}
    }};

    json j = sparkit::data::detail::compressed_row_sparsity_to_json(original);
    auto result = sparkit::data::detail::compressed_row_sparsity_from_json(j);

    CHECK(result.shape() == original.shape());
    CHECK(result.size() == original.size());

    auto orig_rp = original.row_ptr();
    auto res_rp = result.row_ptr();
    REQUIRE(orig_rp.size() == res_rp.size());
    for (std::size_t i = 0; i < orig_rp.size(); ++i) {
      CHECK(orig_rp[i] == res_rp[i]);
    }

    auto orig_ci = original.col_ind();
    auto res_ci = result.col_ind();
    REQUIRE(orig_ci.size() == res_ci.size());
    for (std::size_t i = 0; i < orig_ci.size(); ++i) {
      CHECK(orig_ci[i] == res_ci[i]);
    }
  }

  // -- Coordinate_matrix --

  TEST_CASE("json_serialization - coordinate_matrix_to_json", "[json_serialization]")
  {
    Coordinate_matrix<double> mat{Shape{3, 4}, {
      {Index{0, 1}, 1.5},
      {Index{2, 3}, 2.7}
    }};

    json j = sparkit::data::detail::coordinate_matrix_to_json(mat);

    CHECK(j["shape"][0] == 3);
    CHECK(j["shape"][1] == 4);
    CHECK(j["entries"].size() == 2);
  }

  TEST_CASE("json_serialization - coordinate_matrix_round_trip", "[json_serialization]")
  {
    Coordinate_matrix<double> original{Shape{4, 5}, {
      {Index{0, 0}, 1.0},
      {Index{1, 3}, 2.5},
      {Index{2, 1}, 3.7},
      {Index{3, 4}, 4.2}
    }};

    json j = sparkit::data::detail::coordinate_matrix_to_json(original);
    auto result = sparkit::data::detail::coordinate_matrix_from_json<double>(j);

    CHECK(result.shape() == original.shape());
    CHECK(result.size() == original.size());

    for (config::size_type r = 0; r < 4; ++r) {
      for (config::size_type c = 0; c < 5; ++c) {
        CHECK(result(r, c) == Catch::Approx(original(r, c)));
      }
    }
  }

  TEST_CASE("json_serialization - coordinate_matrix_empty_round_trip", "[json_serialization]")
  {
    Coordinate_matrix<double> original{Shape{3, 3}};

    json j = sparkit::data::detail::coordinate_matrix_to_json(original);
    auto result = sparkit::data::detail::coordinate_matrix_from_json<double>(j);

    CHECK(result.shape() == original.shape());
    CHECK(result.size() == 0);
  }

  // -- Compressed_row_matrix --

  TEST_CASE("json_serialization - compressed_row_matrix_to_json", "[json_serialization]")
  {
    Compressed_row_matrix<double> mat{Shape{3, 4}, {
      {Index{0, 1}, 1.5},
      {Index{1, 2}, 2.7},
      {Index{2, 0}, 3.9}
    }};

    json j = sparkit::data::detail::compressed_row_matrix_to_json(mat);

    CHECK(j["shape"][0] == 3);
    CHECK(j["shape"][1] == 4);
    CHECK(j["row_ptr"].size() == 4);  // nrow + 1
    CHECK(j["col_ind"].size() == 3);
    CHECK(j["values"].size() == 3);
  }

  TEST_CASE("json_serialization - compressed_row_matrix_round_trip", "[json_serialization]")
  {
    Compressed_row_matrix<double> original{Shape{4, 5}, {
      {Index{0, 0}, 1.0},
      {Index{0, 3}, 2.0},
      {Index{1, 1}, 3.0},
      {Index{2, 4}, 4.0},
      {Index{3, 2}, 5.0}
    }};

    json j = sparkit::data::detail::compressed_row_matrix_to_json(original);
    auto result = sparkit::data::detail::compressed_row_matrix_from_json<double>(j);

    CHECK(result.shape() == original.shape());
    CHECK(result.size() == original.size());

    for (config::size_type r = 0; r < 4; ++r) {
      for (config::size_type c = 0; c < 5; ++c) {
        CHECK(result(r, c) == Catch::Approx(original(r, c)));
      }
    }
  }

  TEST_CASE("json_serialization - compressed_row_matrix_empty_round_trip", "[json_serialization]")
  {
    Compressed_row_sparsity sp{Shape{3, 3}, {}};
    Compressed_row_matrix<double> original{sp, {}};

    json j = sparkit::data::detail::compressed_row_matrix_to_json(original);
    auto result = sparkit::data::detail::compressed_row_matrix_from_json<double>(j);

    CHECK(result.shape() == original.shape());
    CHECK(result.size() == 0);
  }

} // end of namespace sparkit::testing
