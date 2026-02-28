#pragma once

//
// ... Standard header files
//
#include <utility>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_sparsity.hpp>

namespace sparkit::data::detail {

  // Greedy 3-phase aggregation (Vanĕk/Mandel/Brezina 1996).
  //
  // Phase 1 (seed): An unaggregated node whose neighbors are all
  //   unaggregated becomes a seed and absorbs its strong neighbors.
  // Phase 2 (cleanup): Remaining unaggregated nodes join the aggregate
  //   of any already-aggregated neighbor.
  // Phase 3 (singletons): Leftover nodes become singleton aggregates.
  //
  // Returns (aggregate_ids[n], n_aggregates).

  inline std::pair<std::vector<config::size_type>, config::size_type>
  aggregate(Compressed_row_sparsity const& strength, config::size_type n) {
    config::size_type const unaggregated = -1;
    std::vector<config::size_type> agg_ids(
      static_cast<std::size_t>(n), unaggregated);
    config::size_type n_agg = 0;

    auto rp = strength.row_ptr();
    auto ci = strength.col_ind();

    // Phase 1: Seed — node with all-unaggregated neighbors becomes seed.
    for (config::size_type i = 0; i < n; ++i) {
      if (agg_ids[static_cast<std::size_t>(i)] != unaggregated) { continue; }

      bool all_free = true;
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        if (agg_ids[static_cast<std::size_t>(ci[p])] != unaggregated) {
          all_free = false;
          break;
        }
      }

      if (all_free) {
        agg_ids[static_cast<std::size_t>(i)] = n_agg;
        for (auto p = rp[i]; p < rp[i + 1]; ++p) {
          agg_ids[static_cast<std::size_t>(ci[p])] = n_agg;
        }
        ++n_agg;
      }
    }

    // Phase 2: Remaining nodes join a neighbor's aggregate.
    for (config::size_type i = 0; i < n; ++i) {
      if (agg_ids[static_cast<std::size_t>(i)] != unaggregated) { continue; }

      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto neighbor_agg = agg_ids[static_cast<std::size_t>(ci[p])];
        if (neighbor_agg != unaggregated) {
          agg_ids[static_cast<std::size_t>(i)] = neighbor_agg;
          break;
        }
      }
    }

    // Phase 3: Singletons — leftover nodes get their own aggregate.
    for (config::size_type i = 0; i < n; ++i) {
      if (agg_ids[static_cast<std::size_t>(i)] == unaggregated) {
        agg_ids[static_cast<std::size_t>(i)] = n_agg;
        ++n_agg;
      }
    }

    return {std::move(agg_ids), n_agg};
  }

} // end of namespace sparkit::data::detail
