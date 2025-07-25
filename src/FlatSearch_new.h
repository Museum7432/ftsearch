#pragma once

#include "RaggedArray.h"
#include "SearchResult.h"
#include "half.hpp"
#include "utils.h"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#define V_BATCH_SIZE_Q 1024
#define V_BATCH_SIZE_X 1024
#define FP32_CACHE_SIZE 16

namespace seqr {

// simply search for Q in Xs (similar to FlatIndexIP)
template <typename T = float>
SearchResult search_baseline(const float *Q, const size_t nq, const T *Xs, const size_t nv, const size_t d, const size_t topk);

template <size_t Q_N, size_t DB_N, size_t batch_level, size_t search_level, typename T>
SearchResult search(const float *Q, const RaggedIndexer<Q_N> &query_I, const T *db, const RaggedIndexer<DB_N> &db_I, const size_t d, const size_t topk);

template <size_t Q_N, size_t DB_N, size_t batch_level, size_t subset_level, size_t search_level, typename T>
SearchResult search_subset(const float *Q, const RaggedIndexer<Q_N> &query_I, const T *db, const RaggedIndexer<DB_N> &db_I, const size_t d, const size_t *ids, const size_t num_ids, const size_t topk);

// sim_buffer is of shape db_vec_size, query_vec_size
template <size_t Q_N, size_t DB_N, size_t batch_level, size_t search_level>
void save_result(SearchResult &re, float *sim_buffer, size_t db_vec_start, size_t db_vec_size, size_t query_vec_start, size_t query_vec_size, size_t db_block_start, size_t db_block_size, size_t query_batch_start, size_t query_batch_size, const RaggedIndexer<Q_N> &query_I, const RaggedIndexer<DB_N> &db_I);


template <size_t Q_N, size_t DB_N, size_t batch_level, size_t query_seq_level, size_t search_level>
SeqSearchResult seq_search(const RaggedVectors<Q_N> &queries, const RaggedVectors<DB_N> &db, size_t topk, const float discount_rate = 1.0);


} // namespace seqr

// template implementation
namespace seqr {

template <typename T>
SearchResult search_baseline(const float *Q, const size_t nq, const T *Xs, const size_t nv, const size_t d, const size_t topk) {
  static_assert(std::is_same_v<T, half_float::half> || std::is_same_v<T, float>, "only support fp16 or fp32 (half or float)");

  SearchResult topk_result(nq, topk);

  // divide the db into shards for each thread with openMP
  shard_loop<true>(0, nv, [&](size_t db_shard_start, size_t db_shard_size) {
    // this is run in each thread
    std::vector<float> fp32_cache_buffer;

    if constexpr (std::is_same_v<T, half_float::half>)
      fp32_cache_buffer.resize(FP32_CACHE_SIZE * d, 0);

    SearchResult local_results(nq, topk);
    std::vector<float> sim_buffer(V_BATCH_SIZE_X * V_BATCH_SIZE_Q);

    // then loop through the block of query and db
    batch_loop(0, nq, V_BATCH_SIZE_Q, [&](size_t query_batch_start, size_t query_batch_size) {
      batch_loop(db_shard_start, db_shard_size, V_BATCH_SIZE_X, [&](size_t db_block_start, size_t db_block_size) {
        // query: (query_batch_start, query_batch_size)
        // db: (db_block_start, db_block_size)

        if (sim_buffer.size() < db_block_size * query_batch_size) {
          sim_buffer.clear();
          sim_buffer.resize(db_block_size * query_batch_size);
        }

        // => (db_block_size, query_batch_size)
        if constexpr (std::is_same_v<T, half_float::half>) {
          similarity_dot_product_trans_out_fp16(
              Q + query_batch_start * d,
              Xs + db_block_start * d,
              sim_buffer.data(),
              query_batch_size, db_block_size, d,
              fp32_cache_buffer.data(), FP32_CACHE_SIZE);

        } else {
          similarity_dot_product_trans_out(
              Q + query_batch_start * d,
              Xs + db_block_start * d,
              sim_buffer.data(),
              query_batch_size, db_block_size, d);
        }

        // then loop over the result
        for_loop(db_block_start, db_block_size, [&](size_t db_i) {
          for_loop(query_batch_start, query_batch_size, [&](size_t q_i) {
            // TODO: use embrace back
            local_results.push(
                ItemScore(sim_buffer[(q_i - query_batch_start) + (db_i - db_block_start) * query_batch_size], db_i),
                q_i);
          });
        });
      });

#pragma omp critical
      {
        // this will clear the local_results
        topk_result.merge(local_results);
      }
    });
  });

  topk_result.sort();

  return topk_result;
}

template <size_t Q_N, size_t DB_N, size_t batch_level, size_t search_level>
void save_result(SearchResult &re, float *sim_buffer, size_t db_vec_start, size_t db_vec_size, size_t query_vec_start, size_t query_vec_size, size_t db_block_start, size_t db_block_size, size_t query_batch_start, size_t query_batch_size, const RaggedIndexer<Q_N> &query_I, const RaggedIndexer<DB_N> &db_I) {

  static constexpr const size_t query_vec_level = Q_N - 1;
  static constexpr const size_t db_vec_level = DB_N - 1;

  if constexpr (db_vec_level - search_level == 0) {
    // 1 vector per database entry
    static_assert(query_vec_level - batch_level == 0, "multiple vector per query on a 1 vector per entry DB is the same as averaging those vectors directly before searching");
    // 1 vector per entry
    for_loop(db_block_start, db_block_size, [&](size_t db_i) {
      for_loop(query_batch_start, query_batch_size, [&](size_t q_i) {
        // TODO: use embrace back
        re.push(
            ItemScore(sim_buffer[(q_i - query_batch_start) + (db_i - db_block_start) * query_batch_size], db_i),
            q_i);
      });
    });
  } else {
    // multiple vectors per database entry
    // max reduction

    db_I.template span_loop<search_level, db_vec_level>(db_block_start, db_block_size, [&](size_t db_entry_idx, size_t db_vec_local_start, size_t db_vec_local_size) {
      // span (db_vec_local_start, db_vec_local_size) is of db_entry_idx

      // the start row of sim_buffer
      auto start_ptr = sim_buffer + (db_vec_local_start - db_vec_start) * query_vec_size;

      // loop through other row
      for (size_t local_vec_idx = 1; local_vec_idx < db_vec_local_size; local_vec_idx++) {
        auto current_ptr = start_ptr + local_vec_idx * query_vec_size;

#pragma omp simd
        for (size_t local_query_vec_idx = 0; local_query_vec_idx < query_vec_size; local_query_vec_idx++) {

          start_ptr[local_query_vec_idx] = std::max(start_ptr[local_query_vec_idx], current_ptr[local_query_vec_idx]);
        }
      }

      if constexpr (query_vec_level - batch_level == 0) {
        // 1 vector per query, just save the result
        for_loop(query_batch_start, query_batch_size, [&](size_t q_i) {
          // TODO: use embrace back
          re.push(
              ItemScore(start_ptr[(q_i - query_batch_start)], db_entry_idx),
              q_i);
        });
      } else {
        // multiple vectors per query
        // use mean reduction (over vectors per query)
        query_I.template span_loop<batch_level, query_vec_level>(query_batch_start, query_batch_size, [&](size_t query_entry_idx, size_t query_vec_local_start, size_t query_vec_local_size) {
          // query_vec_local_size is guaranteed to not be 0

          auto first_q_ptr = start_ptr + query_vec_local_start - query_vec_start;
          float sum = 0;

#pragma omp simd reduction(+ : sum)
          for (size_t local_query_vec_idx = 0; local_query_vec_idx < query_vec_local_size; local_query_vec_idx++) {
            sum += first_q_ptr[local_query_vec_idx];
          }

          sum /= (query_vec_local_size < 1) ? 1 : query_vec_local_size;

          // then save the result
          re.push(
              ItemScore(sum, db_entry_idx),
              query_entry_idx);
        });
      }
    });
  }
}

template <size_t Q_N, size_t DB_N, size_t batch_level, size_t search_level, typename T>
SearchResult search(const float *Q, const RaggedIndexer<Q_N> &query_I, const T *db, const RaggedIndexer<DB_N> &db_I, const size_t d, const size_t topk) {
  static_assert(std::is_same_v<T, half_float::half> || std::is_same_v<T, float>, "only support fp16 or fp32 (half or float)");

  // different to normal search
  // this support multiple vectors per entries (both query and database)
  // during search it is equivalent to flatenning both tensor to 2d ragged array (the input dim and the last dim)

  // assume that the last level manage the vectors
  static constexpr const size_t query_vec_level = Q_N - 1;
  static constexpr const size_t db_vec_level = DB_N - 1;

  const size_t num_queries = query_I.level_size(batch_level);
  const size_t num_db_entries = db_I.level_size(search_level);

  SearchResult topk_result(num_queries, topk);

  // divide the db into shards for each thread with openMP
  db_I.template balance_shard_loop<search_level, db_vec_level, true>(0, num_db_entries, [&](size_t db_shard_start, size_t db_shard_size) {
    // this is run in each thread
    std::vector<float> fp32_cache_buffer;

    if constexpr (std::is_same_v<T, half_float::half>)
      fp32_cache_buffer.resize(FP32_CACHE_SIZE * d, 0);

    SearchResult local_results(num_queries, topk);
    std::vector<float> sim_buffer(V_BATCH_SIZE_X * V_BATCH_SIZE_Q);

    // then loop through the block of query and db
    // batch by number of vector
    query_I.template batch_loop<batch_level, query_vec_level>(0, num_queries, V_BATCH_SIZE_Q, [&](size_t query_batch_start, size_t query_batch_size) {
      // span of queries to span of vectors
      auto [query_vec_start, query_vec_size] = query_I.template map_span<batch_level, query_vec_level>(query_batch_start, query_batch_size);

      db_I.template batch_loop<search_level, db_vec_level>(db_shard_start, db_shard_size, V_BATCH_SIZE_X, [&](size_t db_block_start, size_t db_block_size) {
        auto [db_vec_start, db_vec_size] = db_I.template map_span<search_level, db_vec_level>(db_block_start, db_block_size);

        // block: (query_batch_start, query_batch_size), (db_block_start, db_block_size)
        // vector block: (query_vec_start, query_vec_size), (db_vec_start, db_vec_size)

        if (sim_buffer.size() < db_vec_size * query_vec_size) {
          // now extend the buffer if needed
          sim_buffer.clear();
          sim_buffer.resize(db_vec_size * query_vec_size);
        }
        // Q vector block: (query_vec_start, query_vec_size)
        // db vector block: (db_vec_start, db_vec_size)
        // => (db_vec_size, query_vec_size)
        if constexpr (std::is_same_v<T, half_float::half>) {
          similarity_dot_product_trans_out_fp16(
              Q + query_vec_start * d,
              db + db_vec_start * d,
              sim_buffer.data(),
              query_vec_size, db_vec_size, d,
              fp32_cache_buffer.data(), FP32_CACHE_SIZE);

        } else {
          similarity_dot_product_trans_out(
              Q + query_vec_start * d,
              db + db_vec_start * d,
              sim_buffer.data(),
              query_vec_size, db_vec_size, d);
        }

        // then accumulate the block
        save_result<Q_N, DB_N, batch_level, search_level>(
            local_results,
            sim_buffer.data(),
            db_vec_start, db_vec_size,
            query_vec_start, query_vec_size,
            db_block_start, db_block_size,
            query_batch_start, query_batch_size,
            query_I, db_I);
      });

      // then merge the results
#pragma omp critical
      {
        // this will clear the local_topk_result
        topk_result.merge(local_results);
      }
    });
  });

  topk_result.sort();

  return topk_result;
}

template <size_t Q_N, size_t DB_N, size_t batch_level, size_t subset_level, size_t search_level, typename T>
SearchResult search_subset(const float *Q, const RaggedIndexer<Q_N> &query_I, const T *db, const RaggedIndexer<DB_N> &db_I, const size_t d, const size_t *ids, const size_t num_ids, const size_t topk) {
  static constexpr const size_t query_vec_level = Q_N - 1;
  static constexpr const size_t db_vec_level = DB_N - 1;

  const size_t num_queries = query_I.level_size(batch_level);
  SearchResult topk_result(num_queries, topk);

  // divide num_ids into shards for each thread with openMP
  shard_loop<true>(0, num_ids, [&](size_t start_, size_t size_) {
    // this is run in each thread
    std::vector<float> fp32_cache_buffer;

    if constexpr (std::is_same_v<T, half_float::half>)
      fp32_cache_buffer.resize(FP32_CACHE_SIZE * d, 0);

    SearchResult local_results(num_queries, topk);
    std::vector<float> sim_buffer(V_BATCH_SIZE_X * V_BATCH_SIZE_Q);

    for_loop(start_, size_, [&](size_t id_) {
      auto [db_shard_start, db_shard_size] = db_I.template map_span<subset_level, search_level>(ids[id_], 1);
      // then loop through the block of query and db

      // batch by number of vector
      query_I.template batch_loop<batch_level, query_vec_level>(0, num_queries, V_BATCH_SIZE_Q, [&](size_t query_batch_start, size_t query_batch_size) {
        // span of queries to span of vectors
        auto [query_vec_start, query_vec_size] = query_I.template map_span<batch_level, query_vec_level>(query_batch_start, query_batch_size);

        db_I.template batch_loop<search_level, db_vec_level>(db_shard_start, db_shard_size, V_BATCH_SIZE_X, [&](size_t db_block_start, size_t db_block_size) {
          auto [db_vec_start, db_vec_size] = db_I.template map_span<search_level, db_vec_level>(db_block_start, db_block_size);

          // block: (query_batch_start, query_batch_size), (db_block_start, db_block_size)
          // vector block: (query_vec_start, query_vec_size), (db_vec_start, db_vec_size)

          if (sim_buffer.size() < db_vec_size * query_vec_size) {
            // now extend the buffer if needed
            sim_buffer.clear();
            sim_buffer.resize(db_vec_size * query_vec_size);
          }
          // Q vector block: (query_vec_start, query_vec_size)
          // db vector block: (db_vec_start, db_vec_size)
          // => (db_vec_size, query_vec_size)
          if constexpr (std::is_same_v<T, half_float::half>) {
            similarity_dot_product_trans_out_fp16(
                Q + query_vec_start * d,
                db + db_vec_start * d,
                sim_buffer.data(),
                query_vec_size, db_vec_size, d,
                fp32_cache_buffer.data(), FP32_CACHE_SIZE);

          } else {
            similarity_dot_product_trans_out(
                Q + query_vec_start * d,
                db + db_vec_start * d,
                sim_buffer.data(),
                query_vec_size, db_vec_size, d);
          }

          // then accumulate the block
          save_result<Q_N, DB_N, batch_level, search_level>(
              local_results,
              sim_buffer.data(),
              db_vec_start, db_vec_size,
              query_vec_start, query_vec_size,
              db_block_start, db_block_size,
              query_batch_start, query_batch_size,
              query_I, db_I);
        });

        // then merge the results
      });
    });
#pragma omp critical
    {
      // this will clear the local_topk_result
      topk_result.merge(local_results);
    }
  });

  topk_result.sort();

  return topk_result;
}

} // namespace seqr