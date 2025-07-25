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

namespace seqr {

// SearchResult search_baseline(const float *Q, const size_t nq, const float *Xs, const size_t nv, const size_t d, const size_t topk);

// simply search for Q in Xs
template <typename T = float>
SearchResult search_baseline(const float *Q, const size_t nq, const T *Xs, const size_t nv, const size_t d, const size_t topk);

// search for Q in Xs along the search_level level of Xs,
// if there are dimensions after search_level, then use max reduction over those flaten vector.
// if db dimensions is (videos, frames, vectors), then DB_N=3 and search_level=1 would search for frame, taking the max over vectors.
// then use map_idx_rev and map_span to get the video and local index
// similarly with queries dimensions being (batch, vectors), then Q_N=2 and search_level=1 would use mean reduction over those vector
template <size_t Q_N, size_t DB_N, size_t batch_level, size_t search_level>
SearchResult search(const RaggedVectors<Q_N> &queries, const RaggedVectors<DB_N> &db, size_t topk);

// assume that the dimension next to the batch and search level are the sequence dimension
// if db dimensions is (videos, frames, vectors), then DB_N=3 and search_level=0 would search for sequence of frames (in each videos), taking the max over vectors.
template <size_t Q_N, size_t DB_N, size_t batch_level, size_t search_level>
SeqSearchResult seq_search(const RaggedVectors<Q_N> &queries, const RaggedVectors<DB_N> &db, size_t topk, const float discount_rate = 1.0);

} // namespace seqr

// template implementation
namespace seqr {

template <typename T>
SearchResult search_baseline(const float *Q, const size_t nq, const T *Xs, const size_t nv, const size_t d, const size_t topk) {
  SearchResult topk_result(nq, topk);

  const constexpr size_t BATCH_SIZE_Q = 1024;
  const constexpr size_t BATCH_SIZE_X = 1024;
  const constexpr size_t FP32_CACHE_SIZE = 16;

  batch_loop(0, nq, BATCH_SIZE_Q, [&](size_t query_batch_start, size_t query_batch_size) {
    // use openMP
    shard_loop<true>(0, nv, [&](size_t db_shard_start, size_t db_shard_size) {
      std::vector<float> sim_buffer(BATCH_SIZE_X * query_batch_size);
      SearchResult local_results(nq, topk);

      std::vector<float> fp32_cache_buffer;

      if constexpr (std::is_same_v<T, half_float::half>) {
        fp32_cache_buffer.resize(FP32_CACHE_SIZE * d, 0);
      }

      batch_loop(db_shard_start, db_shard_size, V_BATCH_SIZE_X, [&](size_t db_block_start, size_t db_block_size) {
        if (sim_buffer.size() < db_block_size * query_batch_size) {

          sim_buffer.clear();
          sim_buffer.resize(db_block_size * query_batch_size);
        }

        if constexpr (std::is_same_v<T, half_float::half>) {
          similarity_dot_product_trans_out_fp16(
              Q + query_batch_start * d,
              Xs + db_block_start * d,
              sim_buffer.data(),
              query_batch_size, db_block_size, d,
              fp32_cache_buffer.data(), FP32_CACHE_SIZE);

          //           const constexpr size_t tile = 4;
          //           for (size_t ii = 0; ii < db_block_size; ii += tile) {
          //             // load a tile of Xs into cache
          //             fp16_load(fp32_cache_buffer.data(), Xs + (ii + db_block_start) * d, std::min(tile, db_block_size - ii) * d);

          //             for (size_t jj = 0; jj < query_batch_size; jj += tile) {

          //               for (int i = ii; i < std::min(ii + tile, db_block_size); i++) {

          //                 for (int j = jj; j < std::min(jj + tile, query_batch_size); j++) {

          //                   float sum = 0;
          // #pragma omp simd reduction(+ : sum)
          //                   for (int k = 0; k < d; k++) {
          //                     // sum += Xs[(db_block_start + i) * d + k] + Q[(query_batch_start + j) * d + k];
          //                     sum += fp32_cache_buffer[(i - ii) * d + k] + Q[(query_batch_start + j) * d + k];
          //                   }

          //                   sim_buffer[j + i * nq] = sum;

          //                   // S[j + i * nq] = cblas_sdot(d, Xs + i * d, 1, Q + j * d, 1);
          //                 }
          //               }
          //             }
          //           }

          // for_loop(0, db_block_size, [&](size_t x_i) {
          //   fp16_load(fp32_cache_buffer.data(), Xs + (x_i + db_block_start) * d, d);

          //   cblas_sgemv(
          //       CblasRowMajor, CblasNoTrans,
          //       query_batch_size, d,
          //       1, Q + query_batch_start * d, d,
          //       fp32_cache_buffer.data(), 1, 0,
          //       sim_buffer.data() + x_i * query_batch_size, 1);
          // });

          // batch_loop(db_block_start, db_block_size, FP32_CACHE_SIZE, [&](size_t db_fp16_start, size_t db_fp16_size) {
          //   const T *fp16_start_ptr = Xs + db_fp16_start * d;

          //   // fp16_load(fp32_cache_buffer.data(), fp16_start_ptr, db_fp16_size * d);

          //   similarity_dot_product_trans_out(
          //       Q + query_batch_start * d,
          //       fp32_cache_buffer.data(),
          //       sim_buffer.data() + (db_fp16_start - db_block_start) * query_batch_size,
          //       query_batch_size, db_fp16_size, d);

          //   // use sgemm directly
          //   // cblas_sgemm(
          //   //     CblasRowMajor, CblasNoTrans, CblasTrans,
          //   //     db_fp16_size, query_batch_size, d,
          //   //     1,
          //   //     fp32_cache_buffer.data(), d,
          //   //     Q + query_batch_start * d, d,
          //   //     0, sim_buffer.data() + (db_fp16_start - db_block_start) * query_batch_size, query_batch_size);
          // });

        } else {

          // then compute batch_Xs @ Q
          // => (db_block_size, query_batch_size)
          similarity_dot_product_trans_out(
              Q + query_batch_start * d,
              Xs + db_block_start * d,
              sim_buffer.data(),
              query_batch_size, db_block_size, d);
        }

        for_loop(db_block_start, db_block_size, [&](size_t db_i) {
          for_loop(query_batch_start, query_batch_size, [&](size_t q_i) {
            local_results.push(
                ItemScore(sim_buffer[(q_i - query_batch_start) + (db_i - db_block_start) * query_batch_size], db_i),
                q_i);
          });
        });
      });

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

template <size_t Q_N, size_t DB_N, size_t batch_level, size_t search_level>
SeqSearchResult seq_search(const RaggedVectors<Q_N> &queries, const RaggedVectors<DB_N> &db, size_t topk, const float discount_rate) {
  if (queries.d != db.d)
    throw std::runtime_error("dimension mismatch between query and db");

  // assume that the last level manage the vectors
  static constexpr const size_t query_vec_level = Q_N - 1;
  static constexpr const size_t db_vec_level = DB_N - 1;

  // a query or db entry hold sub_entry
  static constexpr const size_t query_seq_level = search_level + 1;
  static constexpr const size_t db_seq_level = batch_level + 1;

  static_assert(query_seq_level < Q_N, "a dimension after search_level is expected");
  static_assert(db_seq_level < DB_N, "a dimension after batch_level is expected");

  const size_t num_queries = queries.I.level_size(batch_level);
  const size_t d = db.d;

  SeqSearchResult topk_result(num_queries, topk);

  // TODO: Ensure sim_buffer memory is aligned for optimal SIMD performance.
  std::vector<float> query_batch_transposed;

  // loop over batch of queries
  queries.I.template batch_loop<batch_level, query_vec_level>(0, num_queries, [&](size_t query_batch_start, size_t query_batch_size) {
    // get the range of the vectors span
    const auto [query_vec_offset, num_query_vecs] = queries.I.template map_span<batch_level, query_vec_level>(query_batch_start, query_batch_size);
    const auto [query_batch_seq_start, query_batch_seq_size] = db.I.template map_span<search_level, db_seq_level>(query_batch_start, query_batch_size);

    query_batch_transposed.clear();
    query_batch_transposed.resize(num_query_vecs * d);

    // transpose this batch of query
    tranpose(queries.data() + query_vec_offset * d, query_batch_transposed.data(), num_query_vecs, d);
    // shape: (d, num_query_vecs)
    const float *batch_Q_T = query_batch_transposed.data();

    // then loop over shards of Xs with openMP
    db.I.template balance_shard_loop<search_level, db_vec_level, true>(0, db.I.level_size(search_level), [&](size_t db_shard_start, size_t db_shard_size) {
      // we will have a similarity buffer per thread
      std::vector<float> sim_buffer(1000 * num_query_vecs);

      SeqSearchResult local_results(num_query_vecs, topk);

      // storing best matched frame
      std::vector<size_t> traces;
      // best score of the current question
      std::vector<float> score, score_temp;

      // then batch the shard
      db.I.template batch_loop<search_level, db_vec_level>(db_shard_start, db_shard_size, [&](size_t db_block_start, size_t db_block_size) {

        // get the range of the vectors span of this batch
        const auto [db_vec_offset, num_db_vecs] = db.I.template map_span<search_level, db_vec_level>(db_block_start, db_block_size);

        // now extend the buffer if needed
        sim_buffer.clear();
        sim_buffer.reserve(num_db_vecs * num_query_vecs);


        // then compute batch_Xs @ Q_T (batch)
        // shape (num_db_vecs, d) x (d, num_query_vecs)
        // => (num_db_vecs, num_query_vecs)
        rmmm(
          db.data() + db_vec_offset * d, batch_Q_T, sim_buffer.data(),
          num_db_vecs, num_query_vecs, d
        );

  


        // do the accumulation first (only to the seq_level)
        // get the flatten seq entries
        const auto [db_block_seq_start, db_block_seq_size] = db.I.template map_span<search_level, db_seq_level>(db_block_start, db_block_size);

        if constexpr (db_vec_level - db_seq_level == 0) {
          // 1 vector per sequence entry, no accumulation
          static_assert(query_vec_level - query_seq_level == 0, "multiple vector per query on a 1 vector per entry DB is the same as averaging that vectors directly before searching");
        } else {
          // multiple vectors per sequence entry

          db.I.template span_loop<db_seq_level, db_vec_level>(db_block_seq_start, db_block_seq_size, [&](size_t db_seq_entry_idx, size_t db_vec_local_start, size_t db_vec_local_size) {
            // sim_buffer offset from db_vec_offset
            auto start_ptr = sim_buffer.data() + (db_vec_local_start - db_vec_offset) * num_query_vecs;

            // then set the start_ptr element of sim_buffer as the max of this range
            for (size_t local_vec_idx = 1; local_vec_idx < db_vec_local_size; local_vec_idx++) {
              auto current_ptr = start_ptr + local_vec_idx * num_query_vecs;

// then loop through the query vectors, vectorize over batch
#pragma omp simd
              for (size_t local_query_vec_idx = 0; local_query_vec_idx < num_query_vecs; local_query_vec_idx++) {

                start_ptr[local_query_vec_idx] = std::max(start_ptr[local_query_vec_idx], current_ptr[local_query_vec_idx]);
              }
            }

            // then performc query reduction
            if constexpr (query_seq_level - batch_level != 0) {
              // multiple vectors per query
              queries.I.template span_loop<query_seq_level, query_vec_level>(query_batch_seq_start, query_batch_seq_size, [&](size_t query_seq_entry_idx, size_t query_vec_local_start, size_t query_vec_local_size) {
                // query_vec_local_size is guaranteed to not be 0
                // TODO: perhap we should try filtering it

                auto first_q_ptr = start_ptr + query_vec_local_start - query_vec_offset;

                float sum = 0;

// just add it all to the first element
#pragma omp simd reduction(+ : sum)
                for (size_t local_query_vec_idx = 0; local_query_vec_idx < query_vec_local_size; local_query_vec_idx++) {
                  sum += first_q_ptr[local_query_vec_idx];
                }

                first_q_ptr[0] = sum / query_vec_local_size;
              });
            }
          });
        }

        // over the db (videos)
        db.I.template span_loop<search_level, db_seq_level>(db_block_start, db_block_size, [&](size_t db_entry_idx, size_t db_entry_seq_start, size_t db_entry_seq_size) {
          // number of db entries (frames): db_entry_seq_size in entry (video) db_entry_idx

          // loop over the query
          queries.I.template span_loop<batch_level, query_seq_level>(query_batch_start, query_batch_size, [&](size_t query_entry_idx, size_t query_entry_seq_start, size_t query_entry_seq_size) {
            // number of query entries (subset of frames): query_entry_seq_size in query query_entry_idx

            // temporal matching of query query_entry_idx on video db_entry_idx
            traces.resize((query_entry_seq_size - 1) * db_entry_seq_size, 0);
            score.resize(db_entry_seq_size);

            const size_t first_query_vec_idx = queries.I.template map_idx<query_seq_level, query_vec_level>(query_entry_seq_start);
            // copy the first one to score
            for (size_t db_seq_entry_idx = db_entry_seq_start; db_entry_seq_start < db_entry_seq_start + db_entry_seq_size; db_entry_seq_size++) {
              const size_t db_vec_idx = db.I.template map_idx<db_seq_level, db_vec_level>(db_seq_entry_idx);

              score[db_seq_entry_idx - db_entry_seq_start] = sim_buffer[(db_vec_idx - db_vec_offset) * num_query_vecs + first_query_vec_idx - query_vec_offset];
            }
            // score_temp.resize(db_entry_seq_size);
            score_temp = score;

            for (size_t q_idx = 1; q_idx < query_entry_seq_size; q_idx++) {
              const size_t query_vec_idx = queries.I.template map_idx<query_seq_level, query_vec_level>(query_entry_seq_start + q_idx);

              float max_score = score[0];
              size_t max_score_idx = 0;

              for (size_t db_id = 0; db_id < db_entry_seq_size; db_id++) {
                const size_t db_vec_idx = db.I.template map_idx<db_seq_level, db_vec_level>(db_entry_seq_start + db_id);

                max_score *= discount_rate;

                if (score[db_id] > max_score) {
                  max_score = score[db_id];
                  max_score_idx = db_id;
                }

                // item at indice (i + min_item_dist) can only select
                // indicies <= i
                score_temp[db_id] = max_score + sim_buffer[(db_vec_idx - db_vec_offset) * num_query_vecs + query_vec_idx - query_vec_offset];

                // max_score_idx is relative to the start of the seq
                traces[db_id + (q_idx - 1) * db_entry_seq_size] = max_score_idx;
              }

              // important
              std::swap(score, score_temp);
            }

            // then save the result
            for (size_t db_id = 0; db_id < db_entry_seq_size; db_id++) {
              
              // if (!local_results.can_push(score[db_id], query_entry_idx)) {
              //   continue;
              // }

              // get the best sequence that end with db_id
              std::vector<size_t> _seq_ids(query_entry_seq_size, db_id);

              for (size_t q_idx = query_entry_seq_size - 1; q_idx-- > 0;) {
                // q_idx actually start at query_entry_seq_size - 2
                // traces doesn't trace the 0 q_idx
                _seq_ids[q_idx] = traces[q_idx * query_entry_seq_size + _seq_ids[q_idx + 1]];
              }

              local_results.push({score[db_id],
                                  std::move(_seq_ids)},
                                 query_entry_idx);
            }
          });
        }); }, 1000);

      // then merge the results
#pragma omp critical
      {
        // this will clear the local_topk_result

        topk_result.merge(local_results);
      }
    }); }, 1000);

  return topk_result;
}

template <size_t Q_N, size_t DB_N, size_t batch_level, size_t search_level>
SearchResult search(const RaggedVectors<Q_N> &queries, const RaggedVectors<DB_N> &db, size_t topk) {

  if (queries.d != db.d)
    throw std::runtime_error("dimension mismatch between query and db");

  // different to normal search
  // this support multiple vectors per entries (both query and database)
  // during search it is equivalent to flatenning both tensor to 2d ragged array (the input dim and the last dim)

  // assume that the last level manage the vectors
  static constexpr const size_t query_vec_level = Q_N - 1;
  static constexpr const size_t db_vec_level = DB_N - 1;

  const size_t num_queries = queries.I.level_size(batch_level);
  const size_t d = db.d;

  // use single thread cblas
  // auto original_openblas_num_threads = openblas_get_num_threads();
  // openblas_set_num_threads(1);

  SearchResult topk_result(num_queries, topk);

  // TODO: Ensure sim_buffer memory is aligned for optimal SIMD performance.
  // std::vector<float> query_transposed;

  // loop over batch of questions
  queries.I.template batch_loop<batch_level, query_vec_level>(0, num_queries, [&](size_t query_batch_start, size_t query_batch_size) {
    // get the start and size of the actual query
    auto [query_vec_offset, num_query_vecs] = queries.I.template map_span<batch_level>(query_batch_start, query_batch_size);

    // query_transposed.clear();
    // query_transposed.reserve(num_query_vecs * d);

    // // transpose this batch of query
    // tranpose(queries.data() + query_vec_offset * d, query_transposed.data(), num_query_vecs, d);
    // // shape: (d, num_query_vecs)
    // const float *batch_Q_T = query_transposed.data();

    // then loop over shards of Xs with openMP
    db.I.template balance_shard_loop<search_level, db_vec_level, true>(0, db.I.level_size(search_level), [&](size_t db_shard_start, size_t db_shard_size) {
      // we will have a similarity buffer per thread
      std::vector<float> sim_buffer(V_BATCH_SIZE_X * num_query_vecs);

      SearchResult local_results(num_queries, topk);


      // then batch the shard
      db.I.template batch_loop<search_level, db_vec_level>(db_shard_start, db_shard_size, [&](size_t db_block_start, size_t db_block_size) {
        // then map this batch to get the actual vectors
        // auto [db_vec_offset, num_db_vecs] = db.I.map_span(search_level, db_block_start, db_block_size, db_vec_level);
        auto [db_vec_offset, num_db_vecs] = db.I.template map_span<search_level, db_vec_level>(db_block_start, db_block_size);
        
        if (sim_buffer.size() < num_db_vecs * num_query_vecs) {
          // now extend the buffer if needed
          sim_buffer.clear();
          sim_buffer.resize(num_db_vecs * num_query_vecs);
        }

        // then compute batch_Xs @ Q_T (batch)
        // shape (num_db_vecs, d) x (d, num_query_vecs)
        // => (num_db_vecs, num_query_vecs)
        similarity_dot_product_trans_out(
            queries.data() + query_vec_offset * d,
            db.data() + db_vec_offset * d,
            sim_buffer.data(),
            num_query_vecs, num_db_vecs, d);

        if constexpr (db_vec_level - search_level == 0) {
          // 1 vector per database entry
          static_assert(query_vec_level - batch_level == 0, "multiple vector per query on a 1 vector per entry DB is the same as averaging that vectors directly before searching");

          // 1 vector per entry
          // span (db_block_start, db_block_size) == (db_vec_offset, num_db_vecs)

          for_loop(db_block_start, db_block_size, [&](size_t db_entry_idx) {
            for_loop(query_batch_start, query_batch_size, [&](size_t query_entry_idx) {
              // TODO: the main overhead is here

              local_results.push(
                  ItemScore(sim_buffer[(db_entry_idx - db_vec_offset) * num_query_vecs + query_entry_idx - query_vec_offset], db_entry_idx),
                  query_entry_idx);
            });
          });

        } else {

          // multiple vectors per database entry
          // use max reduction
          db.I.template span_loop<search_level, db_vec_level>(db_block_start, db_block_size, [&](size_t db_entry_idx, size_t db_vec_local_start, size_t db_vec_local_size ){
            // span (db_vec_local_start, db_vec_local_size) is of db_entry_idx
  
            // sim_buffer offset from db_vec_offset
            auto start_ptr = sim_buffer.data() + (db_vec_local_start - db_vec_offset) * num_query_vecs;
  
            // then set the start_ptr element of sim_buffer as the max of this range
            for (size_t local_vec_idx = 1; local_vec_idx < db_vec_local_size; local_vec_idx++){
              auto current_ptr = start_ptr + local_vec_idx * num_query_vecs;

// then loop through the query vectors, vectorize over batch
#pragma omp simd
                  for (size_t local_query_vec_idx = 0; local_query_vec_idx < num_query_vecs; local_query_vec_idx++){
                    
                    start_ptr[local_query_vec_idx] = std::max(start_ptr[local_query_vec_idx], current_ptr[local_query_vec_idx]);
                  
                  }
            }

            if constexpr (query_vec_level - batch_level == 0) {
              // 1 vector per query, just save the result
              // span (query_vec_offset, num_query_vecs) == (query_batch_start, query_batch_size)
              
              for (size_t query_entry_idx = query_batch_start; query_entry_idx < query_batch_start+query_batch_size; query_entry_idx++){
                
                local_results.push(
                  ItemScore(start_ptr[query_entry_idx - query_batch_start], db_entry_idx),
                  query_entry_idx
                );
              }
            } else{
              // multiple vectors per query
              // use mean reduction (over vectors per query)

              queries.I.template span_loop<search_level, query_vec_level>(query_batch_start, query_batch_size, [&](size_t query_entry_idx, size_t query_vec_local_start, size_t query_vec_local_size){
                // query_vec_local_size is guaranteed to not be 0
                // TODO: perhap we should try filtering it

                auto first_q_ptr = start_ptr + query_vec_local_start - query_vec_offset;

                float sum = 0;

// just add it all to the first element
#pragma omp simd reduction(+ : sum)
                for (size_t local_query_vec_idx = 0; local_query_vec_idx < query_vec_local_size; local_query_vec_idx++){
                  sum += first_q_ptr[local_query_vec_idx];
                }

                // then save the result
                local_results.push(
                  ItemScore(sum/query_vec_local_size, db_entry_idx),
                  query_entry_idx
                );
                
              });
            }
          });
        }

      },
                                    V_BATCH_SIZE_X);

      // then merge the results
#pragma omp critical
        {
          // this will clear the local_topk_result
          topk_result.merge(local_results);
        }

    }); }, V_BATCH_SIZE_Q);

  // openblas_set_num_threads(original_openblas_num_threads);
  topk_result.sort();

  return topk_result;
}

} // namespace seqr