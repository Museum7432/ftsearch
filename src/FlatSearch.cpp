// #include "FlatSearch.h"
// #include "RaggedArray.h"
// #include "SearchResult.h"
// #include "cblas.h"
// #include "utils.h"
// #include <cstddef>
// #include <iostream>
// #include <utility>
// #include <vector>

// #define BATCH_SIZE_Q 1024
// #define BATCH_SIZE_X 1024

// namespace seqr {

// SearchResult search_baseline(const float *Q, const size_t nq, const float *Xs, const size_t nv, const size_t d, const size_t topk) {

//   SearchResult topk_result(nq, topk);

//   batch_loop(0, nq, BATCH_SIZE_Q, [&](size_t query_batch_start, size_t query_batch_size) {
//     shard_loop<true>(0, nv, [&](size_t db_shard_start, size_t db_shard_size) {
//       std::vector<float> sim_buffer(BATCH_SIZE_X * query_batch_size);
//       SearchResult local_results(nq, topk);

//       batch_loop(db_shard_start, db_shard_size, V_BATCH_SIZE_X, [&](size_t db_block_start, size_t db_block_size) {
//         if (sim_buffer.size() < db_block_size * query_batch_size) {

//           sim_buffer.clear();
//           sim_buffer.resize(db_block_size * query_batch_size);
//         }

//         // then compute batch_Xs @ Q
//         // => (db_block_size, query_batch_size)
//         similarity_dot_product_trans_out(
//             Q + query_batch_start * d,
//             Xs + db_block_start * d,
//             sim_buffer.data(),
//             query_batch_size, db_block_size, d);

//         for_loop(db_block_start, db_block_size, [&](size_t db_i) {
//           for_loop(query_batch_start, query_batch_size, [&](size_t q_i) {
//             local_results.push(
//                 ItemScore(sim_buffer[(q_i - query_batch_start) + (db_i - db_block_start) * query_batch_size], db_i),
//                 q_i);
//           });
//         });
//       });

// #pragma omp critical
//       {
//         // this will clear the local_topk_result
//         topk_result.merge(local_results);
//       }
//     });
//   });

//   topk_result.sort();

//   return topk_result;
// }

// } // namespace seqr