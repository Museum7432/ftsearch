#include "FlatSearch.h"
#include "RaggedArray.h"
#include "half.hpp"
#include "utils.h"
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

// SearchResult search_baseline(const float *Q, const size_t nq, const half_float::half *Xs, const size_t nv, const size_t d, const size_t topk) {
//   SearchResult topk_result(nq, topk);

//   const constexpr size_t BATCH_SIZE_Q = 1024;
//   const constexpr size_t BATCH_SIZE_X = 1024;
//   const constexpr size_t FP32_CACHE_SIZE = 16;

//   batch_loop(0, nq, BATCH_SIZE_Q, [&](size_t query_batch_start, size_t query_batch_size) {
//     shard_loop<true>(0, nv, [&](size_t db_shard_start, size_t db_shard_size) {
//       std::vector<float> sim_buffer(BATCH_SIZE_X * query_batch_size);
//       SearchResult local_results(nq, topk);

//       std::vector<float> fp32_cache_buffer(FP32_CACHE_SIZE * d);

//       batch_loop(db_shard_start, db_shard_size, V_BATCH_SIZE_X, [&](size_t db_block_start, size_t db_block_size) {
//         if (sim_buffer.size() < db_block_size * query_batch_size) {

//           sim_buffer.clear();
//           sim_buffer.resize(db_block_size * query_batch_size);
//         }

//         batch_loop(db_block_start, db_block_size, FP32_CACHE_SIZE, [&](size_t db_fp16_start, size_t db_fp16_size) {
//           const half_float::half *fp16_start_ptr = Xs + db_fp16_start * d;

//           for (size_t i = 0; i < db_fp16_size; i++) {
//             fp32_cache_buffer[i] = static_cast<float>(fp16_start_ptr[i]);
//           }

//           similarity_dot_product_trans_out(
//               Q + query_batch_start * d,
//               fp32_cache_buffer.data(),
//               sim_buffer.data() + (db_fp16_start - db_block_start) * query_batch_size,
//               query_batch_size, db_fp16_size, d);
//         });

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

TEST(search_it, t1) {

  const constexpr size_t d = 1024;
  const constexpr size_t N = 4;
  const constexpr size_t topk = 50;
  const constexpr size_t nq = 10;
  const constexpr size_t n_runs = 100;

  seqr::RaggedVectors<N> db(d);

  auto &I = db.I;

  size_t t0 = std::rand() % 40;
  for (size_t i0 = 0; i0 < t0; i0++) {
    I.open();
    size_t t1 = std::rand() % 10;
    for (size_t i1 = 0; i1 < t1; i1++) {
      I.open();
      size_t t2 = std::rand() % 50;
      for (size_t i2 = 0; i2 < t2; i2++) {

        I.open();
        size_t t3 = std::rand() % 47 + 1;
        I.add(t3);
        I.close();
      }
      I.close();
    }
    I.close();
  }

  auto sw = seqr::StopWatch();

  const size_t n_vecs = I.level_size(N - 1);

  std::vector<float> t;
  t.resize(n_vecs * d);

  auto index_size_gb = seqr::estimate_vector_size_GB(t);

  std::cout
      << "n_vecs: " << n_vecs << ": " << index_size_gb << " GB" << std::endl;

  sw.split_print("init: ");

  // test fill
  seqr::random_fill(t.data(), t.size());

  db.ptr = t.data();

  sw.split_print("fill: ");

  // test 1 vector search
  seqr::RaggedVectors<1> Q(d);

  Q.I.add(nq);

  std::cout << "n_vecs Q: " << nq << std::endl;
  std::cout << "topk: " << topk << std::endl;

  // get 1 random vector from db
  size_t q_offset = std::rand() % (n_vecs - 4);
  Q.ptr = t.data() + q_offset * d;

  sw.reset();

  seqr::search<1, N, 0, 3>(Q, db, topk);

  float baseline_time = 0, target_time = 0;

  for (size_t i = 0; i < n_runs; i++) {
    sw.reset();
    auto re1 = seqr::search<1, N, 0, 3>(Q, db, topk);
    target_time += sw.split();
  }

  for (size_t i = 0; i < n_runs; i++) {

    sw.reset();
    auto re2 = seqr::search_baseline(Q.data(), nq, db.data(), n_vecs, d, topk);
    baseline_time += sw.split();
  }

  // for (size_t i = 0; i < n_runs; i++) {

  //   sw.reset();
  //   auto re1 = seqr::search<1, N, 0, 3>(Q, db, topk);
  //   target_time += sw.split();

  //   sw.reset();
  //   auto re2 = seqr::search_baseline(Q.data(), nq, db.data(), n_vecs, d, topk);
  //   baseline_time += sw.split();

  //   for (size_t q_i = 0; q_i < nq; q_i++) {
  //     EXPECT_EQ(re1.top_ks[q_i].heap[0].id_, q_offset + q_i);

  //     EXPECT_EQ(re2.top_ks[q_i].heap[0].id_, q_offset + q_i);
  //   }
  // }

  baseline_time /= n_runs;
  std::cout << "baseline: " << baseline_time << " ms " << index_size_gb * 1000 / baseline_time << " GB/s" << std::endl;

  target_time /= n_runs;
  std::cout << "current: " << target_time << " ms " << index_size_gb * 1000 / target_time << " GB/s" << std::endl;
}

// TEST(search_it, fp16) {

//   const constexpr size_t d = 1024;
//   const constexpr size_t topk = 50;
//   const constexpr size_t nq = 20;
//   const constexpr size_t nv = 2000000;
//   const constexpr size_t n_runs = 5;

//   auto sw = seqr::StopWatch();

//   std::vector<half_float::half> Xs_fp16(nv * d);
//   std::vector<float> Xs(nv * d);

//   std::vector<float> Q(nq * d);

//   sw.split_print("init: ");

//   seqr::random_fill(Xs.data(), Xs.size(), 13);
//   // seqr::random_fill_fp16(Xs_fp16.data(), Xs.size(), 13);

//   seqr::random_fill(Q.data(), Q.size(), 10);

// #pragma omp parallel for
//   for (size_t i = 0; i < Xs.size(); i++) {
//     Xs_fp16[i] = Xs[i];
//   }

//   sw.split_print("fill: ");

//   std::cout << "num vecs: " << nv << " dim: " << d << std::endl;
//   std::cout << "num query: " << nq << std::endl;
//   std::cout << "topk: " << topk << std::endl;

//   float fp16_time = 0, fp32_time = 0;

//   for (size_t i = 0; i < n_runs; i++) {
//     sw.reset();
//     seqr::search_baseline(Q.data(), nq, Xs.data(), nv, d, topk);
//     fp32_time += sw.split();
//   }

//   for (size_t i = 0; i < n_runs; i++) {
//     sw.reset();
//     seqr::search_baseline(Q.data(), nq, Xs_fp16.data(), nv, d, topk);
//     fp16_time += sw.split();
//   }

//   fp32_time /= n_runs;
//   fp16_time /= n_runs;
//   std::cout << "fp16 search: " << fp16_time << " ms " << std::endl;
//   std::cout << "fp32 search: " << fp32_time << " ms " << std::endl;
// }

TEST(search_it, fp16) {

  const constexpr size_t d = 1024;
  const constexpr size_t topk = 50;
  const constexpr size_t nq = 50;
  const constexpr size_t nv = 2000000;
  const constexpr size_t n_runs = 20;

  auto sw = seqr::StopWatch();

  std::vector<half_float::half> Xs_fp16(nv * d);

  std::vector<float> Q(nq * d);

  sw.split_print("init: ");

  seqr::random_fill_fp16(Xs_fp16.data(), Xs_fp16.size(), 13);

  seqr::random_fill(Q.data(), Q.size(), 10);

  sw.split_print("fill: ");

  std::cout << "num vecs: " << nv << " dim: " << d << " - " << seqr::estimate_vector_size_GB(Xs_fp16) << " GB" << std::endl;
  std::cout << "num query: " << nq << std::endl;
  std::cout << "topk: " << topk << std::endl;

  float fp16_time = 0;

  for (size_t i = 0; i < n_runs; i++) {
    sw.reset();
    seqr::search_baseline(Q.data(), nq, Xs_fp16.data(), nv, d, topk);
    fp16_time += sw.split();
  }

  fp16_time /= n_runs;
  std::cout << "fp16 search: " << fp16_time << " ms " << std::endl;
}

TEST(search_it, fp32) {

  const constexpr size_t d = 1024;
  const constexpr size_t topk = 50;
  const constexpr size_t nq = 50;
  const constexpr size_t nv = 2000000;
  const constexpr size_t n_runs = 20;

  auto sw = seqr::StopWatch();

  std::vector<float> Xs(nv * d);

  std::vector<float> Q(nq * d);

  sw.split_print("init: ");

  seqr::random_fill(Xs.data(), Xs.size(), 13);

  seqr::random_fill(Q.data(), Q.size(), 10);

  sw.split_print("fill: ");

  std::cout << "num vecs: " << nv << " dim: " << d << " - " << seqr::estimate_vector_size_GB(Xs) << " GB" << std::endl;
  std::cout << "num query: " << nq << std::endl;
  std::cout << "topk: " << topk << std::endl;

  float fp16_time = 0, fp32_time = 0;

  for (size_t i = 0; i < n_runs; i++) {
    sw.reset();
    seqr::search_baseline(Q.data(), nq, Xs.data(), nv, d, topk);
    fp32_time += sw.split();
  }

  fp32_time /= n_runs;
  std::cout << "fp32 search: " << fp32_time << " ms " << std::endl;
}