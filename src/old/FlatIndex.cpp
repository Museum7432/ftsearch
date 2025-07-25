#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "FlatIndex.h"
#include "ScoreQueue.h"
#include "types.h"
#include "utils.h"

#include "cblas.h"
#include "omp.h"

namespace seqr {
FlatIndexIP::FlatIndexIP(size_t d) : d(d) {
}

const size_t FlatIndexIP::num_vectors() const {
  return info.num_items();
}

const size_t FlatIndexIP::num_seqs() const { return info.num_seqs(); }

void FlatIndexIP::add_seq(cfloat *xs, size_t n) {
  if (n == 0)
    return;

  flatten_vecs.insert(flatten_vecs.end(), xs, xs + n * d);

  info.add(n);
}

void FlatIndexIP::reset() {
  flatten_vecs.clear();
  info.reset();
}

std::tuple<size_t, size_t> FlatIndexIP::map_idx(size_t abs_vec_id) const {

  return info.idx2seq(abs_vec_id);
}

SearchResult FlatIndexIP::search(cfloat *Q, size_t nq, size_t topk) const {

  const size_t n_seq = num_seqs(), n_vec = num_vectors();

  SearchResult topk_result(nq, topk);

  // keys (n_vec, d)
  cfloat *K = flatten_vecs.data();

  // pretranpose Q (nq, d)
  auto Q_tranpose = std::vector<float>(d * nq);

  batch_loop(0, nq, batch_size_q, [&](size_t bi, size_t bnq) {
    tranpose(Q + bi * d, Q_tranpose.data() + bi * d, bnq, d);
  });
  cfloat *Q_T = Q_tranpose.data();

  shard_loop(0, n_vec, [&](size_t shard_i, size_t shard_nv) {

    std::vector<float> Sims(std::min(batch_size_q, nq) * std::min(batch_size_x, n_vec));
    float *S = Sims.data();

    SearchResult local_topk_result(nq, topk);

    // then we batch the query
    batch_loop(0, nq, batch_size_q, [&](size_t bi, size_t bnq) {
      // and the shard
      batch_loop(shard_i, shard_nv, batch_size_x, [&](size_t bj, size_t bnv) {
        // calculate S = K @ Q_T
        // shape: (bnv, d) x (d, bnq)
        // S: (bnv, bnq)

            rmmm(
              K + bj * d, Q_T + bi * d, S, 
              bnv, d, bnq
            );

        // then save the result to heap
        for (size_t i = 0; i < bnq; i++) {
          for (size_t j = 0; j < bnv; j++) {
            // we will convert the vec_id to the relative one latter
            local_topk_result.push(ItemScore(S[j * bnq + i],  j + bj), bi + i);
          }
        }
      });

#pragma omp critical
      {
        // this will clear the local_topk_result
        topk_result.merge(local_topk_result);
      }

    }); }, true, 0);

  topk_result.map_id(info);

  return topk_result;
}

SeqSearchResult FlatIndexIP::seq_search(cfloat *Q, const size_t nq, const size_t topk, cfloat discount_rate) const{
  
}

} // namespace seqr