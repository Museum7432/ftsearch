/*
like faiss FlatIndex but has sequence search
only support inner product for now

store sequences of vectors instead of vectors
*/
#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include "ScoreQueue.h"
#include "types.h"

static_assert(sizeof(float) == 4, "float is not 32 bits");

namespace seqr {

struct FlatIndexIP {

  // number of vector in 1 batch,
  // will be ignored in seq_search if one sequence is longer than it
  size_t batch_size_x = 10000;

  // number of query in 1 batch
  size_t batch_size_q = 1000;

  // vector dimension
  size_t d;

  std::vector<float> flatten_vecs;

  // starting indices of sequences,
  // the last value is just to store the ending of the last sequence
  // std::vector<size_t> start_ids;
  SeqInfos info;

  FlatIndexIP(size_t d);

  void add_seq(cfloat *xs, size_t n);

  void reset();
  // get vector
  std::vector<float> get_vector(const size_t seq_id, const size_t vec_id) const;

  // map the abs_vec_id to seq id and vec id
  std::tuple<size_t, size_t> map_idx(size_t abs_vec_id) const;

  const size_t num_vectors() const;
  const size_t num_seqs() const;

  // perform batch search, similar to faiss
  // (score, seq_id, vec_id)
  SearchResult search(cfloat *Q, size_t nq, size_t topk) const;

  // search for a subset of sequence in the index
  SeqSearchResult seq_search(cfloat *Q, const size_t nq, const size_t topk, cfloat discount_rate = 1) const;

  // TODO: subset search
};
} // namespace seqr
