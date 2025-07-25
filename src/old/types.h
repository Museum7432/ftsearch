#pragma once
#include <cstddef>
#include <vector>

namespace seqr {
using id_arr = std::vector<size_t>;
using search_result = std::tuple<std::vector<float>, id_arr, id_arr>;
using cfloat = const float;

// for storing infos about sequences of vector (starting, ending)
struct SeqInfos {
  std::vector<size_t> start_ids;

  SeqInfos();

  // add a sequence of length n
  void add(size_t n);

  void reset();

  size_t num_seqs() const;
  size_t num_items() const;

  size_t num_items(size_t seq_id) const;

  size_t get_start(size_t seq_id) const;
  size_t get_end(size_t seq_id) const;

  // map the flatten indice to the sequence id
  std::tuple<size_t, size_t> idx2seq(size_t item_idx) const;
};

} // namespace seqr