#include "types.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <tuple>

namespace seqr {

SeqInfos::SeqInfos() {
  start_ids = {0};
}

void SeqInfos::add(size_t n) {
  size_t next_start_idx = start_ids.back() + n;
  start_ids.push_back(next_start_idx);
}

void SeqInfos::reset() {
  start_ids.clear();
  start_ids = {0};
}

size_t SeqInfos::num_seqs() const {
  return start_ids.size() - 1;
}

size_t SeqInfos::num_items() const {
  return start_ids.back();
}

std::tuple<size_t, size_t> SeqInfos::idx2seq(size_t item_idx) const {

  assert(item_idx >= 0 && item_idx < num_items());
  // get the closest start indice that is > item_idx
  auto it = std::upper_bound(start_ids.begin(), start_ids.end(), item_idx);

  --it;

  return std::make_tuple(it - start_ids.begin(), item_idx - *it);
}

size_t SeqInfos::num_items(size_t seq_id) const {
  return start_ids[seq_id + 1] - start_ids[seq_id];
}

size_t SeqInfos::get_end(size_t seq_id) const {
  return start_ids[seq_id + 1] - 1;
}
size_t SeqInfos::get_start(size_t seq_id) const {
  return start_ids[seq_id];
}

} // namespace seqr