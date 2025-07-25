#include "ScoreQueue.h"
#include "types.h"
#include <cstddef>

namespace seqr {
bool Score::operator<(const Score &other) const { return score < other.score; }
bool Score::operator>(const Score &other) const { return score > other.score; }
void Score::map_id(const SeqInfos &info) {}

ItemScore::ItemScore(float score_, size_t abs_vec_id_)
    : Score{score_}, vec_id(abs_vec_id_) {}

SeqScore::SeqScore(float score_, id_arr abs_vec_ids_)
    : Score{score_}, abs_vec_ids(abs_vec_ids_) {}

void ItemScore::map_id(const SeqInfos &info) {
  auto [seq_id_, vec_id_] = info.idx2seq(abs_vec_id);
  seq_id = seq_id_;
  vec_id = vec_id_;
}

void SeqScore::map_id(const SeqInfos &info) {
  size_t tmp = abs_vec_ids.front();
  auto [seq_id_, tmp_mapped] = info.idx2seq(tmp);
  seq_id = seq_id_;

  vec_ids.clear();
  for (int i = 0; i < abs_vec_ids.size(); i++) {
    vec_ids.push_back(abs_vec_ids[i] - tmp + tmp_mapped);
  }
}
} // namespace seqr