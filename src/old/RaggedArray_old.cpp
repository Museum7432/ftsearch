// #include "RaggedArray.h"
// #include <algorithm>
// #include <cassert>
// #include <cstddef>
// #include <iostream>
// #include <ostream>
// #include <stdexcept>
// #include <tuple>

// #include "utils.h"

// namespace seqr {

// RaggedIndexer::RaggedIndexer(size_t num_dims) : num_dims(num_dims), current_level(0) {

//   if (num_dims < 2)
//     throw std::runtime_error("1D ragged array is not supported");

//   for (size_t i = 0; i + 1 < num_dims; i++) {
//     // the last value is the total size
//     level_indices.push_back({0});
//   }
// }

// void RaggedIndexer::reset() {
//   for (auto &l : level_indices) {
//     l.clear();
//     l.push_back(0);
//   }
//   current_level = 0;
// }

// void RaggedIndexer::open() {
//   if (current_level + 1 >= num_dims)
//     throw std::runtime_error("exceed maximum bracket depth");
//   // if there is a previous level, extend the last element
//   if (current_level > 0) {
//     level_indices[current_level - 1].back()++;
//   }
//   // add a new entry to the current level then move to the next one
//   level_indices[current_level].push_back(level_indices[current_level].back());
//   current_level++;
// }

// void RaggedIndexer::close() {
//   if (current_level <= 0)
//     throw std::runtime_error("exceed minimum bracket depth");
//   // moving back 1 level
//   current_level--;
// }

// void RaggedIndexer::add(size_t n) {
//   if (current_level + 1 != num_dims)
//     throw std::runtime_error("only add items at last level");
//   // if we are at the last level, just add to the count
//   level_indices[current_level - 1].back() += n;
// }

// size_t RaggedIndexer::level_size(size_t d) const {
//   if (d >= num_dims)
//     throw std::runtime_error("exceed maximum level");

//   if (d + 1 == num_dims)
//     return level_indices.back().back();

//   return level_indices[d].size() - 1;
// }
// size_t RaggedIndexer::map_idx(size_t idx, size_t from_level, size_t to_level) const {
//   if (from_level > to_level)
//     throw std::runtime_error("only for upper level to lower level");

//   if (to_level >= num_dims)
//     throw std::runtime_error("exceed maximum level");

//   size_t mapped_idx = idx;

//   for (size_t l = from_level; l < to_level; l++) {
//     mapped_idx = level_indices[l][mapped_idx];
//   }
//   return mapped_idx;
// }

// size_t RaggedIndexer::map_idx_rev(size_t idx, size_t from_level, size_t to_level) const {

//   if (from_level < to_level)
//     throw std::runtime_error("only for lower level to upper level");

//   if (from_level >= num_dims)
//     throw std::runtime_error("exceed maximum level");

//   if (idx >= level_size(from_level))
//     throw std::runtime_error("id out of bound");

//   // get the closest start indice that is > than the map id
//   auto it = std::upper_bound(level_indices[to_level].begin(), level_indices[to_level].end(), idx, [&](size_t needle, size_t candidate) {
//     return needle < map_idx(candidate, to_level + 1, from_level);
//   });

//   --it;

//   return it - level_indices[to_level].begin();
// }
// std::tuple<size_t, size_t> RaggedIndexer::map_span(size_t from_level, size_t start_, size_t size_, size_t to_level) const {
//   // TODO: validate the input
//   size_t mapped_start_ = map_idx(start_, from_level, to_level);
//   // exclusive
//   size_t end_ = start_ + size_;
//   size_t next_mapped_start_ = map_idx(end_, from_level, to_level);

//   return std::make_tuple(mapped_start_, next_mapped_start_ - mapped_start_);
// }

// std::tuple<size_t, size_t> RaggedIndexer::get_shard(size_t start_, size_t size_, size_t level_, size_t size_level, size_t n_workers, size_t worker_id) const {
//   // if (level_ >= size_level)
//   // throw std::runtime_error("when level_ == size_level, just use normal sharding");
//   // map the span to size_level for dividing the range

//   auto [f_start_, f_size_] = map_span(level_, start_, size_, size_level);
//   // then get the ideal start and end
//   auto [f_start_shard, f_size_shard] = get_shard_range(f_start_, f_size_, n_workers, worker_id);

//   // inclusive
//   auto f_end_shard = f_start_shard + f_size_shard;

//   if (level_ + 1 == num_dims)
//     return std::make_tuple(f_start_shard, f_size_shard);

//   auto it_begin = level_indices[level_].begin() + start_, it_end = level_indices[level_].begin() + start_ + size_;

//   auto start_shard = std::lower_bound(
//                          it_begin,
//                          it_end,
//                          f_start_shard,
//                          [&](size_t candidate, size_t needle) {
//                            // the candidate is value of level_indices[level_] so it map level_ + 1
//                            return map_idx(candidate, level_ + 1, size_level) < needle;
//                          }) -
//                      level_indices[level_].begin();

//   auto end_shard = std::lower_bound(
//                        it_begin,
//                        it_end,
//                        f_end_shard,
//                        [&](size_t candidate, size_t needle) {
//                          return map_idx(candidate, level_ + 1, size_level) < needle;
//                        }) -
//                    level_indices[level_].begin();

//   // just to be sure
//   if (worker_id == 0)
//     start_shard = start_;

//   if (worker_id + 1 == n_workers)
//     end_shard = start_ + size_;

//   return std::make_tuple(start_shard, end_shard - start_shard);
// }

// RaggedVectors::RaggedVectors(size_t vd, size_t seq_d) : d(vd), I(seq_d) {}

// void RaggedVectors::add(const float *xs, size_t n) {
//   if (ptr != nullptr)
//     throw std::runtime_error("can't add items to a non own array");
//   if (n == 0)
//     return;
//   owned.insert(owned.end(), xs, xs + n * d);
// }

// void RaggedVectors::reset() {
//   I.reset();
//   owned.clear();
// }

// bool RaggedVectors::ready() const {
//   // no idea...
//   if (ptr != nullptr)
//     return true;

//   if (I.level_size(d - 1) == owned.size() / d)
//     return true;

//   return false;
// }

// const float *RaggedVectors::data() const {
//   if (ptr != nullptr)
//     return ptr;
//   return owned.data();
// }

// }; // namespace seqr