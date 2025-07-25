#pragma once
#include "types.h"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <queue>
#include <vector>

namespace seqr {

struct Score {
  float score;

  // Define a comparison operator for max heap
  bool operator<(const Score &other) const;
  bool operator>(const Score &other) const;

  void map_id(const SeqInfos &info);
};

// for topk queue
struct ItemScore : Score {
  size_t seq_id;
  size_t vec_id; // relative w.r.t seq_id

  size_t abs_vec_id;

  ItemScore(float score_, size_t abs_vec_id_);

  void map_id(const SeqInfos &info);
};

struct SeqScore : Score {
  size_t seq_id;
  id_arr vec_ids; // relative w.r.t seq_id

  id_arr abs_vec_ids;

  SeqScore(float score_, id_arr abs_vec_ids_);
  void map_id(const SeqInfos &info);
};

template <typename Item>
struct TopK {

  // min heap since we only need to track the lowest score in top K
  std::vector<Item> heap;
  size_t k;

  TopK(size_t k_);

  void push(const Item &value);

  size_t size() const;

  // Item toppop();
  // const Item &top() const;

  // bool empty() const;

  // merge the other topK into it
  void merge(TopK<Item> &other);

  void reset();

  void map_id(const SeqInfos &info);
};

template <typename Item>
struct BatchTopK {

  std::vector<TopK<Item>> top_ks;
  size_t batch_size;

  BatchTopK(size_t batch_size, size_t k_);

  void push(const Item &value, size_t batch_idx);

  void merge(BatchTopK<Item> &other);

  void reset();

  void map_id(const SeqInfos &info);
};

using ItemHeap = BatchTopK<ItemScore>;
using SeqHeap = BatchTopK<SeqScore>;

using SearchResult = BatchTopK<ItemScore>;
using SeqSearchResult = BatchTopK<SeqScore>;

} // namespace seqr

// implementation of the template struct
namespace seqr {

template <typename Item>
TopK<Item>::TopK(size_t k_) : k(k_) {
  heap.reserve(k_);
}

template <typename Item>
void TopK<Item>::push(const Item &value) {

  if (size() < k) {
    heap.push_back(value);
    std::push_heap(heap.begin(), heap.end(), std::greater<>());

  } else if (value > heap.front()) {
    // push the root to the final position
    std::pop_heap(heap.begin(), heap.end(), std::greater<>());

    heap.back() = value;

    std::push_heap(heap.begin(), heap.end(), std::greater<>());
  }
}

template <typename Item>
size_t TopK<Item>::size() const {
  return heap.size();
}

template <typename Item>
void TopK<Item>::reset() {
  heap.clear();
}

template <typename Item>
void TopK<Item>::merge(TopK<Item> &other) {
  for (auto i : other.heap) {
    push(i);
  }
  // clear the other heap
  other.reset();
}

template <typename Item>
void TopK<Item>::map_id(const SeqInfos &info) {
  for (int i = 0; i < heap.size(); i++) {
    heap[i].map_id(info);
  }
}

template <typename Item>
BatchTopK<Item>::BatchTopK(size_t batch_size, size_t k_) {
  for (size_t i = 0; i < batch_size; i++) {
    top_ks.emplace_back(k_);
  }
}

template <typename Item>
void BatchTopK<Item>::push(const Item &value, size_t batch_idx) {
  top_ks[batch_idx].push(value);
}

template <typename Item>
void BatchTopK<Item>::merge(BatchTopK<Item> &other) {
  for (size_t i = 0; i < batch_size; i++) {
    top_ks[i].merge(other.top_ks[i]);
  }
}

template <typename Item>
void BatchTopK<Item>::reset() {
  for (size_t i = 0; i < batch_size; i++) {
    top_ks[i].reset();
  }
}

template <typename Item>
void BatchTopK<Item>::map_id(const SeqInfos &info) {
  for (int i = 0; i < top_ks.size(); i++) {
    top_ks[i].map_id(info);
  }
}

} // namespace seqr