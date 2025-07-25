#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>
namespace seqr {

struct Score {
  float score;

  // Define a comparison operator for max heap
  inline bool operator<(const Score &other) const;
  inline bool operator<(const float other_score) const;

  inline bool operator>(const Score &other) const;
  inline bool operator>(const float other_score) const;
};

// for topk queue
struct ItemScore : Score {

  size_t id_;

  ItemScore(float score_ = 0, size_t id_ = 0);
};

struct SeqScore : Score {
  std::vector<size_t> ids_; // relative w.r.t seq_id

  SeqScore(float score_ = 0, std::vector<size_t> ids_ = {});
};

template <typename Item>
struct TopK {

  // min heap since we only need to track the lowest score in top K
  std::vector<Item> heap;
  size_t k;

  TopK(size_t k_);

  inline void push(const Item &value);

  size_t size() const;

  bool can_push(const float score) const;

  // Item toppop();
  // const Item &top() const;

  // bool empty() const;

  // merge the other topK into it
  inline void merge(TopK<Item> &other);

  void sort();

  void reset();
};

template <typename Item>
struct BufferTopK {

  std::vector<ItemScore> buffer;
  const size_t k;

  BufferTopK(size_t k_);

  // copy push
  void push(const Item &value);

  // move push
  void push(Item &&value);

  template <typename... Args>
  void emplace(Args &&...args);

  // flush all item outside of the topK
  void flush();

  void merge(BufferTopK<Item> &other);
  void sort();
  void reset();

  bool empty() const;

  // bool can_push(const float score) const;
};

template <typename Item>
struct BatchTopK {

  std::vector<TopK<Item>> top_ks;
  // std::vector<BufferTopK<Item>> top_ks;
  size_t batch_size;

  BatchTopK(size_t batch_size, size_t k_);

  inline void push(const Item &value, size_t batch_idx);

  // inline bool can_push(const float score, size_t batch_idx) const;

  inline void merge(BatchTopK<Item> &other);

  void sort();

  void reset();
};

using SearchResult = BatchTopK<ItemScore>;
using SeqSearchResult = BatchTopK<SeqScore>;

} // namespace seqr

// implementation of the template struct
namespace seqr {

inline bool Score::operator<(const Score &other) const { return score < other.score; }
inline bool Score::operator<(const float other_score) const { return score < other_score; }

inline bool Score::operator>(const Score &other) const { return score > other.score; }
inline bool Score::operator>(const float other_score) const { return score > other_score; }

template <typename Item>
TopK<Item>::TopK(size_t k_) : k(k_) {
  heap.reserve(k_);
}

template <typename Item>
void TopK<Item>::sort() {
  std::sort(heap.begin(), heap.end(), std::greater<>());
}

template <typename Item>
inline void TopK<Item>::push(const Item &value) {

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
bool TopK<Item>::can_push(const float score) const {
  if (size() < k) {
    return true;
  }
  if (score > heap.front()) {
    return true;
  }
  return false;
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
BufferTopK<Item>::BufferTopK(size_t k_) : k(k_) {
  buffer.reserve(4 * k_);
}

template <typename Item>
void BufferTopK<Item>::push(const Item &value) {
  if (buffer.size() >= 4 * k)
    flush();

  buffer.push_back(value);
}

template <typename Item>
void BufferTopK<Item>::push(Item &&value) {
  if (buffer.size() >= 4 * k)
    flush();

  buffer.push_back(std::move(value));
}

template <typename Item>
template <typename... Args>
void BufferTopK<Item>::emplace(Args &&...args) {
  if (buffer.size() >= 4 * k) {
    flush();
  }
  buffer.emplace_back(std::forward<Args>(args)...); // Construct in-place
}

template <typename Item>
void BufferTopK<Item>::flush() {
  if (buffer.size() <= k)
    return;

  std::nth_element(buffer.begin(), buffer.begin() + k, buffer.end(), std::greater<>());

  buffer.resize(k);
}

template <typename Item>
void BufferTopK<Item>::reset() {
  buffer.clear();
}

template <typename Item>
void BufferTopK<Item>::merge(BufferTopK<Item> &other) {
  if (k != other.k)
    throw std::runtime_error("cannot merge topk buffers with different k");
  if (other.empty())
    return;

  flush();
  other.flush();

  buffer.insert(buffer.end(), std::make_move_iterator(other.buffer.begin()), std::make_move_iterator(other.buffer.end()));
  flush();

  // delete other buffer
  other.reset();
}

template <typename Item>
bool BufferTopK<Item>::empty() const {
  return buffer.empty();
}

template <typename Item>
void BufferTopK<Item>::sort() {
  flush();
  std::sort(buffer.begin(), buffer.end(), std::greater<>());
}

template <typename Item>
BatchTopK<Item>::BatchTopK(size_t batch_size, size_t k_) : batch_size(batch_size) {
  for (size_t i = 0; i < batch_size; i++) {
    top_ks.emplace_back(k_);
  }
}

template <typename Item>
inline void BatchTopK<Item>::push(const Item &value, size_t batch_idx) {
  top_ks[batch_idx].push(value);
}

template <typename Item>
void BatchTopK<Item>::sort() {
  for (size_t i = 0; i < batch_size; i++) {
    top_ks[i].sort();
  }
}

// template <typename Item>
// inline bool BatchTopK<Item>::can_push(const float score, size_t batch_idx) const {
//   return top_ks[batch_idx].can_push(score);
// }

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

} // namespace seqr