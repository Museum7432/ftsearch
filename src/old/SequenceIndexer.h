/*
help storing sequences of different sizes in a continuous array.
*/

#include <cstddef>
#include <vector>
namespace seqr {

struct SeqIndexer {
  // CSR format
  std::vector<size_t> start_ids;

  SeqIndexer();

  // add a sequence of length n
  void add(size_t n = 0);

  // extend the last sequence by n
  void extend(size_t n);

  void reset();

  size_t num_seqs() const;
  size_t num_items() const;

  size_t num_items(size_t seq_id) const;

  size_t get_start(size_t seq_id) const;
  size_t get_end(size_t seq_id) const;
};



} // namespace seqr
