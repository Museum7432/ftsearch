#ifndef FT_SEARCH_H
#define FT_SEARCH_H

#include "utils.h"
#include <cassert>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include <cstddef> // for std::size_t
static_assert(sizeof(float) == 4, "float is not 32 bits");

struct SeqInfo
{
    // starting and ending indecies of the seq vectors
    // within flat_embs
    size_t start_idx;
    size_t end_idx;

    std::string seq_name;

    // SeqInfo(const std::string &seq_name, size_t start_idx, size_t end_idx) : seq_name(seq_name), start_idx(start_idx), end_idx(end_idx) {}
};

class FTSearch
{
public:
    FTSearch(size_t vec_dim)
        : vec_dim(vec_dim) {}

    size_t num_seqs() const;
    // number of vectors
    size_t num_vecs() const;

    // add a new sequence of vectors,
    // seq_name is the identification of the sequence and
    // will be returned along with the indices of the result
    // in this sequence,
    // n is the number of vectors
    void add_seq(float *arr, size_t n, std::string seq_name);
    // removing a seq is not implimented yet

    void reset();

    // perform batch search
    // similar to faiss
    // return tuple for simplicity
    std::tuple<std::vector<float>, std::vector<size_t>> search(const float *Q, size_t nq, size_t topk) const;

    // perform sequence search (also called Temporal search)
    // min_item_dist is the minimum distance in indice between two selected vectors (consecutively)
    // discount_rate: penalize large gap between two selected vectors
    std::tuple<std::vector<float>, std::vector<size_t>> seq_search(
        const float *Q,
        const size_t nq,
        const size_t topk,
        const size_t min_item_dist = 1,
        const float discount_rate = 1) const;

    std::vector<float> get_vec(size_t vec_idx) const;

    // private:

    // compute the similarity between seq and queries
    // and save it into S, assumes that S have enough space.
    // used internally
    void _compute_similarity_per_seq(const size_t start_idx, const size_t end_idx, const float *Q, const size_t nq, float *S) const;
    // concatenate all videos' embeddings into a single array for
    // faster access time
    // (total number of frames x vec_dim)
    std::vector<float> flat_embs;

    // (number of sequences)
    std::vector<SeqInfo> seq_infos;

    // map the vector in flat_embs to SeqInfo
    // (total number of frames)
    std::vector<SeqInfo *> idx2seq_info;

    size_t vec_dim;
};

#endif