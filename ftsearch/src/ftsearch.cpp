#include "ftsearch.h"
#include "cblas.h"
#include "omp.h"
#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "utils.h"

void safePrefetch(const float *array, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        // Ensure we are within bounds before prefetching
        if (i + 4 < size)
        {
            __builtin_prefetch(&array[i + 4], 0, 1); // Prefetch for read
        }
    }
}

size_t FTSearch::num_seqs() const { return seq_infos.size(); }
size_t FTSearch::num_vecs() const { return flat_embs.size() / vec_dim; }

void FTSearch::add_seq(float *arr, size_t n, std::string seq_id)
{
    if (n == 0)
        return;

    // the index of the first vector after being added into the flat_embs
    size_t start_idx = num_vecs();

    flat_embs.insert(flat_embs.end(), arr, arr + n * vec_dim);

    // index of the last vector
    size_t end_idx = num_vecs() - 1;

    // seq_infos.push_back(SeqInfo(seq_id, start_idx, end_idx));

    seq_infos.push_back({start_idx, end_idx, seq_id});

    // pointer to the stored seq_info
    // SeqInfo *seq_info_ptr = &seq_infos.back();
    // SeqInfo *seq_info_ptr = &seq_infos[seq_infos.size() - 1];

    size_t seq_info_idx = seq_infos.size() - 1;

    // std::vector<SeqInfo *> idx_mapping(n, seq_info_ptr);

    // idx2seq_info.insert(idx2seq_info.end(), idx_mapping.begin(), idx_mapping.end());

    idx2seq_info.resize(n + idx2seq_info.size(), seq_info_idx);

    assert(idx2seq_info.size() == num_vecs());
}

void FTSearch::reset()
{
    flat_embs.clear();
    seq_infos.clear();
    idx2seq_info.clear();
}

struct Item
{
    size_t id;
    float sim;

    // Define a comparison operator for max heap
    bool operator<(const Item &other) const
    {
        return sim < other.sim;
    }
    bool operator>(const Item &other) const
    {
        return sim > other.sim;
    }
};

void FTSearch::_compute_similarity_per_seq(const size_t start_idx, const size_t end_idx, const float *Q, const size_t nq, float *S) const
{
    // we only use nested omp loop when a seq is too long
#pragma omp parallel for if (end_idx - start_idx > 5000)
    for (size_t emb_idx = start_idx; emb_idx <= end_idx; emb_idx++)
    {
        const float *embs_ptr = flat_embs.data() + emb_idx * vec_dim;

        if (nq > 500)
        {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, nq, vec_dim, 1, Q, vec_dim, embs_ptr, 1, 0, S + emb_idx, num_vecs());
        }
        else
        {
            for (size_t q_idx = 0; q_idx < nq; q_idx++)
            {
                const float *query_ptr = Q + q_idx * vec_dim;

                S[emb_idx + q_idx * num_vecs()] = cblas_sdot(vec_dim, embs_ptr, 1, query_ptr, 1);
            }
        }
    }
}

std::tuple<std::vector<float>, std::vector<size_t>> FTSearch::search(const float *Q, size_t nq, size_t topk) const
{
    // loop through all seq_infos
    // and compute results for each video
    // unless number of queries is larger than 50
    // this is fast enough

    const size_t n_seqs = num_seqs(),
                 n_vecs = num_vecs();

    // parwise similarities matrix
    // TODO: allow the user to cache this
    // (nq, num_vecs)
    std::vector<float> S(nq * n_vecs);

    // separate heap for each query
    std::vector<BoundedMinHeap<Item>> result_heap;
    for (int i = 0; i < nq; i++)
        result_heap.emplace_back(topk);

#pragma omp parallel
    {
        const size_t num_threads = omp_get_num_threads(),
                     thread_id = omp_get_thread_num();

        // chunking
        const size_t base_batch_size = n_seqs / num_threads;
        const size_t remainder = n_seqs % num_threads;

        const size_t start_seq_idx = thread_id * base_batch_size + std::min(thread_id, remainder);
        const size_t end_seq_idx = start_seq_idx + base_batch_size + (thread_id < remainder ? 1 : 0) - 1;

        if (start_seq_idx < n_seqs)
        {
            // stored the topk result from the computed S
            std::vector<BoundedMinHeap<Item>> local_result_heap;
            // separate heap for each query
            for (int i = 0; i < nq; i++)
                local_result_heap.emplace_back(topk);

            for (size_t seq_idx = start_seq_idx; seq_idx <= end_seq_idx; seq_idx++)
            {

                SeqInfo seq_info = seq_infos[seq_idx];
                // TODO: perform seq filtering here (base on the string recorded in seq_info)
                // calculate the similarities into S
                _compute_similarity_per_seq(seq_info.start_idx, seq_info.end_idx, Q, nq, S.data());

                // separate the computation of similarity and heap to optimize the memory access pattern
                for (size_t q_idx = 0; q_idx < nq; q_idx++)
                {
                    for (size_t emb_idx = seq_info.start_idx; emb_idx <= seq_info.end_idx; emb_idx++)
                    {
                        local_result_heap[q_idx].push({emb_idx, S[emb_idx + q_idx * n_vecs]});
                    }
                }
            }

// combine topK in a critical region
// this should only add a few ms in latency
#pragma omp critical
            {
                for (size_t q_idx = 0; q_idx < nq; q_idx++)
                {
                    while (!local_result_heap[q_idx].isEmpty())
                    {
                        auto item = local_result_heap[q_idx].toppop();

                        result_heap[q_idx].push(item);
                    }
                }
            }
        }
    }

    std::vector<float> sims(nq * topk, 0);
    std::vector<size_t> ids(nq * topk, -1);

    for (size_t q_idx = 0; q_idx < nq; q_idx++)
    {
        const size_t n_res = result_heap[q_idx].size();

        for (size_t i = 0; i < n_res; i++)
        {
            auto item = result_heap[q_idx].toppop();

            sims[q_idx * topk + n_res - 1 - i] = item.sim;
            ids[q_idx * topk + n_res - 1 - i] = item.id;
        }
    }

    return std::make_tuple(sims, ids);
}

struct SeqItem
{
    // result is a sequence of id
    std::vector<size_t> ids;
    float sim;

    // Define a comparison operator for max heap
    bool operator<(const SeqItem &other) const
    {
        return sim < other.sim;
    }
    bool operator>(const SeqItem &other) const
    {
        return sim > other.sim;
    }
};

std::tuple<std::vector<float>, std::vector<size_t>> FTSearch::seq_search(const float *Q, const size_t nq, const size_t topk, const size_t min_item_dist, const float discount_rate) const
{
    const size_t n_seqs = num_seqs(),
                 n_vecs = num_vecs();

    // parwise similarities matrix
    // TODO: allow the user to cache this
    // (nq, num_vecs)
    std::vector<float> S(nq * n_vecs);

    // we only have a single result now
    BoundedMinHeap<SeqItem> result_heap(topk);

#pragma omp parallel
    {
        const size_t num_threads = omp_get_num_threads(),
                     thread_id = omp_get_thread_num();

        // chunking
        const size_t base_batch_size = n_seqs / num_threads;
        const size_t remainder = n_seqs % num_threads;

        const size_t start_seq_idx = thread_id * base_batch_size + std::min(thread_id, remainder);
        const size_t end_seq_idx = start_seq_idx + base_batch_size + (thread_id < remainder ? 1 : 0) - 1;

        if (start_seq_idx < n_seqs)
        {
            // stored the topk result from the computed S
            BoundedMinHeap<SeqItem> local_result_heap(topk);

            for (size_t seq_idx = start_seq_idx; seq_idx <= end_seq_idx; seq_idx++)
            {

                const SeqInfo seq_info = seq_infos[seq_idx];
                // TODO: perform seq filtering here (base on the string recorded in seq_info)
                // calculate the similarities into S
                _compute_similarity_per_seq(seq_info.start_idx, seq_info.end_idx, Q, nq, S.data());

                // temporal_matching
                const size_t seq_len = seq_info.end_idx - seq_info.start_idx + 1;

                std::vector<size_t> traces((nq - 1) * seq_len, 0);

                // float at indices i is the maximum sum of similarities of all sequences that stop at indice i
                // create a new copy
                std::vector<float> score(S.data() + seq_info.start_idx, S.data() + seq_info.end_idx + 1);
                std::vector<float> score_temp(S.data() + seq_info.start_idx, S.data() + seq_info.end_idx + 1);

                for (size_t q_idx = 1; q_idx < nq; q_idx++)
                {
                    // cumulative max on score
                    float max_score = score[0];
                    size_t max_score_idx = 0;

                    const float *S_seq_q = S.data() + seq_info.start_idx + q_idx * n_vecs;

                    for (size_t i = 0; i + min_item_dist < seq_len; i++)
                    {
                        max_score *= discount_rate;

                        if (score[i] > max_score)
                        {
                            max_score = score[i];
                            max_score_idx = i;
                        }

                        // item at indice (i + min_item_dist) can only select
                        // indicies <= i
                        score_temp[i + min_item_dist] = max_score + S_seq_q[i + min_item_dist];

                        // max_score_idx is relative to the start of the seq
                        traces[i + min_item_dist + (q_idx - 1) * seq_len] = max_score_idx + seq_info.start_idx;
                    }
                    // important
                    std::swap(score, score_temp);
                }

                // then save the result
                // only trace the seq_ids when we can push into the queue
                for (size_t i = min_item_dist * (nq - 1); i < seq_len; i++)
                {
                    // if (score[i] > 499)std::cout << "1" << std::endl;
                    // if score is lower than the lowest item in heap
                    if (!local_result_heap.isEmpty() and score[i] < local_result_heap.top().sim)
                    {
                        continue;
                    }

                    std::vector<size_t> _seq_ids(nq, i + seq_info.start_idx);

                    for (size_t q_idx = nq - 1; q_idx-- > 0;)
                    {
                        // q_idx actually start at nq - 2
                        _seq_ids[q_idx] = traces[q_idx * seq_len + _seq_ids[q_idx + 1] - seq_info.start_idx];
                    }

                    local_result_heap.push({_seq_ids, score[i]});
                }
            }

// combine topK in a critical region
// this should only add a few ms in latency
#pragma omp critical
            {
                while (!local_result_heap.isEmpty())
                {
                    auto item = local_result_heap.toppop();

                    result_heap.push(item);
                }
            }
        }
    }

    std::vector<float> sims(topk, 0);
    std::vector<size_t> ids(topk * nq, -1);

    const size_t n_res = result_heap.size();

    for (size_t i = 0; i < n_res; i++)
    {
        auto item = result_heap.toppop();

        sims[n_res - 1 - i] = item.sim;

        for (size_t q_idx = 0; q_idx < nq; q_idx++)
        {
            ids[(n_res - 1 - i) * nq + q_idx] = item.ids[q_idx];
        }
    }

    return std::make_tuple(sims, ids);
}

std::vector<float> FTSearch::get_vec(size_t vec_idx) const
{
    return std::vector<float>(flat_embs.data() + vec_idx * vec_dim, flat_embs.data() + (vec_idx + 1) * vec_dim);
}

const SeqInfo FTSearch::get_info(const size_t vec_idx) const
{
    return seq_infos[idx2seq_info[vec_idx]];
}

const SeqInfo FTSearch::get_seq_info(const size_t seq_idx) const
{
    return seq_infos[seq_idx];
}

// SimpleIndex
void SimpleIndex::add(float *arr, size_t n)
{
    if (n == 0)
        return;

    flat_embs.insert(flat_embs.end(), arr, arr + n * vec_dim);
}
size_t SimpleIndex::num_vecs() const { return flat_embs.size() / vec_dim; }

void SimpleIndex::reset()
{
    flat_embs.clear();
}

std::tuple<std::vector<float>, std::vector<size_t>> SimpleIndex::search(const float *Q, size_t nq, size_t topk) const
{
    const size_t n_vecs = num_vecs();

    // parwise similarities matrix
    // TODO: allow the user to cache this
    // (nq, num_vecs)
    // std::vector<float> S(nq * n_vecs);

    // separate heap for each query
    std::vector<BoundedMinHeap<Item>> result_heap;
    for (int i = 0; i < nq; i++)
        result_heap.emplace_back(topk);

    auto original_openblas_num_threads = openblas_get_num_threads();
    openblas_set_num_threads(1);

#pragma omp parallel
    {
        const size_t num_threads = omp_get_num_threads(),
                     thread_id = omp_get_thread_num();

        // chunking
        const size_t base_batch_size = n_vecs / num_threads;
        const size_t remainder = n_vecs % num_threads;

        const size_t start_idx = thread_id * base_batch_size + std::min(thread_id, remainder);
        const size_t end_idx = start_idx + base_batch_size + (thread_id < remainder ? 1 : 0) - 1;

        if (start_idx < n_vecs)
        {
            // stored the topk result from the computed S
            std::vector<BoundedMinHeap<Item>> local_result_heap;
            // separate heap for each query
            for (int i = 0; i < nq; i++)
            {
                local_result_heap.emplace_back(topk);
            }

            const size_t num_vecs_local = end_idx - start_idx + 1;
            std::vector<float> S_local(nq * num_vecs_local);

            // cblas_sgemm is really expensive
            if (nq < 15)
            {
                // #pragma omp parallel for if (end_idx - start_idx > 5000)
                for (size_t vec_idx = start_idx; vec_idx <= end_idx; vec_idx++)
                {

                    const float *embs_ptr = flat_embs.data() + vec_idx * vec_dim;

                    float *re_ptr = S_local.data() + vec_idx - start_idx;

                    // cblas_sgemv(CblasRowMajor, CblasNoTrans, nq, vec_dim, 1, Q, vec_dim, embs_ptr, 1, 0, re_ptr, num_vecs_local);

                    for (size_t q_idx = 0; q_idx < nq; q_idx++)
                    {
                        const float *query_ptr = Q + q_idx * vec_dim;

                        re_ptr[q_idx * num_vecs_local] = cblas_sdot(vec_dim, embs_ptr, 1, query_ptr, 1);
                    }
                }
            }
            else
            {
                // embs: (num_vecs_local, vec_dim)
                // Q: (nq, vec_dim)
                // res: (nq, num_vecs_local)
                // Q * embs.T

                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    nq, num_vecs_local, vec_dim,
                    1,
                    Q, vec_dim,
                    flat_embs.data() + start_idx * vec_dim, vec_dim,
                    0,
                    S_local.data(), num_vecs_local);
            }

            // separate the computation of similarity and heap to optimize the memory access pattern
            for (size_t q_idx = 0; q_idx < nq; q_idx++)
            {
                for (size_t vec_idx = start_idx; vec_idx <= end_idx; vec_idx++)
                {
                    local_result_heap[q_idx].push({vec_idx, S_local[vec_idx - start_idx + q_idx * num_vecs_local]});
                }
            }

// combine topK in a critical region
// this should only add a few ms in latency
#pragma omp critical
            {
                for (size_t q_idx = 0; q_idx < nq; q_idx++)
                {
                    while (!local_result_heap[q_idx].isEmpty())
                    {
                        auto item = local_result_heap[q_idx].toppop();

                        result_heap[q_idx].push(item);
                    }
                }
            }
        }
    }

    openblas_set_num_threads(original_openblas_num_threads);

    std::vector<float> sims(nq * topk, 0);
    std::vector<size_t> ids(nq * topk, -1);

    for (size_t q_idx = 0; q_idx < nq; q_idx++)
    {
        const size_t n_res = result_heap[q_idx].size();

        for (size_t i = 0; i < n_res; i++)
        {
            auto item = result_heap[q_idx].toppop();

            sims[q_idx * topk + n_res - 1 - i] = item.sim;
            ids[q_idx * topk + n_res - 1 - i] = item.id;
        }
    }

    return std::make_tuple(sims, ids);
}
