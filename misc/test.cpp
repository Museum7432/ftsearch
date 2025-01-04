#include "ftsearch.h"
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <sys/resource.h>
#include <vector>

#include <cstdlib> // for rand() and RAND_MAX
#include <ctime>
#include <omp.h>

void printMemoryUsage()
{
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "Memory Usage: " << usage.ru_maxrss / 1024 << " MB" << std::endl;
}

void print_tuple_vectors(const std::tuple<std::vector<float>, std::vector<size_t>> &t)
{
    const auto &float_vec = std::get<0>(t);
    const auto &size_t_vec = std::get<1>(t);

    std::cout << "Float vector: ";
    for (float f : float_vec)
    {
        std::cout << f << " ";
    }
    std::cout << std::endl;

    std::cout << "Size_t vector: ";
    for (size_t s : size_t_vec)
    {
        std::cout << s << " ";
    }
    std::cout << std::endl;
}

void fillVector(std::vector<float> &vec)
{
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::uniform_real_distribution<float> distribution(0.0f,
                                                                           1.0f);

    for (int i = 0; i < vec.size(); ++i)
    {
        vec[i] = distribution(generator);
    }
}

std::vector<float> create_rand_vector(size_t n)
{
    std::vector<float> t(n);

    fillVector(t);

    return t;
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    size_t vec_dim = 512;

    FTSearch ft = FTSearch(vec_dim);

    int test_size = 0;
#pragma omp parallel for
    for (int i = 0; i < 100; i++)
    {
        size_t seq_size = rand() % 1000 + 1000;

        test_size += seq_size;
        std::vector<float> t = create_rand_vector(vec_dim * seq_size);

#pragma omp critical
        {
            ft.add_seq(t.data(), seq_size, std::to_string(i));

            // std::cout << ft.seq_infos.back().start_idx << " " << ft.seq_infos.back().end_idx << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    // Output the duration in milliseconds
    std::cout << "init time: " << duration_ms.count() << " milliseconds"
              << std::endl;

    printMemoryUsage();

    std::cout << ft.get_size() << " " << test_size << std::endl;

    // search test
    auto queries = create_rand_vector(2 * vec_dim);

    std::tuple<std::vector<float>, std::vector<size_t>> re;

    // ##################################################################################
    start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < 10; i++)
    {
        // ft.search(queries.data(), 2, 5);
        re = ft.search(ft.flat_embs.data() + 10123 * vec_dim, 3, 1);

    }
    end = std::chrono::high_resolution_clock::now();
    duration_ms = end - start;

    print_tuple_vectors(re);
    std::cout << "search time: " << duration_ms.count() / 10 << " milliseconds"
              << std::endl;


    // ##################################################################################
    start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < 10; i++)
    {
        // ft.search(queries.data(), 2, 5);
        re = ft.seq_search(ft.flat_embs.data() + 10123 * vec_dim, 3, 2);

    }
    end = std::chrono::high_resolution_clock::now();
    duration_ms = end - start;

    print_tuple_vectors(re);
    std::cout << "search time: " << duration_ms.count() / 10 << " milliseconds"
              << std::endl;
    return 0;
}