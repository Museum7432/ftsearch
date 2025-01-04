#ifndef UTILS_H
#define UTILS_H
#include <cstddef>
#include <deque>
#include <queue>
#include <vector>

// invert L2
// dot product
// only dot product is implemented right now

// 1d dot product,
// a: (d,),
// b: (d,),
// result c: float
float dot_product(const float *a, const float *b, const size_t d);

void mdot_product(const float *A, const float *b, float *c, size_t m, size_t n);

// 2d x 2d pairwise dot product
// A: (m, d),
// B: (n, d),
// C: (m, n)
void mmdot_product(const float *A, const float *B, float *C, const size_t m,
                   const size_t d, const size_t n);

template <typename Obs>
class BoundedMinHeap
{
public:
    BoundedMinHeap(size_t maxSize) : maxSize(maxSize) {}

    void push(const Obs &value)
    {
        if (minHeap.size() < maxSize)
        {
            minHeap.push(value);
        }
        else if (value > minHeap.top())
        {
            minHeap.pop(); // Remove the smallest element
            minHeap.push(value);
        }
    }

    Obs toppop()
    {
        Obs item = minHeap.top();
        minHeap.pop();
        return item;
    }

    const Obs& top() const
    {
        return minHeap.top();
    }

    bool isEmpty() const
    {
        return minHeap.empty();
    }

    size_t size() const
    {
        return minHeap.size();
    }

private:
    std::priority_queue<Obs, std::deque<Obs>, std::greater<Obs>> minHeap;
    // std::priority_queue<Obs, std::vector<Obs>, std::greater<Obs>> minHeap;
    // std::priority_queue<Item> minHeap;
    size_t maxSize; // Maximum size of the set
};

#endif
