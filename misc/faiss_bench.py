import numpy as np
import faiss
import time

# Parameters
d = 512  # Dimension of the vectors
n = 1500000  # Number of vectors in the dataset
k = 5  # Number of nearest neighbors to retrieve
n_queries = 10  # Number of queries

# Generate random dataset
np.random.seed(42)
data = np.random.random((n, d)).astype('float32')
queries = np.random.random((n_queries, d)).astype('float32')

# Create a FAISS index for flat dot product
index = faiss.IndexFlatIP(d)  # IP stands for Inner Product (dot product)

# Add the dataset to the index
index.add(data)

# Benchmarking the search
start_time = time.time()
for i in range(10):
    distances, indices = index.search(queries, k)  # Search for k nearest neighbors
end_time = time.time()

# Calculate elapsed time
elapsed_time = (end_time - start_time) / 10
print(f"Time taken for searching {n_queries} queries: {elapsed_time:.4f} seconds")
print(f"Average time per query: {elapsed_time / n_queries:.6f} seconds")

# Optionally, print some results
print("Sample results (indices of nearest neighbors):")
print(indices[:5])  # Print indices of the first 5 queries
print("Sample results (distances):")
print(distances[:5])  # Print distances of the first 5 queries
