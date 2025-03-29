# About

This is the C++ implementation of the temporal search in the [Fusionista: Fusion of 3-D Information of Video in Retrieval System](https://doi.org/10.1007/978-981-96-2074-6_33) paper

# Instalation
require `openblas`, `openmp` and `cmake`.
```conda install openblas openmp cmake```

then install with:
```pip install .```
or
```pip install git+https://github.com/Museum7432/ftsearch.git```

# Usage
it is similar to Faiss, but designed more for searching on videos (a sequence of encoded vectors of keyframes with OpenClip for example).

```python
import ftsearch
import numpy as np

# 512 is vector dimension
ft = ftsearch.FTSearch(512)

for i in range(100):
    # a video with 1000 keyframes
    # this number can be arbitrary
    data = np.random.random((1000, 512)).astype("float32")
    
    # input the sequence along with the video name (might change later)
    ft.add_seq(data, f"vid_{i}")


query = data[[1,3,4]]

# batch query search
# similar to Faiss search
# it only supports dot product for now
re = ft.search(query, 2)

# the score
print(re[0])
# the indices of the scores
print(re[1])


# sequental search (or temporal search)
# it find the best subset of vectors within each sequence
# that best match the input vector
# min_item_dist is the minimum distance in indice between two selected vectors (consecutively)
# discount_rate: penalize large gap between two selected vectors
re = ft.seq_search(query, topk=1, min_item_dist=1, discount_rate=0.9)

print(re[0])
print(re[1])

# you can also get infos from the returned indices
print(ft.get_info(23123))
```

# Performance
Until issue [4121](https://github.com/facebookresearch/faiss/issues/4121#issue-2778286481) is fixed. This search function (both functions) is 2 to 3 times faster than Faiss' flat index searcher.
