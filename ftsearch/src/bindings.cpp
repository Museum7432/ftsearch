#include "ftsearch.h" // Make sure to include your class's header
#include <cstddef>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion of vectors

namespace py = pybind11;

auto search_with_numpy(const FTSearch &self, py::array_t<float> Q, size_t topk)
{

    py::buffer_info buf = Q.request();

    if (buf.ndim != 2)
    {
        throw std::runtime_error("Input array must be 2D");
    }

    auto ptr = static_cast<float *>(buf.ptr);
    size_t nq = buf.shape[0];
    size_t dim = buf.shape[1];

    if (dim != self.vec_dim)
    {
        throw std::runtime_error("Input array's dimension does not match vec_dim");
    }

    auto result = self.search(ptr, nq, topk);

    const auto &distances = std::get<0>(result);
    const auto &indices = std::get<1>(result);

    py::array_t<float> distances_array({nq, topk}, distances.data());
    py::array_t<size_t> indices_array({nq, topk}, indices.data());

    return py::make_tuple(distances_array, indices_array);
}

auto seq_search_with_numpy(const FTSearch &self, py::array_t<float> Q, size_t topk, size_t min_item_dist = 1, float discount_rate = 1)
{

    py::buffer_info buf = Q.request();

    if (buf.ndim != 2)
    {
        throw std::runtime_error("Input array must be 2D");
    }

    auto ptr = static_cast<float *>(buf.ptr);
    size_t nq = buf.shape[0];
    size_t dim = buf.shape[1];

    if (dim != self.vec_dim)
    {
        throw std::runtime_error("Input array's dimension does not match vec_dim");
    }

    auto result = self.seq_search(ptr, nq, topk, min_item_dist, discount_rate);

    const auto &distances = std::get<0>(result);
    const auto &indices = std::get<1>(result);

    py::array_t<float> distances_array(topk, distances.data());
    py::array_t<size_t> indices_array({topk, nq}, indices.data());

    return py::make_tuple(distances_array, indices_array);
}

void add_sequence(FTSearch &self, py::array_t<float> array, const std::string &seq_name)
{
    py::buffer_info buf_info = array.request();

    if (buf_info.ndim != 2)
    {
        throw std::runtime_error("Input array must be 2D.");
    }

    float *ptr = static_cast<float *>(buf_info.ptr);
    size_t n = buf_info.shape[0];
    size_t vec_dim = buf_info.shape[1];

    if (vec_dim != self.vec_dim)
    {
        throw std::runtime_error("Input vector dimension does not match initialized dimension.");
    }

    self.add_seq(ptr, n, seq_name);
}

auto get_vec(const FTSearch &self, size_t vec_idx)
{

    auto vec = self.get_vec(vec_idx);

    py::array_t<float> vec_np(vec.size(), vec.data());

    return vec_np;
}

auto get_info(const FTSearch &self, size_t vec_idx)
{
    auto info = self.get_info(vec_idx);
    py::dict info_dict;

    info_dict["start_idx"] = info.start_idx;
    info_dict["end_idx"] = info.end_idx;
    info_dict["seq_name"] = info.seq_name;
    return info_dict;
}

PYBIND11_MODULE(ftsearch_module, m)
{
    py::class_<FTSearch>(m, "FTSearch")
        .def(py::init<size_t>(), py::arg("vec_dim"))
        .def("num_seqs", &FTSearch::num_seqs)
        .def("num_vecs", &FTSearch::num_vecs)
        .def("get_info", &get_info, py::arg("vec_idx"))
        .def("get_vec", &get_vec, py::arg("vec_idx"))
        .def("add_seq", &add_sequence, py::arg("arr"), py::arg("seq_name"))
        .def("reset", &FTSearch::reset)
        .def("search", &search_with_numpy, py::arg("Q"), py::arg("topk"))
        .def("seq_search", &seq_search_with_numpy, py::arg("Q"), py::arg("topk"), py::arg("min_item_dist") = 1, py::arg("discount_rate") = 1)
        .def_readwrite("vec_dim", &FTSearch::vec_dim);
}