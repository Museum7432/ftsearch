#include "ftsearch.h" // Make sure to include your class's header
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

auto seq_search_with_numpy(const FTSearch &self, py::array_t<float> Q, size_t topk)
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

    auto result = self.seq_search(ptr, nq, topk);

    const auto &distances = std::get<0>(result);
    const auto &indices = std::get<1>(result);

    py::array_t<float> distances_array(topk, distances.data());
    py::array_t<size_t> indices_array({topk, nq}, indices.data());

    return py::make_tuple(distances_array, indices_array);
}

void add_sequence(FTSearch &ftsearch, py::array_t<float> array, const std::string &seq_name)
{
    py::buffer_info buf_info = array.request();

    if (buf_info.ndim != 2)
    {
        throw std::runtime_error("Input array must be 2D.");
    }

    float *ptr = static_cast<float *>(buf_info.ptr);
    size_t n = buf_info.shape[0];
    size_t vec_dim = buf_info.shape[1];

    if (vec_dim != ftsearch.vec_dim)
    {
        throw std::runtime_error("Input vector dimension does not match initialized dimension.");
    }

    ftsearch.add_seq(ptr, n, seq_name);
}

PYBIND11_MODULE(ftsearch_module, m)
{
    py::class_<FTSearch>(m, "FTSearch")
        .def(py::init<size_t>(), py::arg("vec_dim"))
        .def("get_size", &FTSearch::get_size)
        .def("add_seq", &add_sequence, py::arg("arr"), py::arg("seq_name"))
        .def("reset", &FTSearch::reset)
        .def("search", &search_with_numpy, py::arg("Q"), py::arg("topk"))
        .def("seq_search", &seq_search_with_numpy, py::arg("Q"), py::arg("topk"))
        .def_readwrite("vec_dim", &FTSearch::vec_dim);
}