#include "lob_engine.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(lob_engine, m) {
  py::enum_<lob::Side>(m, "Side")
      .value("Bid", lob::Side::Bid)
      .value("Ask", lob::Side::Ask)
      .export_values();

  py::class_<lob::LimitOrderBook>(m, "LimitOrderBook")
      .def(py::init<>())
      .def("add_order", &lob::LimitOrderBook::addOrder,
           py::arg("id"), py::arg("price"), py::arg("qty"), py::arg("side"))
      .def("cancel_order", &lob::LimitOrderBook::cancelOrder)
      .def("match", &lob::LimitOrderBook::match)
      .def("top_levels",
           [](const lob::LimitOrderBook &book, std::size_t depth) {
             auto [bids, asks] = book.topLevels(depth);
             py::array_t<double> array({2, static_cast<py::ssize_t>(depth), 2});
             auto view = array.mutable_unchecked<3>();

             for (std::size_t i = 0; i < depth; ++i) {
               if (i < bids.size()) {
                 view(0, i, 0) = bids[i].price;
                 view(0, i, 1) = static_cast<double>(bids[i].qty);
               } else {
                 view(0, i, 0) = 0.0;
                 view(0, i, 1) = 0.0;
               }

               if (i < asks.size()) {
                 view(1, i, 0) = asks[i].price;
                 view(1, i, 1) = static_cast<double>(asks[i].qty);
               } else {
                 view(1, i, 0) = 0.0;
                 view(1, i, 1) = 0.0;
               }
             }

             return array;
           },
           py::arg("depth") = 5)
      .def("last_latency_ns", [](const lob::LimitOrderBook &book) {
        return book.lastLatency().count();
      });
}
