/*
 * Copyright (c) 2023 Huazhong University of Science and Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author:  Muyuan Shen <muyuan_shen@hust.edu.cn>
 */

#include "obs-act-list.h"

#include "ns3/ai-module.h"

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Cpp2PyMsgVector);
PYBIND11_MAKE_OPAQUE(ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Py2CppMsgVector);

PYBIND11_MODULE(cpp_py_interface, m)
{
    py::class_<std::array<double, 4>>(m, "QosArray")
        .def(py::init<>())
        .def("size", &std::array<double, 4>::size)
        .def("__len__", [](const std::array<double, 4>& arr) { return arr.size(); })
        .def("__getitem__", [](const std::array<double, 4>& arr, uint32_t i) {
            if (i >= arr.size())
            {
                std::cerr << "Invalid index " << i << " for std::array, whose size is "
                          << arr.size() << std::endl;
                exit(1);
            }
            return arr.at(i);
        });

    py::class_<Obs>(m, "PyEnvStruct")
        .def(py::init<>())
        .def_readwrite("nodeId", &Obs::nodeId)
        .def_readwrite("agentType", &Obs::agentType)
        .def_readwrite("counter", &Obs::counter)
        .def_readwrite("simTime", &Obs::simTime)
        .def_readwrite("thpt", &Obs::thpt)
        .def_readwrite("thptReq", &Obs::thptReq)
        .def_readwrite("delay", &Obs::delay)
        .def_readwrite("delayReq", &Obs::delayReq)
        .def_readwrite("selfD2lt", &Obs::selfD2lt)
        .def_readwrite("selfChannelUtil", &Obs::selfChannelUtil)
        .def_readwrite("ackRatio", &Obs::ackRatio)
        .def_readwrite("ac", &Obs::ac)
        .def_readwrite("cw", &Obs::cw)
        .def_readwrite("nDeferredAccess", &Obs::nDeferredAccess)
        .def_readwrite("waitedSlots", &Obs::waitedSlots)
        .def_readwrite("nBssStas", &Obs::nBssStas)
        .def_readwrite("nObssStas", &Obs::nObssStas)
        .def_readwrite("bssRssi", &Obs::bssRssi)
        .def_readwrite("obssRssi", &Obs::obssRssi)
        .def_readwrite("bssD2lt", &Obs::bssD2lt)
        .def_readwrite("otherD2lt", &Obs::otherD2lt)
        .def_readwrite("bssChannelUtil", &Obs::bssChannelUtil)
        .def_readwrite("otherChannelUtil", &Obs::otherChannelUtil);

    py::class_<Act>(m, "PyActStruct")
        .def(py::init<>())
        .def_readwrite("ccaSensitivity", &Act::ccaSensitivity)
        .def_readwrite("backoff", &Act::backoff);

    py::class_<ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Cpp2PyMsgVector>(m, "PyEnvVector")
        .def("resize",
             static_cast<void (ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Cpp2PyMsgVector::*)(
                 ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Cpp2PyMsgVector::size_type)>(
                 &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Cpp2PyMsgVector::resize))
        .def("__len__", &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Cpp2PyMsgVector::size)
        .def(
            "__getitem__",
            [](ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Cpp2PyMsgVector& vec, uint32_t i) -> Obs& {
                if (i >= vec.size())
                {
                    std::cerr << "Invalid index " << i << " for vector, whose size is "
                              << vec.size() << std::endl;
                    exit(1);
                }
                return vec.at(i);
            },
            py::return_value_policy::reference);

    py::class_<ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Py2CppMsgVector>(m, "PyActVector")
        .def("resize",
             static_cast<void (ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Py2CppMsgVector::*)(
                 ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Py2CppMsgVector::size_type)>(
                 &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Py2CppMsgVector::resize))
        .def("__len__", &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Py2CppMsgVector::size)
        .def(
            "__getitem__",
            [](ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::Py2CppMsgVector& vec, uint32_t i) -> Act& {
                if (i >= vec.size())
                {
                    std::cerr << "Invalid index " << i << " for vector, whose size is "
                              << vec.size() << std::endl;
                    exit(1);
                }
                return vec.at(i);
            },
            py::return_value_policy::reference);

    py::class_<ns3::Ns3AiMsgInterfaceImpl<Obs, Act>>(m, "Ns3AiMsgInterfaceImpl")
        .def(py::init<bool,
                      bool,
                      bool,
                      uint32_t,
                      const char*,
                      const char*,
                      const char*,
                      const char*>())
        .def("PyRecvBegin", &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::PyRecvBegin)
        .def("PyRecvEnd", &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::PyRecvEnd)
        .def("PySendBegin", &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::PySendBegin)
        .def("PySendEnd", &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::PySendEnd)
        .def("PyGetFinished", &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::PyGetFinished)
        .def("GetCpp2PyVector",
             &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::GetCpp2PyVector,
             py::return_value_policy::reference)
        .def("GetPy2CppVector",
             &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::GetPy2CppVector,
             py::return_value_policy::reference);
}
