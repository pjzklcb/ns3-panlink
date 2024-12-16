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

template <typename T, size_t N>
void
bind_array(py::module& m, const std::string& name)
{
    py::class_<std::array<T, N>>(m, name.c_str())
        .def(py::init<>())
        .def("size", &std::array<T, N>::size)
        .def("__len__", [](const std::array<T, N>& arr) { return arr.size(); })
        .def("__getitem__",
             [](const std::array<T, N>& arr, uint32_t i) {
                 if (i >= arr.size())
                 {
                     std::cerr << "Invalid index " << i << " for std::array, whose size is "
                               << arr.size() << std::endl;
                     exit(1);
                 }
                 return arr.at(i);
             })
        .def("__iter__",
             [](const std::array<T, N>& arr) { return py::make_iterator(arr.begin(), arr.end()); });
}

PYBIND11_MODULE(cpp_py_interface, m)
{
    bind_array<double, 4>(m, "Double4Array");

    py::class_<SelfLinkObs>(m, "SelfLinkObs")
        .def(py::init<>())
        .def_readwrite("period", &SelfLinkObs::period)
        .def_readwrite("nTxDataPackets", &SelfLinkObs::nTxDataPackets)
        .def_readwrite("nAckedDataPackets", &SelfLinkObs::nAckedDataPackets)
        .def_readwrite("lastTxDataDelay", &SelfLinkObs::lastTxDataDelay)
        .def_readwrite("lastTxOkDelay", &SelfLinkObs::lastTxOkDelay)
        .def_readwrite("meanDataArrivalInterval", &SelfLinkObs::meanDataArrivalInterval)
        .def_readwrite("stdDataArrivalInterval", &SelfLinkObs::stdDataArrivalInterval)
        .def_readwrite("maxDataArrivalInterval", &SelfLinkObs::maxDataArrivalInterval)
        .def_readwrite("minDataArrivalInterval", &SelfLinkObs::minDataArrivalInterval)
        .def_readwrite("meanTxDataInterval", &SelfLinkObs::meanTxDataInterval)
        .def_readwrite("stdTxDataInterval", &SelfLinkObs::stdTxDataInterval)
        .def_readwrite("maxTxDataInterval", &SelfLinkObs::maxTxDataInterval)
        .def_readwrite("minTxDataInterval", &SelfLinkObs::minTxDataInterval)
        .def_readwrite("channelUtil", &SelfLinkObs::channelUtil);

    py::class_<NeighborStaObs>(m, "NeighborStaObs")
        .def(py::init<>())
        .def_readwrite("role", &NeighborStaObs::role)
        .def_readwrite("period", &NeighborStaObs::period)
        .def_readwrite("nTxDataPackets", &NeighborStaObs::nTxDataPackets)
        .def_readwrite("nAckedDataPackets", &NeighborStaObs::nAckedDataPackets)
        .def_readwrite("lastTxDataDelay", &NeighborStaObs::lastTxDataDelay)
        .def_readwrite("lastTxOkDelay", &NeighborStaObs::lastTxOkDelay)
        .def_readwrite("meanTxDataInterval", &NeighborStaObs::meanTxDataInterval)
        .def_readwrite("stdTxDataInterval", &NeighborStaObs::stdTxDataInterval)
        .def_readwrite("maxTxDataInterval", &NeighborStaObs::maxTxDataInterval)
        .def_readwrite("minTxDataInterval", &NeighborStaObs::minTxDataInterval)
        .def_readwrite("channelUtil", &NeighborStaObs::channelUtil)
        .def_readwrite("meanRssi", &NeighborStaObs::meanRssi);

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
        .def_readwrite("selfLinkObs", &Obs::selfLinkObs)
        .def_readwrite("nInQueuePackets", &Obs::nInQueuePackets)
        .def_readwrite("nExpiredPackets", &Obs::nExpiredPackets)
        .def_readwrite("nDroppedPackets", &Obs::nDroppedPackets)
        .def_readwrite("nSendingPackets", &Obs::nSendingPackets)
        .def_readwrite("nAckedPackets", &Obs::nAckedPackets)
        .def_readwrite("nLowDelayPackets", &Obs::nLowDelayPackets)
        .def_readwrite("nHighDelayPackets", &Obs::nHighDelayPackets)
        .def_readwrite("ac", &Obs::ac)
        .def_readwrite("cw", &Obs::cw)
        .def_readwrite("nDeferredAccess", &Obs::nDeferredAccess)
        .def_readwrite("waitedSlots", &Obs::waitedSlots)
        .def_readwrite("neighborStaObs", &Obs::neighborStaObs)
        .def_readwrite("nStasWithPackets", &Obs::nStasWithPackets)
        .def_readwrite("minNeighborBackoff", &Obs::minNeighborBackoff);

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
        .def("PySetRestart", &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::PySetRestart)
        .def("GetCpp2PyVector",
             &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::GetCpp2PyVector,
             py::return_value_policy::reference)
        .def("GetPy2CppVector",
             &ns3::Ns3AiMsgInterfaceImpl<Obs, Act>::GetPy2CppVector,
             py::return_value_policy::reference);
}
