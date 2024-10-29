/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 SIGNET Lab, Department of Information Engineering,
 * University of Padova
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
 */
#ifndef BURST_SINK_HELPER_H
#define BURST_SINK_HELPER_H

#include "ns3/application-container.h"
#include "ns3/application-helper.h"
#include "ns3/burst-sink.h"
#include "ns3/ipv4-address.h"
#include "ns3/node-container.h"
#include "ns3/object-factory.h"

namespace ns3
{

/**
 * \ingroup burstsink
 * \brief A helper to make it easier to instantiate an ns3::BurstSink on a set of nodes.
 */
class BurstSinkHelper : public ApplicationHelper
{
  public:
    /**
     * Create a BurstSinkHelper to make it easier to work with BurstyApplications
     *
     * \param protocol the name of the protocol to use to receive traffic
     *        This string identifies the socket factory type used to create
     *        sockets for the applications. A typical value would be
     *        ns3::UdpSocketFactory.
     * \param address the address of the sink,
     *
     */
    BurstSinkHelper(const std::string protocol, const Address address);
};

} // namespace ns3

#endif /* BURST_SINK_HELPER_H */
