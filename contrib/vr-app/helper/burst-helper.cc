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
 *
 */
#include "burst-helper.h"

#include "ns3/burst-application.h"
#include "ns3/data-rate.h"
#include "ns3/inet-socket-address.h"
#include "ns3/names.h"
#include "ns3/packet-socket-address.h"
#include "ns3/pointer.h"
#include "ns3/random-variable-stream.h"
#include "ns3/simple-burst-generator.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"
#include "ns3/vr-burst-generator.h"

namespace ns3
{

BurstHelper::BurstHelper(const std::string protocol, const Address address)
    : ApplicationHelper("ns3::BurstApplication")
{
    m_factory.Set("Protocol", StringValue(protocol));
    m_factory.Set("Remote", AddressValue(address));
}

void
BurstHelper::SetBurstGenerator(std::string type,
                               std::string n1,
                               const AttributeValue& v1,
                               std::string n2,
                               const AttributeValue& v2,
                               std::string n3,
                               const AttributeValue& v3,
                               std::string n4,
                               const AttributeValue& v4)
{
    m_burstGeneratorFactory.SetTypeId(type);
    m_burstGeneratorFactory.Set(n1, v1);
    m_burstGeneratorFactory.Set(n2, v2);
    m_burstGeneratorFactory.Set(n3, v3);
    m_burstGeneratorFactory.Set(n4, v4);
}

void
BurstHelper::SetBurstGeneratorAttribute(std::string name, const AttributeValue& value)
{
    m_burstGeneratorFactory.Set(name, value);
}

Ptr<Application>
BurstHelper::DoInstall(Ptr<Node> node)
{
    auto app = ApplicationHelper::DoInstall(node);
    Ptr<BurstGenerator> generator = m_burstGeneratorFactory.Create<BurstGenerator>();

    app->SetAttribute("BurstGenerator", PointerValue(generator));

    return app;
}

} // namespace ns3
