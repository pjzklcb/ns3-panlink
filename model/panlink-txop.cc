/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-txop.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkTxop");

NS_OBJECT_ENSURE_REGISTERED(PanlinkTxop);

TypeId
PanlinkTxop::GetTypeId()
{
    static TypeId tid = TypeId("ns3::PanlinkTxop")
                            .SetParent<QosTxop>()
                            .SetGroupName("Panlink")
                            .AddConstructor<PanlinkTxop>();
    return tid;
}

PanlinkTxop::PanlinkTxop()
{
    NS_LOG_FUNCTION(this);
}

PanlinkTxop::~PanlinkTxop()
{
    NS_LOG_FUNCTION(this);
}

std::unique_ptr<Txop::LinkEntity>
PanlinkTxop::CreateLinkEntity() const
{
    return std::make_unique<PanLinkEntity>();
}

PanlinkTxop::PanLinkEntity&
PanlinkTxop::GetLink(uint8_t linkId) const
{
    return static_cast<PanLinkEntity&>(Txop::GetLink(linkId));
}

} // namespace ns3
