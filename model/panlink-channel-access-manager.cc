/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-channel-access-manager.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkChannelAccessManager");

NS_OBJECT_ENSURE_REGISTERED(PanlinkChannelAccessManager);

TypeId
PanlinkChannelAccessManager::GetTypeId()
{
    static TypeId tid = TypeId("ns3::PanlinkChannelAccessManager")
                            .SetParent<ChannelAccessManager>()
                            .SetGroupName("Panlink")
                            .AddConstructor<PanlinkChannelAccessManager>();
    return tid;
}

PanlinkChannelAccessManager::PanlinkChannelAccessManager()
{
    NS_LOG_FUNCTION(this);
}

PanlinkChannelAccessManager::~PanlinkChannelAccessManager()
{
    NS_LOG_FUNCTION(this);
}

} // namespace ns3
