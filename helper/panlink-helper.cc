/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-helper.h"

#include "ns3/log.h"
#include "ns3/ns3-ai-msg-interface.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkHelper");

PanlinkHelper::PanlinkHelper()
    : PanlinkHelper("My")
{
    NS_LOG_FUNCTION(this);
}

PanlinkHelper::~PanlinkHelper()
{}

PanlinkHelper::PanlinkHelper(std::string timestamp)
    : m_timestamp(timestamp)
{
    NS_LOG_FUNCTION(this);

    // Create Ns3AiMsgInterface instance
    auto msgInterface = Ns3AiMsgInterface::Get();
    msgInterface->SetIsMemoryCreator(false);
    msgInterface->SetUseVector(true);
    msgInterface->SetHandleFinish(true);

    msgInterface->SetNames(m_timestamp + " Seg",
                           m_timestamp + " Cpp to Python Msg",
                           m_timestamp + " Python to Cpp Msg",
                           m_timestamp + " Lockable");
}

} // namespace ns3