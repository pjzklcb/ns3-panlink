/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-mac-helper.h"

#include "ns3/ap-emlsr-manager.h"
#include "ns3/eht-configuration.h"
#include "ns3/emlsr-manager.h"
#include "ns3/multi-user-scheduler.h"
#include "ns3/panlink-channel-access-manager.h"
#include "ns3/panlink-frame-exchange-manager.h"
#include "ns3/panlink-sta-wifi-mac.h"
#include "ns3/panlink-txop.h"
#include "ns3/pointer.h"
#include "ns3/sensing-mac.h"
#include "ns3/wifi-assoc-manager.h"
#include "ns3/wifi-net-device.h"

namespace ns3
{

PanlinkMacHelper::PanlinkMacHelper()
{
    if (m_mac.GetTypeId() != PanlinkStaWifiMac::GetTypeId())
    // && m_mac.GetTypeId() != PanlinkApWifiMac::GetTypeId())
    {
        m_channelAccessManager.SetTypeId("ns3::PanlinkChannelAccessManager");
        m_frameExchangeManager.SetTypeId("ns3::PanlinkFrameExchangeManager");
        for (const auto& [aci, ac] : wifiAcList)
        {
            m_edca[aci].SetTypeId("ns3::PanlinkTxop");
        }
    }

    m_sensingMac.SetTypeId("ns3::SensingMac");
}

PanlinkMacHelper::~PanlinkMacHelper()
{
}

Ptr<WifiMac>
PanlinkMacHelper::Create(Ptr<WifiNetDevice> device, WifiStandard standard) const
{
    if (m_mac.GetTypeId() != PanlinkStaWifiMac::GetTypeId())
    // && m_mac.GetTypeId() != PanlinkApWifiMac::GetTypeId())
    {
        return WifiMacHelper::Create(device, standard);
    }

    NS_ABORT_MSG_IF(standard == WIFI_STANDARD_UNSPECIFIED, "No standard specified!");

    // Panlink is based on 802.11ax, so we need to force the correct standard settings
    ObjectFactory macObjectFactory = m_mac;
    macObjectFactory.Set("QosSupported", BooleanValue(true));

    // create PanlinkTxop objects
    for (auto [aci, edca] : m_edca)
    {
        edca.Set("AcIndex", EnumValue<AcIndex>(aci));
        std::stringstream ss;
        ss << aci << "_Txop";
        auto s = ss.str().substr(3); // discard "AC "
        macObjectFactory.Set(s, PointerValue(edca.Create<QosTxop>()));
    }

    // WaveNetDevice (through ns-3.38) stores PHY entities in a different member than WifiNetDevice,
    // hence GetNPhys() would return 0
    auto nLinks = std::max<uint8_t>(device->GetNPhys(), 1);

    // create Channel Access Managers
    std::vector<Ptr<ChannelAccessManager>> caManagers;

    for (uint8_t linkId = 0; linkId < nLinks; ++linkId)
    {
        caManagers.emplace_back(m_channelAccessManager.Create<ChannelAccessManager>());
    }

    Ptr<WifiMac> mac = macObjectFactory.Create<WifiMac>();
    mac->SetDevice(device);
    mac->SetAddress(Mac48Address::Allocate());
    device->SetMac(mac);
    mac->SetChannelAccessManagers(caManagers);

    // create sensing MAC
    auto sensingMac = m_sensingMac.Create<SensingMac>();
    sensingMac->SetWifiMac(mac);

    // create Frame Exchange Managers, each with an attached protection manager and ack manager
    std::vector<Ptr<FrameExchangeManager>> feManagers;
    for (uint8_t linkId = 0; linkId < nLinks; ++linkId)
    {
        auto fem = m_frameExchangeManager.Create<FrameExchangeManager>();
        feManagers.emplace_back(fem);

        auto protectionManager = m_protectionManager.Create<WifiProtectionManager>();
        protectionManager->SetWifiMac(mac);
        protectionManager->SetLinkId(linkId);
        fem->SetProtectionManager(protectionManager);

        auto ackManager = m_ackManager.Create<WifiAckManager>();
        ackManager->SetWifiMac(mac);
        ackManager->SetLinkId(linkId);
        fem->SetAckManager(ackManager);

        // bind sensing MAC
        StaticCast<PanlinkFrameExchangeManager>(fem)->SetSensingMac(sensingMac);

        // 11be MLDs require a MAC address to be assigned to each STA
        fem->SetAddress(device->GetNPhys() > 1 ? Mac48Address::Allocate() : mac->GetAddress());
    }

    mac->SetFrameExchangeManagers(feManagers);

    Ptr<WifiMacQueueScheduler> queueScheduler = m_queueScheduler.Create<WifiMacQueueScheduler>();
    mac->SetMacQueueScheduler(queueScheduler);

    // create and install the Multi User Scheduler if this is an HE AP
    Ptr<ApWifiMac> apMac;
    if (standard >= WIFI_STANDARD_80211ax && m_muScheduler.IsTypeIdSet() &&
        (apMac = DynamicCast<ApWifiMac>(mac)))
    {
        Ptr<MultiUserScheduler> muScheduler = m_muScheduler.Create<MultiUserScheduler>();
        apMac->AggregateObject(muScheduler);
    }

    // create and install the Association Manager if this is a STA
    auto staMac = DynamicCast<PanlinkStaWifiMac>(mac);
    if (staMac)
    {
        Ptr<WifiAssocManager> assocManager = m_assocManager.Create<WifiAssocManager>();
        staMac->SetAssocManager(assocManager);

        // install sensing MAC
        staMac->SetSensingMac(sensingMac);
    }

    // create and install the EMLSR Manager if this is an EHT non-AP MLD with EMLSR activated
    if (BooleanValue emlsrActivated;
        standard >= WIFI_STANDARD_80211be && staMac && staMac->GetNLinks() > 1 &&
        device->GetEhtConfiguration()->GetAttributeFailSafe("EmlsrActivated", emlsrActivated) &&
        emlsrActivated.Get())
    {
        auto emlsrManager = m_emlsrManager.Create<EmlsrManager>();
        staMac->SetEmlsrManager(emlsrManager);
    }

    // create and install the AP EMLSR Manager if this is an EHT AP MLD with EMLSR activated
    if (BooleanValue emlsrActivated;
        standard >= WIFI_STANDARD_80211be && apMac && apMac->GetNLinks() > 1 &&
        device->GetEhtConfiguration()->GetAttributeFailSafe("EmlsrActivated", emlsrActivated) &&
        emlsrActivated.Get())
    {
        auto apEmlsrManager = m_apEmlsrManager.Create<ApEmlsrManager>();
        apMac->SetApEmlsrManager(apEmlsrManager);
    }

    return mac;
}

} // namespace ns3
