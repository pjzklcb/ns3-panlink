/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef APP_INSTALLER
#define APP_INSTALLER

#include "panlink-utils.h"

#include "ns3/applications-module.h"
#include "ns3/vr-app-module.h"

#include <rapidjson/document.h>

namespace ns3
{

/**
 * The information of an application specified in the configuration file.
 */
struct AppInfo
{
    std::string m_payloadInterval = "ns3::ConstantRandomVariable[Constant=0.1]";
    std::string m_payloadSize = "ns3::ConstantRandomVariable[Constant=1000]";
    uint32_t m_fragmentSize{1500};
};

typedef std::map<std::string /* application name */, AppInfo> AppDict;
typedef std::pair<uint32_t /* dst node id */, uint16_t /* port */> SinkApp;

/**
 * @ingroup panlink
 *
 * The application installer. Sink applications will be installed automatically based on
 * the source applications.
 */
class AppInstaller
{
  public:
    /**
     * Create a AppInstaller in an empty state: app dict must be set before calling
     * ns3::AppInstaller::InstallApplications
     */
    AppInstaller();
    ~AppInstaller();

    /**
     * Read application profile from JSON file.
     *
     * @param filename the name of the JSON file
     */
    void ReadAppProfileFromJson(std::string filename);

    /**
     * Install sink applications to the given nodes.
     *
     * @param nodes the given nodes
     * @param srcApps the source applications
     */
    ApplicationContainer InstallApplications(const NodeContainer& nodes,
                                             const Ipv4InterfaceContainer& interfaces,
                                             const SrcAppMap& srcApps);

    /**
     * Assigns a unique (monotonically increasing) stream number to all implemented application
     * helpers. Return the number of streams (possibly zero) that have been assigned. The Install()
     * method should have previously been called by the user.
     *
     * @param stream first stream index to use
     * @param nodes NodeContainer of the set of nodes for which the application should be modified
     * to use a fixed stream
     * @return the number of stream indices assigned by this installer.
     */
    int64_t AssignStreams(NodeContainer nodes, int64_t stream);

  private:
    /**
     * Install sink applications to the given nodes.
     *
     * @param nodes the given nodes
     */
    ApplicationContainer InstallSinkApplications(NodeContainer nodes);

    /**
     * Install a source application to the given node.
     *
     * @param node the given node
     * @param dstAddr the destination address
     * @param srcApp the source application
     */
    ApplicationContainer DoInstallApplication(Ptr<Node> node,
                                              InetSocketAddress dstAddr,
                                              SrcApp srcApp);

    AppDict m_appDict;                           //!< application dictionary
    std::set<SinkApp> m_sinkAppList;             //!< list of sink applications
    std::vector<ApplicationHelper> m_appHelpers; //!< list of application helpers
    Ptr<UniformRandomVariable> m_jitter;         //!< jitter
};

} // namespace ns3

#endif /* APP_INSTALLER */
