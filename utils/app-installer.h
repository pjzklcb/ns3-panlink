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

namespace ns3
{

/**
 * \ingroup panlink
 *
 * The application installer. Sink applications will be installed automatically based on
 * the source applications.
 */
class AppInstaller
{
  public:
    /**
     * \param appMap the application map
     */
    AppInstaller(AppMap appMap);
    ~AppInstaller();

    // delete default constructor, copy constructor and assignment operator
    AppInstaller() = delete;
    AppInstaller(const AppInstaller&) = delete;
    AppInstaller& operator=(const AppInstaller&) = delete;

    /**
     * Install sink applications to the given nodes.
     *
     * \param nodes the given nodes
     * \param srcApps the source applications
     */
    ApplicationContainer Install(NodeContainer nodes,
                                 Ipv4InterfaceContainer interfaces,
                                 SrcAppMap srcApps);

    /**
     * Install a source application to the given node.
     *
     * \param node the given node
     * \param dstAddr the destination address
     * \param appName the application name
     */
    ApplicationContainer Install(Ptr<Node> node, InetSocketAddress dstAddr, std::string appName);

    /**
     * Assigns a unique (monotonically increasing) stream number to all implemented application
     * helpers. Return the number of streams (possibly zero) that have been assigned. The Install()
     * method should have previously been called by the user.
     *
     * \param stream first stream index to use
     * \param nodes NodeContainer of the set of nodes for which the application should be modified
     * to use a fixed stream
     * \return the number of stream indices assigned by this installer.
     */
    int64_t AssignStreams(NodeContainer nodes, int64_t stream);

  private:
    /**
     * Install sink applications to the given nodes.
     *
     * \param nodes the given nodes
     */
    ApplicationContainer InstallSinkApp(NodeContainer nodes);

    AppMap m_appMap;                             //!< application map
    std::vector<SinkApp> m_sinkAppList;          //!< list of sink applications
    std::vector<ApplicationHelper> m_appHelpers; //!< list of application helpers
};

} // namespace ns3

#endif /* APP_INSTALLER */
