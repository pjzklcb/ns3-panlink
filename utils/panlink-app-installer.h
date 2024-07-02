/*
 * Copyright (c) 2024
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
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_APP_INSTALLER_H
#define PANLINK_APP_INSTALLER_H

#include "panlink-utils.h"

#include "ns3/applications-module.h"
#include "ns3/vr-app-module.h"

namespace ns3
{

typedef std::map<uint32_t /* src node id */, std::vector<SrcApp>> SrcAppMap;

typedef std::tuple<uint32_t /* dst node id */, std::string /* type */, uint32_t /* port */> SinkApp;

/**
 * \ingroup panlink
 *
 * The application installer for Panlink. Sink applications will be installed automatically based on
 * the source applications.
 */
class PanlinkAppInstaller
{
  public:
    /**
     * \param appMap the application map
     */
    PanlinkAppInstaller(AppMap appMap);
    ~PanlinkAppInstaller();

    // delete default constructor, copy constructor and assignment operator
    PanlinkAppInstaller() = delete;
    PanlinkAppInstaller(const PanlinkAppInstaller&) = delete;
    PanlinkAppInstaller& operator=(const PanlinkAppInstaller&) = delete;

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

#endif /* PANLINK_APP_INSTALLER_H */
