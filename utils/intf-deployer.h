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

#ifndef INTF_DEPLOYER_H
#define INTF_DEPLOYER_H

#include "panlink-utils.h"

#include "ns3/app-installer.h"
#include "ns3/wifi-module.h"

#include <rapidjson/document.h>

namespace ns3
{

typedef std::map<uint32_t /* src node id */, std::vector<SrcApp>> SrcAppMap;

typedef std::tuple<uint32_t /* dst node id */, std::string /* type */, uint32_t /* port */> SinkApp;

/**
 * \ingroup panlink
 *
 * Deploy and setup interference environment based on the config file.
 */
class IntfDeployer
{
  public:
    IntfDeployer();
    ~IntfDeployer();

    /**
     * Read and check the config file.
     */
    void ReadIntfConfigFile(std::string filename);

  private:
    rapidjson::Document m_document;
};

} // namespace ns3

#endif /* INTF_DEPLOYER_H */
