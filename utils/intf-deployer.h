/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
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
