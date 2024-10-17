/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_HELPER_H
#define PANLINK_HELPER_H

#include <string>

namespace ns3
{

/**
 * \ingroup panlink
 */

class PanlinkHelper
{
  public:
    PanlinkHelper();
    virtual ~PanlinkHelper();

    PanlinkHelper(std::string timestamp);

  private:
    std::string m_timestamp;
};

} // namespace ns3

#endif /* PANLINK_HELPER_H */