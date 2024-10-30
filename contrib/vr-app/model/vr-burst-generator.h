/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 SIGNET Lab, Department of Information Engineering,
 * University of Padova
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
 */

#ifndef VR_BURST_GENERATOR_H
#define VR_BURST_GENERATOR_H

#include <ns3/burst-generator.h>
#include <ns3/data-rate.h>
#include <ns3/my-random-variable-stream.h>

namespace ns3
{

/**
 * \ingroup applications
 *
 * \brief VR burst generator
 *
 * This burst generator creates synthetic traces, aiming to
 * emulate the traffic from a Virtual Reality (VR) application.
 *
 * The user can set a frame rate and the target data rate
 * of the application.
 * Further details on the model used can be found in the reference
 * paper (see README.md).
 *
 */
class VrBurstGenerator : public BurstGenerator
{
  public:
    /**
     * Different VR applications can be choosen.
     * Please check the documentation for further information.
     */
    enum VrAppName
    {
        VirusPopper = 0,
        Minecraft,
        GoogleEarthVrCities,
        GoogleEarthVrTour
    };

    VrBurstGenerator();
    ~VrBurstGenerator() override;

    // inherited from Object
    static TypeId GetTypeId();

    // inherited from BurstGenerator
    std::pair<uint32_t, Time> GenerateBurst() override;
    bool HasNextBurst() override;
    int64_t AssignStreams(int64_t stream) override;

    // Set/Get app parameters
    /**
     * Set the frame rate of the VR application
     * \param frameRate the frame rate in FPS
     */
    void SetFrameRate(double frameRate);
    /**
     * Get the frame rate of the VR application
     * \return the frame rate in FPS
     */
    double GetFrameRate() const;

    /**
     * Set the target data rate of the VR application
     * \param targetDataRate the target data rate
     */
    void SetTargetDataRate(DataRate targetDataRate);
    /**
     * Get the target data rate of the VR application
     * \return the target data rate
     */
    DataRate GetTargetDataRate() const;

    /**
     * Set the app name of the VR application
     * \param vrAppName the app name
     */
    void SetVrAppName(VrAppName vrAppName);
    /**
     * Get the app name of the VR application
     * \return the app name
     */
    VrAppName GetVrAppName() const;

  protected:
    void DoDispose() override;

  private:
    /**
     * Setup the random variables generating the burst parameters
     * according to the frame rate and the target data rate.
     *
     * For further information, please check the reference paper (see README.md).
     */
    void SetupModel();

    double m_frameRate{60};           //!< The frame rate of the VR application [FPS]
    DataRate m_targetDataRate{50};    //!< The target data rate of the VR application
    VrAppName m_appName{VirusPopper}; //!< The name of the VR application

    Ptr<LogisticRandomVariable> m_periodRv{0};    //!< RNG for period duration [s]
    Ptr<LogisticRandomVariable> m_frameSizeRv{0}; //!< RNG for frame size [B]
};

} // namespace ns3

#endif // VR_BURST_GENERATOR_H
