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

#ifndef TRACE_FILE_BURST_GENERATOR_H
#define TRACE_FILE_BURST_GENERATOR_H

#include <ns3/burst-generator.h>

#include <queue>

namespace ns3
{

class Time;

/**
 * \ingroup applications
 *
 * \brief Reads a traffic trace file to generate bursts
 *
 * The generator reads a trace file and generates bursts accordingly.
 * A trace file should be formatted following the guidelines given
 * by the documentation of ns3::CsvReader.
 *
 * If the same trace file is used by multiple users in the same network,
 * the generated bursts can be decoupled by assigning different
 * StartTimes to different users, if the trace is long enough with respect
 * to the simulation duration.
 *
 */
class TraceFileBurstGenerator : public BurstGenerator
{
  public:
    TraceFileBurstGenerator();
    ~TraceFileBurstGenerator() override;

    // inherited from Object
    static TypeId GetTypeId();

    // inherited from BurstGenerator
    std::pair<uint32_t, Time> GenerateBurst() override;
    bool HasNextBurst() override;

    /**
     * \brief Returns the duration of the imported trace starting from the StartTime attribute
     * \return the duration of the trace
     */
    double GetTraceDuration();

  protected:
    void DoDispose() override;

  private:
    // Private attribute accessors
    void SetStartTime(double startTime);
    double GetStartTime() const;

    void SetTraceFile(std::string traceFile);
    std::string GetTraceFile() const;

    /**
     * Clears the burst queue
     */
    void ClearBurstQueue();

    /**
     * Import the trace file into the burst queue
     */
    void ImportTrace();

    std::string m_traceFile{""}; //!< The name of the trace file
    double m_startTime{
        0.0}; //!< The trace will only generate traced traffic after a start time offset
    double m_traceDuration{-1.0}; //!< The duration of the trace file considering the start time
    bool m_isFinalized{
        false}; //!< The generator is finalized only once ImportTrace ends with no errors
    std::queue<std::pair<uint32_t, Time>>
        m_burstQueue; //!< The burst information read from the trace
};

} // namespace ns3

#endif // TRACE_FILE_BURST_GENERATOR_H
