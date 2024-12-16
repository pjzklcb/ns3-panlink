/*
 * Copyright (c) 2023 Huazhong University of Science and Technology
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
 * Authors:  Muyuan Shen <muyuan_shen@hust.edu.cn>
 */

#ifndef NS3_AI_SEMAPHORE_H
#define NS3_AI_SEMAPHORE_H

#include <cstdint>
#include <thread>
#include <atomic>

/**
 * \brief Structure providing semaphore operations
 */
struct Ns3AiSemaphore
{
    explicit Ns3AiSemaphore() = default;

    static inline uint32_t atomic_read32(std::atomic<uint32_t>& mem)
    {
        return mem.load(std::memory_order_acquire);
    }

    static inline bool atomic_cas32(std::atomic<uint32_t>& mem, uint32_t with, uint32_t cmp)
    {
        return mem.compare_exchange_strong(cmp, with, std::memory_order_acq_rel, std::memory_order_acquire);
    }

    static inline uint32_t atomic_add32(std::atomic<uint32_t>& mem, uint32_t val)
    {
        return mem.fetch_add(val, std::memory_order_acq_rel);
    }

    static inline bool atomic_add_unless32(std::atomic<uint32_t>& mem, uint32_t value, uint32_t unless_this)
    {
        uint32_t old = atomic_read32(mem);
        while (old != unless_this && !atomic_cas32(mem, old + value, old))
        {
            old = atomic_read32(mem);
        }
        return old != unless_this;
    }

    // try semaphore operation (non-blocking)
    static inline bool sem_try_wait(std::atomic<uint32_t>& mem)
    {
        uint32_t old = atomic_read32(mem);
        if (old > 0)
        {
            return atomic_cas32(mem, old - 1, old);
        }
        return false;
    }

    // semaphore operation (blocking)
    static inline void sem_wait(std::atomic<uint32_t>& mem)
    {
        while (!sem_try_wait(mem))
        {
            std::this_thread::yield(); // avoid busy-waiting
        }
    }

    // increase semaphore
    static inline uint32_t sem_post(std::atomic<uint32_t>& mem)
    {
        uint32_t old = atomic_read32(mem);
        if (old == UINT32_MAX)
        {
            throw std::overflow_error("Semaphore value overflow");
        }
        return atomic_add32(mem, 1);
    }
};

#endif // NS3_AI_SEMAPHORE_H
