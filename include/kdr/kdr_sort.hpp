#pragma once

// Copyright (c) 2026 Kadir Karahan <kadir.karahan2009@gmail.com>
// Enhanced & Hardened Edition with Cross-Platform Support, Strict Aliasing Safety,
// ThreadPool, 16/8-bit Types, Counting Sort Fast-Path & Complete Architecture Fallback.
// License: PolyForm Noncommercial License 1.0.0 (see LICENSE).
//          Commercial use requires separate authorization from the copyright owner.

/**
 * KDR short — Ultra High-Performance Adaptive SIMD MSD Radix Sort (C++17)
 *
 * Supported Architectures:
 *  - AVX-512 (__AVX512F__): 4x-unrolled bit-MSD + 8-ZMM in-register bitonic networks (0-byte heap)
 *  - AVX2 (__AVX2__): 3-phase 256-way block-buffered in-place MSD + L2 branchless LSD (512 KB scratch)
 *  - Scalar / ARM / Portable Fallback: Pure ISO C++17 block-buffered MSD (compiles everywhere)
 *
 * Supported Types:
 *  - All Integral Types: uint32_t, int32_t, uint64_t, int64_t, long, unsigned long,
 *                        uint16_t, int16_t, uint8_t, int8_t, char, signed char, unsigned char
 *  - Floating Point: float, double (IEEE-754 bijective mapping, deterministic NaN handling)
 *  - Strings: std::string (8-byte big-endian word cache, LCP skip, O(N) move cycles, 64-bit index)
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

// ============================================================================
// FIX 1 & 10: PORTABLE COMPILER & ARCHITECTURE INTRINSICS
// ============================================================================
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#define KDR_X86_SIMD_AVAILABLE 1
#endif
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#include <cstdlib>
#define KDR_ALWAYS_INLINE __forceinline
#define KDR_LAMBDA_INLINE
#define KDR_NOINLINE __declspec(noinline)

inline int kdr_clz64(uint64_t x) noexcept {
    unsigned long idx;
    if (_BitScanReverse64(&idx, x)) return 63 - static_cast<int>(idx);
    return 64;
}
inline uint64_t kdr_bswap64(uint64_t x) noexcept {
    return _byteswap_uint64(x);
}
inline int kdr_popcount32(unsigned int x) noexcept {
    return __popcnt(x);
}
#elif defined(__GNUC__) || defined(__clang__)
#define KDR_ALWAYS_INLINE __attribute__((always_inline)) inline
#define KDR_LAMBDA_INLINE __attribute__((always_inline))
#define KDR_NOINLINE __attribute__((noinline))

inline int kdr_clz64(uint64_t x) noexcept {
    return x ? __builtin_clzll(x) : 64;
}
inline uint64_t kdr_bswap64(uint64_t x) noexcept {
    return __builtin_bswap64(x);
}
inline int kdr_popcount32(unsigned int x) noexcept {
    return __builtin_popcount(x);
}
#else
#define KDR_ALWAYS_INLINE inline
#define KDR_LAMBDA_INLINE
#define KDR_NOINLINE

inline int kdr_clz64(uint64_t x) noexcept {
    if (x == 0) return 64;
    int n = 0;
    if ((x & 0xFFFFFFFF00000000ULL) == 0) { n += 32; x <<= 32; }
    if ((x & 0xFFFF000000000000ULL) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF00000000000000ULL) == 0) { n += 8;  x <<= 8;  }
    if ((x & 0xF000000000000000ULL) == 0) { n += 4;  x <<= 4;  }
    if ((x & 0xC000000000000000ULL) == 0) { n += 2;  x <<= 2;  }
    if ((x & 0x8000000000000000ULL) == 0) { n += 1; }
    return n;
}
inline uint64_t kdr_bswap64(uint64_t x) noexcept {
    return ((x & 0xFF00000000000000ULL) >> 56) |
           ((x & 0x00FF000000000000ULL) >> 40) |
           ((x & 0x0000FF0000000000ULL) >> 24) |
           ((x & 0x000000FF00000000ULL) >> 8)  |
           ((x & 0x00000000FF000000ULL) << 8)  |
           ((x & 0x0000000000FF0000ULL) << 24) |
           ((x & 0x000000000000FF00ULL) << 40) |
           ((x & 0x00000000000000FFULL) << 56);
}
inline int kdr_popcount32(unsigned int x) noexcept {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
#endif

namespace kdr {
namespace detail {

// OPTIMIZATION 1: 16-element blocks (64 bytes for uint32_t = exactly 1 cache line)
// Keeps 256-bucket merged_buf at 16 KB, perfectly fitting in 32 KB L1D Cache!
constexpr size_t BLOCK_SIZE = 16;
constexpr size_t SCRATCH_CAP = 131072;
constexpr size_t INSERTION_CUTOFF = 48;
constexpr size_t COUNTING_SORT_MAX_RANGE = 65536;

// OPTIMIZATION 2: Thread-Local Static Scratch Buffer (Zero Dynamic Heap Allocations)
template <typename U>
inline U* get_tls_scratch() {
    thread_local std::vector<U> tls_buf;
    if (tls_buf.size() < SCRATCH_CAP) {
        tls_buf.resize(SCRATCH_CAP);
    }
    return tls_buf.data();
}

// ============================================================================
// FIX 7: WARM THREAD POOL (ZERO OS THREAD CREATION OVERHEAD, DEADLOCK-FREE, ZERO-SPIN)
// ============================================================================
class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    template <typename F>
    void parallel_for(int num_tasks, F&& task_fn) {
        if (num_tasks <= 1) {
            task_fn(0);
            return;
        }
        std::atomic<int> next_task{0};
        int worker_count = std::min<int>(num_tasks, static_cast<int>(workers_.size() + 1));
        std::atomic<int> completed{0};
        std::mutex completion_mutex;
        std::condition_variable completion_cv;

        auto worker_work = [&]() {
            while (true) {
                int t = next_task.fetch_add(1, std::memory_order_relaxed);
                if (t >= num_tasks) break;
                task_fn(t);
            }
            if (completed.fetch_add(1, std::memory_order_acq_rel) + 1 >= worker_count) {
                std::unique_lock<std::mutex> lock(completion_mutex);
                completion_cv.notify_all();
            }
        };

        // Submit tasks to queue for workers
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            for (int i = 0; i < worker_count - 1; ++i) {
                tasks_.emplace(worker_work);
            }
        }
        cv_.notify_all();

        // Main thread participates
        worker_work();

        // Wait for all participating workers to complete without busy spinning.
        // Work-stealing helper loop prevents nested recursive deadlocks:
        while (completed.load(std::memory_order_acquire) < worker_count) {
            std::function<void()> local_task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (!tasks_.empty()) {
                    local_task = std::move(tasks_.front());
                    tasks_.pop();
                }
            }
            if (local_task) {
                local_task();
            } else {
                std::unique_lock<std::mutex> lock(completion_mutex);
                if (completed.load(std::memory_order_acquire) < worker_count) {
                    completion_cv.wait_for(lock, std::chrono::microseconds(100), [&]() {
                        return completed.load(std::memory_order_acquire) >= worker_count;
                    });
                }
            }
        }
    }

private:
    ThreadPool() : stop_(false) {
        int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        for (int i = 1; i < threads; ++i) {
            workers_.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& th : workers_) {
            if (th.joinable()) th.join();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_;
};

// ============================================================================
// FIX 3: COMPREHENSIVE BIJECTIVE ORDER-PRESERVING RADIX KEY TRAITS
// ============================================================================
template <typename T, typename Enable = void>
struct RadixTraits;

template <> struct RadixTraits<uint32_t> {
    using U = uint32_t;
    static inline U to_key(uint32_t x) noexcept { return x; }
    static inline uint32_t from_key(U k) noexcept { return k; }
};

template <> struct RadixTraits<uint64_t> {
    using U = uint64_t;
    static inline U to_key(uint64_t x) noexcept { return x; }
    static inline uint64_t from_key(U k) noexcept { return k; }
};

template <> struct RadixTraits<int32_t> {
    using U = uint32_t;
    static inline U to_key(int32_t x) noexcept {
        return static_cast<uint32_t>(x) ^ 0x80000000u;
    }
    static inline int32_t from_key(U k) noexcept {
        return static_cast<int32_t>(k ^ 0x80000000u);
    }
};

template <> struct RadixTraits<int64_t> {
    using U = uint64_t;
    static inline U to_key(int64_t x) noexcept {
        return static_cast<uint64_t>(x) ^ 0x8000000000000000ULL;
    }
    static inline int64_t from_key(U k) noexcept {
        return static_cast<int64_t>(k ^ 0x8000000000000000ULL);
    }
};

// Portable support for every other integral type (long, unsigned long, long long,
// unsigned long long, wchar_t, char16_t, ...). Whichever of these is not already one of
// the exact-width types specialised in this file is mapped, by size and signedness,
// onto the matching fixed-width traits. This avoids redefinition errors on LP64
// (Linux: long == int64_t) as well as on LLP64 (Windows) and macOS.
template <typename T>
inline constexpr bool is_exact_radix_type_v =
    std::is_same_v<T, int8_t>  || std::is_same_v<T, uint8_t>  ||
    std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
    std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
    std::is_same_v<T, char>;

template <size_t N, bool Signed> struct fixed_int;
template <> struct fixed_int<1, true>  { using type = int8_t; };
template <> struct fixed_int<1, false> { using type = uint8_t; };
template <> struct fixed_int<2, true>  { using type = int16_t; };
template <> struct fixed_int<2, false> { using type = uint16_t; };
template <> struct fixed_int<4, true>  { using type = int32_t; };
template <> struct fixed_int<4, false> { using type = uint32_t; };
template <> struct fixed_int<8, true>  { using type = int64_t; };
template <> struct fixed_int<8, false> { using type = uint64_t; };

template <typename T>
struct RadixTraits<T, std::enable_if_t<std::is_integral_v<T> &&
                                       !std::is_same_v<T, bool> &&
                                       !is_exact_radix_type_v<T>>> {
    using Fixed = typename fixed_int<sizeof(T), std::is_signed_v<T>>::type;
    using U = typename RadixTraits<Fixed>::U;
    static inline U to_key(T x) noexcept { return RadixTraits<Fixed>::to_key(static_cast<Fixed>(x)); }
    static inline T from_key(U k) noexcept { return static_cast<T>(RadixTraits<Fixed>::from_key(k)); }
};

// 16-bit Integer Support
template <> struct RadixTraits<int16_t> {
    using U = uint16_t;
    static inline U to_key(int16_t x) noexcept {
        return static_cast<uint16_t>(x) ^ 0x8000u;
    }
    static inline int16_t from_key(U k) noexcept {
        return static_cast<int16_t>(k ^ 0x8000u);
    }
};

template <> struct RadixTraits<uint16_t> {
    using U = uint16_t;
    static inline U to_key(uint16_t x) noexcept { return x; }
    static inline uint16_t from_key(U k) noexcept { return k; }
};

// 8-bit Integer & Character Support
template <> struct RadixTraits<int8_t> {
    using U = uint8_t;
    static inline U to_key(int8_t x) noexcept {
        return static_cast<uint8_t>(x) ^ 0x80u;
    }
    static inline int8_t from_key(U k) noexcept {
        return static_cast<int8_t>(k ^ 0x80u);
    }
};

template <> struct RadixTraits<uint8_t> {
    using U = uint8_t;
    static inline U to_key(uint8_t x) noexcept { return x; }
    static inline uint8_t from_key(U k) noexcept { return k; }
};

template <> struct RadixTraits<char> {
    using U = uint8_t;
    static inline U to_key(char x) noexcept {
        if constexpr (std::is_signed_v<char>) {
            return static_cast<uint8_t>(static_cast<int8_t>(x) ^ 0x80);
        } else {
            return static_cast<uint8_t>(x);
        }
    }
    static inline char from_key(U k) noexcept {
        if constexpr (std::is_signed_v<char>) {
            return static_cast<char>(static_cast<int8_t>(k ^ 0x80));
        } else {
            return static_cast<char>(k);
        }
    }
};

// FIX 5: IEEE-754 Floating Point with strict bijective bit mapping
template <> struct RadixTraits<float> {
    using U = uint32_t;
    static inline U to_key(float x) noexcept {
        uint32_t u;
        std::memcpy(&u, &x, sizeof(u));
        uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(u >> 31)) | 0x80000000u;
        return u ^ mask;
    }
    static inline float from_key(U k) noexcept {
        uint32_t mask = ((k >> 31) - 1u) | 0x80000000u;
        uint32_t u = k ^ mask;
        float x;
        std::memcpy(&x, &u, sizeof(x));
        return x;
    }
};

template <> struct RadixTraits<double> {
    using U = uint64_t;
    static inline U to_key(double x) noexcept {
        uint64_t u;
        std::memcpy(&u, &x, sizeof(u));
        uint64_t mask = static_cast<uint64_t>(-static_cast<int64_t>(u >> 63)) | 0x8000000000000000ULL;
        return u ^ mask;
    }
    static inline double from_key(U k) noexcept {
        uint64_t mask = ((k >> 63) - 1ULL) | 0x8000000000000000ULL;
        uint64_t u = k ^ mask;
        double x;
        std::memcpy(&x, &u, sizeof(x));
        return x;
    }
};

template <typename U>
inline void insertion_sort(U* first, U* last) noexcept {
    for (U* i = first + 1; i < last; ++i) {
        U val = *i;
        U* j = i;
        while (j > first && *(j - 1) > val) {
            *j = *(j - 1);
            --j;
        }
        *j = val;
    }
}

// FIX 5: Robust NaN-safe monotonic prefix check (0=unsorted, 1=ascending, 2=descending)
template <typename T>
inline int quick_monotonic_check(const T* data, size_t n) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        // Fast abort on NaN: NaNs cannot be trivially monotonic
        for (size_t i = 0; i < std::min<size_t>(n, 64); ++i) {
            if (std::isnan(data[i])) return 0;
        }
    }

    size_t probe = std::min<size_t>(n, 64);
    bool asc = true, desc = true;
    for (size_t i = 1; i < probe; ++i) {
        asc &= (data[i - 1] <= data[i]);
        desc &= (data[i - 1] >= data[i]);
    }
    if (!asc && !desc) return 0;

    for (size_t i = probe; i < n; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(data[i])) return 0;
        }
        asc &= (data[i - 1] <= data[i]);
        desc &= (data[i - 1] >= data[i]);
        if (!asc && !desc) return 0;
    }
    if (asc) return 1;
    if (desc) return 2;
    return 0;
}

// FIX 9: Ultra-Fast Counting Sort for Small Value Ranges (K << N)
template <typename U>
inline bool try_counting_sort(U* data, size_t n, U min_val, U max_val) {
    uint64_t range = static_cast<uint64_t>(max_val) - static_cast<uint64_t>(min_val) + 1;
    if (range > COUNTING_SORT_MAX_RANGE || n < 32768) return false;

    std::vector<uint32_t> counts(static_cast<size_t>(range), 0);
    for (size_t i = 0; i < n; ++i) {
        counts[static_cast<size_t>(data[i] - min_val)]++;
    }

    size_t out_idx = 0;
    for (size_t r = 0; r < range; ++r) {
        uint32_t c = counts[r];
        if (c > 0) {
            U val = static_cast<U>(min_val + r);
            std::fill_n(data + out_idx, c, val);
            out_idx += c;
        }
    }
    return true;
}

#ifdef __AVX512F__
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

// ============================================================================
// BACKEND 1: AVX-512 IN-PLACE BIT-MSD RADIX + 8-ZMM BITONIC NETWORKS (O(1) RAM)
// ============================================================================
inline __m512i sort16_u32(__m512i v) noexcept {
    auto cs = [](__m512i a, __m512i idx, __mmask16 hm) KDR_LAMBDA_INLINE {
        __m512i p = _mm512_permutexvar_epi32(idx, a);
        return _mm512_mask_blend_epi32(hm, _mm512_min_epu32(a, p), _mm512_max_epu32(a, p));
    };
    v = cs(v, _mm512_set_epi32(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1), 0x6666);
    v = cs(v, _mm512_set_epi32(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2), 0x3C3C);
    v = cs(v, _mm512_set_epi32(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1), 0x5A5A);
    v = cs(v, _mm512_set_epi32(11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4), 0x0FF0);
    v = cs(v, _mm512_set_epi32(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2), 0x33CC);
    v = cs(v, _mm512_set_epi32(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1), 0x55AA);
    v = cs(v, _mm512_set_epi32(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8), 0xFF00);
    v = cs(v, _mm512_set_epi32(11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4), 0xF0F0);
    v = cs(v, _mm512_set_epi32(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2), 0xCCCC);
    v = cs(v, _mm512_set_epi32(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1), 0xAAAA);
    return v;
}

inline __m512i bm16_u32(__m512i v) noexcept {
    auto cs = [](__m512i a, __m512i idx, __mmask16 hm) KDR_LAMBDA_INLINE {
        __m512i p = _mm512_permutexvar_epi32(idx, a);
        return _mm512_mask_blend_epi32(hm, _mm512_min_epu32(a, p), _mm512_max_epu32(a, p));
    };
    v = cs(v, _mm512_set_epi32(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8), 0xFF00);
    v = cs(v, _mm512_set_epi32(11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4), 0xF0F0);
    v = cs(v, _mm512_set_epi32(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2), 0xCCCC);
    v = cs(v, _mm512_set_epi32(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1), 0xAAAA);
    return v;
}

inline void bm32_u32(__m512i& a, __m512i& b) noexcept {
    __m512i lo = _mm512_min_epu32(a, b), hi = _mm512_max_epu32(a, b);
    a = bm16_u32(lo); b = bm16_u32(hi);
}

inline void bm64_u32(__m512i& v0, __m512i& v1, __m512i& v2, __m512i& v3) noexcept {
    __m512i a0 = _mm512_min_epu32(v0, v2), b0 = _mm512_max_epu32(v0, v2);
    __m512i a1 = _mm512_min_epu32(v1, v3), b1 = _mm512_max_epu32(v1, v3);
    bm32_u32(a0, a1); bm32_u32(b0, b1);
    v0 = a0; v1 = a1; v2 = b0; v3 = b1;
}

inline void merge32_u32(__m512i& a, __m512i& b) noexcept {
    __m512i rev = _mm512_set_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    __m512i br = _mm512_permutexvar_epi32(rev, b);
    __m512i lo = _mm512_min_epu32(a, br), hi = _mm512_max_epu32(a, br);
    a = bm16_u32(lo); b = bm16_u32(hi);
}

inline void merge64_u32(__m512i& v0, __m512i& v1, __m512i& v2, __m512i& v3) noexcept {
    __m512i rev = _mm512_set_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    __m512i v3r = _mm512_permutexvar_epi32(rev, v3), v2r = _mm512_permutexvar_epi32(rev, v2);
    __m512i a0 = _mm512_min_epu32(v0, v3r), b1 = _mm512_max_epu32(v0, v3r);
    __m512i a1 = _mm512_min_epu32(v1, v2r), b0 = _mm512_max_epu32(v1, v2r);
    bm32_u32(a0, a1); bm32_u32(b0, b1);
    v0 = a0; v1 = a1; v2 = b0; v3 = b1;
}

inline void sort_small_u32_zmm(uint32_t* p, size_t n) noexcept {
    if (n <= 1) return;
    if (n == 2) { if (p[0] > p[1]) std::swap(p[0], p[1]); return; }
    __m512i inf = _mm512_set1_epi32(-1);
    if (n <= 16) {
        __mmask16 m = static_cast<__mmask16>((1u << n) - 1u);
        _mm512_mask_storeu_epi32(p, m, sort16_u32(_mm512_mask_loadu_epi32(inf, m, p)));
        return;
    }
    if (n <= 32) {
        __m512i v0 = sort16_u32(_mm512_loadu_si512(p));
        __mmask16 m1 = static_cast<__mmask16>((1u << (n - 16)) - 1u);
        __m512i v1 = sort16_u32(_mm512_mask_loadu_epi32(inf, m1, p + 16));
        merge32_u32(v0, v1);
        _mm512_storeu_si512(p, v0);
        _mm512_mask_storeu_epi32(p + 16, m1, v1);
        return;
    }
    if (n <= 64) {
        __m512i v0 = sort16_u32(_mm512_loadu_si512(p));
        __m512i v1 = sort16_u32(_mm512_loadu_si512(p + 16));
        __m512i v2, v3; __mmask16 m2, m3;
        if (n <= 48) {
            m2 = static_cast<__mmask16>((1u << (n - 32)) - 1u); m3 = 0;
            v2 = sort16_u32(_mm512_mask_loadu_epi32(inf, m2, p + 32)); v3 = inf;
        } else {
            m2 = 0xFFFF; m3 = static_cast<__mmask16>((1u << (n - 48)) - 1u);
            v2 = sort16_u32(_mm512_loadu_si512(p + 32));
            v3 = sort16_u32(_mm512_mask_loadu_epi32(inf, m3, p + 48));
        }
        merge32_u32(v0, v1); merge32_u32(v2, v3); merge64_u32(v0, v1, v2, v3);
        _mm512_storeu_si512(p, v0); _mm512_storeu_si512(p + 16, v1);
        _mm512_mask_storeu_epi32(p + 32, m2, v2);
        if (m3) _mm512_mask_storeu_epi32(p + 48, m3, v3);
        return;
    }
    __m512i v[8]; __mmask16 masks[8];
    for (int i = 0; i < 8; ++i) {
        size_t off = static_cast<size_t>(i) * 16;
        if (off + 16 <= n) { masks[i] = 0xFFFF; v[i] = sort16_u32(_mm512_loadu_si512(p + off)); }
        else if (off < n)  { masks[i] = static_cast<__mmask16>((1u << (n - off)) - 1u); v[i] = sort16_u32(_mm512_mask_loadu_epi32(inf, masks[i], p + off)); }
        else               { masks[i] = 0; v[i] = inf; }
    }
    merge32_u32(v[0], v[1]); merge32_u32(v[2], v[3]); merge32_u32(v[4], v[5]); merge32_u32(v[6], v[7]);
    merge64_u32(v[0], v[1], v[2], v[3]); merge64_u32(v[4], v[5], v[6], v[7]);
    __m512i rev = _mm512_set_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    __m512i r4 = _mm512_permutexvar_epi32(rev, v[4]), r5 = _mm512_permutexvar_epi32(rev, v[5]);
    __m512i r6 = _mm512_permutexvar_epi32(rev, v[6]), r7 = _mm512_permutexvar_epi32(rev, v[7]);
    __m512i a0 = _mm512_min_epu32(v[0], r7), b3 = _mm512_max_epu32(v[0], r7);
    __m512i a1 = _mm512_min_epu32(v[1], r6), b2 = _mm512_max_epu32(v[1], r6);
    __m512i a2 = _mm512_min_epu32(v[2], r5), b1 = _mm512_max_epu32(v[2], r5);
    __m512i a3 = _mm512_min_epu32(v[3], r4), b0 = _mm512_max_epu32(v[3], r4);
    bm64_u32(a0, a1, a2, a3); bm64_u32(b0, b1, b2, b3);
    _mm512_storeu_si512(p + 0, a0); _mm512_storeu_si512(p + 16, a1);
    _mm512_storeu_si512(p + 32, a2); _mm512_storeu_si512(p + 48, a3);
    if (masks[4]) _mm512_mask_storeu_epi32(p + 64, masks[4], b0);
    if (masks[5]) _mm512_mask_storeu_epi32(p + 80, masks[5], b1);
    if (masks[6]) _mm512_mask_storeu_epi32(p + 96, masks[6], b2);
    if (masks[7]) _mm512_mask_storeu_epi32(p + 112, masks[7], b3);
}

inline __m512i sort8_u64(__m512i v) noexcept {
    auto cs = [](__m512i a, __m512i idx, __mmask8 hm) KDR_LAMBDA_INLINE {
        __m512i p = _mm512_permutexvar_epi64(idx, a);
        return _mm512_mask_blend_epi64(hm, _mm512_min_epu64(a, p), _mm512_max_epu64(a, p));
    };
    v = cs(v, _mm512_set_epi64(6,7,4,5,2,3,0,1), 0x66);
    v = cs(v, _mm512_set_epi64(5,4,7,6,1,0,3,2), 0x3C);
    v = cs(v, _mm512_set_epi64(6,7,4,5,2,3,0,1), 0x5A);
    v = cs(v, _mm512_set_epi64(3,2,1,0,7,6,5,4), 0xF0);
    v = cs(v, _mm512_set_epi64(5,4,7,6,1,0,3,2), 0xCC);
    v = cs(v, _mm512_set_epi64(6,7,4,5,2,3,0,1), 0xAA);
    return v;
}

inline __m512i bm8_u64(__m512i v) noexcept {
    auto cs = [](__m512i a, __m512i idx, __mmask8 hm) KDR_LAMBDA_INLINE {
        __m512i p = _mm512_permutexvar_epi64(idx, a);
        return _mm512_mask_blend_epi64(hm, _mm512_min_epu64(a, p), _mm512_max_epu64(a, p));
    };
    v = cs(v, _mm512_set_epi64(3,2,1,0,7,6,5,4), 0xF0);
    v = cs(v, _mm512_set_epi64(5,4,7,6,1,0,3,2), 0xCC);
    v = cs(v, _mm512_set_epi64(6,7,4,5,2,3,0,1), 0xAA);
    return v;
}

inline void bm16_u64(__m512i& a, __m512i& b) noexcept {
    __m512i lo = _mm512_min_epu64(a, b), hi = _mm512_max_epu64(a, b);
    a = bm8_u64(lo); b = bm8_u64(hi);
}

inline void bm32_u64(__m512i& v0, __m512i& v1, __m512i& v2, __m512i& v3) noexcept {
    __m512i a0 = _mm512_min_epu64(v0, v2), b0 = _mm512_max_epu64(v0, v2);
    __m512i a1 = _mm512_min_epu64(v1, v3), b1 = _mm512_max_epu64(v1, v3);
    bm16_u64(a0, a1); bm16_u64(b0, b1);
    v0 = a0; v1 = a1; v2 = b0; v3 = b1;
}

inline void merge16_u64(__m512i& a, __m512i& b) noexcept {
    __m512i rev = _mm512_set_epi64(0,1,2,3,4,5,6,7);
    __m512i br = _mm512_permutexvar_epi64(rev, b);
    __m512i lo = _mm512_min_epu64(a, br), hi = _mm512_max_epu64(a, br);
    a = bm8_u64(lo); b = bm8_u64(hi);
}

inline void merge32_u64(__m512i& v0, __m512i& v1, __m512i& v2, __m512i& v3) noexcept {
    __m512i rev = _mm512_set_epi64(0,1,2,3,4,5,6,7);
    __m512i v3r = _mm512_permutexvar_epi64(rev, v3), v2r = _mm512_permutexvar_epi64(rev, v2);
    __m512i a0 = _mm512_min_epu64(v0, v3r), b1 = _mm512_max_epu64(v0, v3r);
    __m512i a1 = _mm512_min_epu64(v1, v2r), b0 = _mm512_max_epu64(v1, v2r);
    bm16_u64(a0, a1); bm16_u64(b0, b1);
    v0 = a0; v1 = a1; v2 = b0; v3 = b1;
}

inline void sort_small_u64_zmm(uint64_t* p, size_t n) noexcept {
    if (n <= 1) return;
    if (n == 2) { if (p[0] > p[1]) std::swap(p[0], p[1]); return; }
    __m512i inf = _mm512_set1_epi64(-1LL);
    if (n <= 8) {
        __mmask8 m = static_cast<__mmask8>((1u << n) - 1u);
        _mm512_mask_storeu_epi64(p, m, sort8_u64(_mm512_mask_loadu_epi64(inf, m, p)));
        return;
    }
    if (n <= 16) {
        __m512i v0 = sort8_u64(_mm512_loadu_si512(p));
        __mmask8 m1 = static_cast<__mmask8>((1u << (n - 8)) - 1u);
        __m512i v1 = sort8_u64(_mm512_mask_loadu_epi64(inf, m1, p + 8));
        merge16_u64(v0, v1);
        _mm512_storeu_si512(p, v0); _mm512_mask_storeu_epi64(p + 8, m1, v1);
        return;
    }
    if (n <= 32) {
        __m512i v0 = sort8_u64(_mm512_loadu_si512(p)), v1 = sort8_u64(_mm512_loadu_si512(p + 8));
        __m512i v2, v3; __mmask8 m2, m3;
        if (n <= 24) { m2 = static_cast<__mmask8>((1u << (n - 16)) - 1u); m3 = 0; v2 = sort8_u64(_mm512_mask_loadu_epi64(inf, m2, p + 16)); v3 = inf; }
        else         { m2 = 0xFF; m3 = static_cast<__mmask8>((1u << (n - 24)) - 1u); v2 = sort8_u64(_mm512_loadu_si512(p + 16)); v3 = sort8_u64(_mm512_mask_loadu_epi64(inf, m3, p + 24)); }
        merge16_u64(v0, v1); merge16_u64(v2, v3); merge32_u64(v0, v1, v2, v3);
        _mm512_storeu_si512(p, v0); _mm512_storeu_si512(p + 8, v1);
        _mm512_mask_storeu_epi64(p + 16, m2, v2);
        if (m3) _mm512_mask_storeu_epi64(p + 24, m3, v3);
        return;
    }
    __m512i v[8]; __mmask8 masks[8];
    for (int i = 0; i < 8; ++i) {
        size_t off = static_cast<size_t>(i) * 8;
        if (off + 8 <= n) { masks[i] = 0xFF; v[i] = sort8_u64(_mm512_loadu_si512(p + off)); }
        else if (off < n) { masks[i] = static_cast<__mmask8>((1u << (n - off)) - 1u); v[i] = sort8_u64(_mm512_mask_loadu_epi64(inf, masks[i], p + off)); }
        else              { masks[i] = 0; v[i] = inf; }
    }
    merge16_u64(v[0], v[1]); merge16_u64(v[2], v[3]); merge16_u64(v[4], v[5]); merge16_u64(v[6], v[7]);
    merge32_u64(v[0], v[1], v[2], v[3]); merge32_u64(v[4], v[5], v[6], v[7]);
    __m512i rev = _mm512_set_epi64(0,1,2,3,4,5,6,7);
    __m512i r4 = _mm512_permutexvar_epi64(rev, v[4]), r5 = _mm512_permutexvar_epi64(rev, v[5]);
    __m512i r6 = _mm512_permutexvar_epi64(rev, v[6]), r7 = _mm512_permutexvar_epi64(rev, v[7]);
    __m512i a0 = _mm512_min_epu64(v[0], r7), b3 = _mm512_max_epu64(v[0], r7);
    __m512i a1 = _mm512_min_epu64(v[1], r6), b2 = _mm512_max_epu64(v[1], r6);
    __m512i a2 = _mm512_min_epu64(v[2], r5), b1 = _mm512_max_epu64(v[2], r5);
    __m512i a3 = _mm512_min_epu64(v[3], r4), b0 = _mm512_max_epu64(v[3], r4);
    bm32_u64(a0, a1, a2, a3); bm32_u64(b0, b1, b2, b3);
    _mm512_storeu_si512(p + 0, a0); _mm512_storeu_si512(p + 8, a1);
    _mm512_storeu_si512(p + 16, a2); _mm512_storeu_si512(p + 24, a3);
    if (masks[4]) _mm512_mask_storeu_epi64(p + 32, masks[4], b0);
    if (masks[5]) _mm512_mask_storeu_epi64(p + 40, masks[5], b1);
    if (masks[6]) _mm512_mask_storeu_epi64(p + 48, masks[6], b2);
    if (masks[7]) _mm512_mask_storeu_epi64(p + 56, masks[7], b3);
}

template <typename U>
inline U* bit_partition_avx512(U* first, U* last, int bit) noexcept {
    constexpr size_t V = 64 / sizeof(U); // 16 for u32, 8 for u64
    U bit_val = static_cast<U>(1ULL << bit);
    size_t n = static_cast<size_t>(last - first);
    if (n < 2 * V) {
        U* l = first; U* r = last;
        while (l < r) {
            while (l < r && ((*l & bit_val) == 0)) ++l;
            while (l < r && ((*(r - 1) & bit_val) != 0)) --r;
            if (l < r) { --r; std::swap(*l, *r); ++l; }
        }
        return l;
    }

    __m512i vbit = (sizeof(U) == 4)
        ? _mm512_set1_epi32(static_cast<int>(bit_val))
        : _mm512_set1_epi64(static_cast<long long>(bit_val));

    U* write_l = first;
    U* write_r = last;

    auto compress_fast = [&](__m512i vec) KDR_LAMBDA_INLINE {
        if constexpr (sizeof(U) == 4) {
            __mmask16 m_one = _mm512_test_epi32_mask(vec, vbit);
            __mmask16 m_zero = static_cast<__mmask16>(~m_one);
            int n_zero = kdr_popcount32(static_cast<unsigned>(m_zero));
            int n_one = 16 - n_zero;
            _mm512_storeu_si512(write_l, _mm512_maskz_compress_epi32(m_zero, vec));
            write_l += n_zero;
            write_r -= n_one;
            _mm512_mask_compressstoreu_epi32(write_r, m_one, vec);
        } else {
            __mmask8 m_one = _mm512_test_epi64_mask(vec, vbit);
            __mmask8 m_zero = static_cast<__mmask8>(~m_one);
            int n_zero = kdr_popcount32(static_cast<unsigned>(m_zero));
            int n_one = 8 - n_zero;
            _mm512_storeu_si512(write_l, _mm512_maskz_compress_epi64(m_zero, vec));
            write_l += n_zero;
            write_r -= n_one;
            _mm512_mask_compressstoreu_epi64(write_r, m_one, vec);
        }
    };

    // FIX 4: Safe exact compress store preventing unaligned overwrite
    auto compress_exact = [&](__m512i vec) KDR_LAMBDA_INLINE {
        if constexpr (sizeof(U) == 4) {
            __mmask16 m_one = _mm512_test_epi32_mask(vec, vbit);
            __mmask16 m_zero = static_cast<__mmask16>(~m_one);
            int n_zero = kdr_popcount32(static_cast<unsigned>(m_zero));
            int n_one = 16 - n_zero;
            _mm512_mask_compressstoreu_epi32(write_l, m_zero, vec);
            write_l += n_zero;
            write_r -= n_one;
            _mm512_mask_compressstoreu_epi32(write_r, m_one, vec);
        } else {
            __mmask8 m_one = _mm512_test_epi64_mask(vec, vbit);
            __mmask8 m_zero = static_cast<__mmask8>(~m_one);
            int n_zero = kdr_popcount32(static_cast<unsigned>(m_zero));
            int n_one = 8 - n_zero;
            _mm512_mask_compressstoreu_epi64(write_l, m_zero, vec);
            write_l += n_zero;
            write_r -= n_one;
            _mm512_mask_compressstoreu_epi64(write_r, m_one, vec);
        }
    };

    if (n >= 8 * V) {
        __m512i vl0 = _mm512_loadu_si512(first);
        __m512i vl1 = _mm512_loadu_si512(first + V);
        __m512i vl2 = _mm512_loadu_si512(first + 2 * V);
        __m512i vl3 = _mm512_loadu_si512(first + 3 * V);
        __m512i vr0 = _mm512_loadu_si512(last - 4 * V);
        __m512i vr1 = _mm512_loadu_si512(last - 3 * V);
        __m512i vr2 = _mm512_loadu_si512(last - 2 * V);
        __m512i vr3 = _mm512_loadu_si512(last - V);

        U* read_l = first + 4 * V;
        U* read_r = last - 4 * V;

        while (static_cast<size_t>(read_r - read_l) >= 4 * V) {
            if (static_cast<size_t>(read_l - write_l) <= static_cast<size_t>(write_r - read_r)) {
                __m512i v0 = _mm512_loadu_si512(read_l);
                __m512i v1 = _mm512_loadu_si512(read_l + V);
                __m512i v2 = _mm512_loadu_si512(read_l + 2 * V);
                __m512i v3 = _mm512_loadu_si512(read_l + 3 * V);
                read_l += 4 * V;
                compress_fast(v0); compress_fast(v1); compress_fast(v2); compress_fast(v3);
            } else {
                read_r -= 4 * V;
                __m512i v0 = _mm512_loadu_si512(read_r);
                __m512i v1 = _mm512_loadu_si512(read_r + V);
                __m512i v2 = _mm512_loadu_si512(read_r + 2 * V);
                __m512i v3 = _mm512_loadu_si512(read_r + 3 * V);
                compress_fast(v0); compress_fast(v1); compress_fast(v2); compress_fast(v3);
            }
        }
        while (static_cast<size_t>(read_r - read_l) >= V) {
            __m512i vec;
            if (static_cast<size_t>(read_l - write_l) <= static_cast<size_t>(write_r - read_r)) {
                vec = _mm512_loadu_si512(read_l); read_l += V;
            } else {
                read_r -= V; vec = _mm512_loadu_si512(read_r);
            }
            compress_fast(vec);
        }
        size_t rem = static_cast<size_t>(read_r - read_l);
        U rem_buf[V];
        if (rem > 0) std::memcpy(rem_buf, read_l, rem * sizeof(U));

        // FIX 4: Use exact store when flushing residual preloaded slack vectors!
        compress_exact(vl0); compress_exact(vl1); compress_exact(vl2); compress_exact(vl3);
        compress_exact(vr0); compress_exact(vr1); compress_exact(vr2); compress_exact(vr3);

        for (size_t i = 0; i < rem; ++i) {
            U x = rem_buf[i];
            if ((x & bit_val) == 0) *write_l++ = x; else *--write_r = x;
        }
        return write_l;
    }

    __m512i v_left = _mm512_loadu_si512(first);
    __m512i v_right = _mm512_loadu_si512(last - V);
    U* read_l = first + V; U* read_r = last - V;
    while (static_cast<size_t>(read_r - read_l) >= V) {
        __m512i vec;
        if (static_cast<size_t>(read_l - write_l) <= static_cast<size_t>(write_r - read_r)) {
            vec = _mm512_loadu_si512(read_l); read_l += V;
        } else {
            read_r -= V; vec = _mm512_loadu_si512(read_r);
        }
        compress_exact(vec);
    }
    size_t rem = static_cast<size_t>(read_r - read_l);
    U rem_buf[V];
    if (rem > 0) std::memcpy(rem_buf, read_l, rem * sizeof(U));
    compress_exact(v_left); compress_exact(v_right);
    for (size_t i = 0; i < rem; ++i) {
        U x = rem_buf[i];
        if ((x & bit_val) == 0) *write_l++ = x; else *--write_r = x;
    }
    return write_l;
}

template <typename U>
void bit_msd_avx512_rec(U* first, U* last, int bit) noexcept {
    constexpr size_t NET_CUTOFF = (sizeof(U) == 4) ? 128 : 64;
    while (static_cast<size_t>(last - first) > NET_CUTOFF && bit >= 0) {
        U* mid = bit_partition_avx512(first, last, bit);
        --bit;
        if (mid - first < last - mid) {
            if (mid - first > 1) bit_msd_avx512_rec(first, mid, bit);
            first = mid;
        } else {
            if (last - mid > 1) bit_msd_avx512_rec(mid, last, bit);
            last = mid;
        }
    }
    size_t n = static_cast<size_t>(last - first);
    if (n > 1) {
        if constexpr (sizeof(U) == 4) sort_small_u32_zmm(first, n);
        else                          sort_small_u64_zmm(first, n);
    }
}

template <typename U>
void bit_msd_avx512_parallel(U* first, U* last, int bit, int num_threads) {
    if (num_threads <= 1 || static_cast<size_t>(last - first) < 262144 || bit < 0) {
        bit_msd_avx512_rec(first, last, bit);
        return;
    }
    U* mid = bit_partition_avx512(first, last, bit);
    int left_t = num_threads / 2;
    int right_t = num_threads - left_t;

    ThreadPool::instance().parallel_for(2, [&](int task_id) {
        if (task_id == 0) {
            bit_msd_avx512_parallel(first, mid, bit - 1, left_t);
        } else {
            bit_msd_avx512_parallel(mid, last, bit - 1, right_t);
        }
    });
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif // __AVX512F__

// ============================================================================
// BACKEND 2 & 10: BLOCK-BUFFERED IN-PLACE MSD (AVX2 & PORTABLE SCALAR)
// ============================================================================
template <typename U, size_t B>
inline void permute_blocks_avx2(U* data, int shift, size_t blk_next[256], const size_t blk_end[256]) noexcept {
    constexpr size_t BYTES = B * sizeof(U);
#if defined(__AVX2__)
    if constexpr (BYTES == 64) {
        for (int b = 0; b < 256; ++b) {
            while (blk_next[b] < blk_end[b]) {
                size_t cur = blk_next[b];
                U* cur_ptr = data + cur * B;
                uint8_t dest = static_cast<uint8_t>((cur_ptr[0] >> shift) & 0xFF);
                if (dest == b) { blk_next[b]++; continue; }
                __m256i* cp = reinterpret_cast<__m256i*>(cur_ptr);
                __m256i t0 = _mm256_loadu_si256(cp + 0), t1 = _mm256_loadu_si256(cp + 1);
                do {
                    size_t target = blk_next[dest]++;
                    __m256i* tp = reinterpret_cast<__m256i*>(data + target * B);
                    __m256i s0 = _mm256_loadu_si256(tp + 0), s1 = _mm256_loadu_si256(tp + 1);
                    _mm256_storeu_si256(tp + 0, t0); _mm256_storeu_si256(tp + 1, t1);
                    t0 = s0; t1 = s1;
                    U first_elem = static_cast<U>(_mm256_cvtsi256_si32(t0));
                    dest = static_cast<uint8_t>((first_elem >> shift) & 0xFF);
                } while (dest != b);
                _mm256_storeu_si256(cp + 0, t0); _mm256_storeu_si256(cp + 1, t1);
                blk_next[b]++;
            }
        }
        return;
    } else if constexpr (BYTES == 128) {
        for (int b = 0; b < 256; ++b) {
            while (blk_next[b] < blk_end[b]) {
                size_t cur = blk_next[b];
                U* cur_ptr = data + cur * B;
                uint8_t dest = static_cast<uint8_t>((cur_ptr[0] >> shift) & 0xFF);
                if (dest == b) { blk_next[b]++; continue; }
                __m256i* cp = reinterpret_cast<__m256i*>(cur_ptr);
                __m256i t0 = _mm256_loadu_si256(cp + 0), t1 = _mm256_loadu_si256(cp + 1);
                __m256i t2 = _mm256_loadu_si256(cp + 2), t3 = _mm256_loadu_si256(cp + 3);
                do {
                    size_t target = blk_next[dest]++;
                    __m256i* tp = reinterpret_cast<__m256i*>(data + target * B);
                    __m256i s0 = _mm256_loadu_si256(tp + 0), s1 = _mm256_loadu_si256(tp + 1);
                    __m256i s2 = _mm256_loadu_si256(tp + 2), s3 = _mm256_loadu_si256(tp + 3);
                    _mm256_storeu_si256(tp + 0, t0); _mm256_storeu_si256(tp + 1, t1);
                    _mm256_storeu_si256(tp + 2, t2); _mm256_storeu_si256(tp + 3, t3);
                    t0 = s0; t1 = s1; t2 = s2; t3 = s3;
                    U first_elem;
                    if constexpr (sizeof(U) == 8) {
                        first_elem = static_cast<U>(_mm_cvtsi128_si64(_mm256_castsi256_si128(t0)));
                    } else {
                        first_elem = static_cast<U>(_mm256_cvtsi256_si32(t0));
                    }
                    dest = static_cast<uint8_t>((first_elem >> shift) & 0xFF);
                } while (dest != b);
                _mm256_storeu_si256(cp + 0, t0); _mm256_storeu_si256(cp + 1, t1);
                _mm256_storeu_si256(cp + 2, t2); _mm256_storeu_si256(cp + 3, t3);
                blk_next[b]++;
            }
        }
        return;
    } else if constexpr (BYTES == 256) {
        for (int b = 0; b < 256; ++b) {
            while (blk_next[b] < blk_end[b]) {
                size_t cur = blk_next[b];
                U* cur_ptr = data + cur * B;
                uint8_t dest = static_cast<uint8_t>((cur_ptr[0] >> shift) & 0xFF);
                if (dest == b) { blk_next[b]++; continue; }
                __m256i* cp = reinterpret_cast<__m256i*>(cur_ptr);
                __m256i t0 = _mm256_loadu_si256(cp + 0), t1 = _mm256_loadu_si256(cp + 1);
                __m256i t2 = _mm256_loadu_si256(cp + 2), t3 = _mm256_loadu_si256(cp + 3);
                __m256i t4 = _mm256_loadu_si256(cp + 4), t5 = _mm256_loadu_si256(cp + 5);
                __m256i t6 = _mm256_loadu_si256(cp + 6), t7 = _mm256_loadu_si256(cp + 7);
                do {
                    size_t target = blk_next[dest]++;
                    __m256i* tp = reinterpret_cast<__m256i*>(data + target * B);
                    __m256i s0 = _mm256_loadu_si256(tp + 0), s1 = _mm256_loadu_si256(tp + 1);
                    __m256i s2 = _mm256_loadu_si256(tp + 2), s3 = _mm256_loadu_si256(tp + 3);
                    __m256i s4 = _mm256_loadu_si256(tp + 4), s5 = _mm256_loadu_si256(tp + 5);
                    __m256i s6 = _mm256_loadu_si256(tp + 6), s7 = _mm256_loadu_si256(tp + 7);
                    _mm256_storeu_si256(tp + 0, t0); _mm256_storeu_si256(tp + 1, t1);
                    _mm256_storeu_si256(tp + 2, t2); _mm256_storeu_si256(tp + 3, t3);
                    _mm256_storeu_si256(tp + 4, t4); _mm256_storeu_si256(tp + 5, t5);
                    _mm256_storeu_si256(tp + 6, t6); _mm256_storeu_si256(tp + 7, t7);
                    t0 = s0; t1 = s1; t2 = s2; t3 = s3; t4 = s4; t5 = s5; t6 = s6; t7 = s7;
                    U first_elem = static_cast<U>(_mm_cvtsi128_si64(_mm256_castsi256_si128(t0)));
                    dest = static_cast<uint8_t>((first_elem >> shift) & 0xFF);
                } while (dest != b);
                _mm256_storeu_si256(cp + 0, t0); _mm256_storeu_si256(cp + 1, t1);
                _mm256_storeu_si256(cp + 2, t2); _mm256_storeu_si256(cp + 3, t3);
                _mm256_storeu_si256(cp + 4, t4); _mm256_storeu_si256(cp + 5, t5);
                _mm256_storeu_si256(cp + 6, t6); _mm256_storeu_si256(cp + 7, t7);
                blk_next[b]++;
            }
        }
        return;
    }
#endif
    // Portable Scalar block permutation fallback
    alignas(64) U tmp_blk[B], swap_blk[B];
    for (int b = 0; b < 256; ++b) {
        while (blk_next[b] < blk_end[b]) {
            size_t cur = blk_next[b];
            U* cur_ptr = data + cur * B;
            uint8_t dest = static_cast<uint8_t>((cur_ptr[0] >> shift) & 0xFF);
            if (dest == b) { blk_next[b]++; continue; }
            std::memcpy(tmp_blk, cur_ptr, BYTES);
            do {
                size_t target = blk_next[dest]++;
                U* tgt_ptr = data + target * B;
                std::memcpy(swap_blk, tgt_ptr, BYTES);
                std::memcpy(tgt_ptr, tmp_blk, BYTES);
                std::memcpy(tmp_blk, swap_blk, BYTES);
                dest = static_cast<uint8_t>((tmp_blk[0] >> shift) & 0xFF);
            } while (dest != b);
            std::memcpy(cur_ptr, tmp_blk, BYTES);
            blk_next[b]++;
        }
    }
}

// OPTIMIZATION: Inline AVX2 Vector Block Store (Bypasses memcpy function call overhead)
template <typename U, size_t B>
KDR_ALWAYS_INLINE void store_block_fast(U* dest, const U* src) noexcept {
    constexpr size_t BYTES = B * sizeof(U);
#if defined(KDR_X86_SIMD_AVAILABLE)
    if constexpr (BYTES == 64) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src) + 0);
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src) + 1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest) + 0, v0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest) + 1, v1);
        return;
    } else if constexpr (BYTES == 128) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src) + 0);
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src) + 1);
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src) + 2);
        __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src) + 3);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest) + 0, v0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest) + 1, v1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest) + 2, v2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest) + 3, v3);
        return;
    }
#endif
    std::memcpy(dest, src, BYTES);
}

template <typename U, size_t B = BLOCK_SIZE>
void block_partition_inplace(U* data, size_t n, int shift, size_t final_start[256], size_t count[256], int num_threads) {
    alignas(64) U merged_buf[256][B];
    uint32_t merged_len[256] = {0};
    size_t full_blocks[256] = {0};
    size_t total_M = 0;

    if (num_threads <= 1 || n < 524288) {
        size_t w = 0;
        for (size_t r = 0; r < n; ++r) {
#if defined(KDR_X86_SIMD_AVAILABLE)
            if (r + 32 < n) {
                _mm_prefetch(reinterpret_cast<const char*>(data + r + 32), _MM_HINT_T0);
            }
#endif
            U v = data[r];
            uint8_t d = static_cast<uint8_t>((v >> shift) & 0xFF);
            uint32_t len = merged_len[d];
            merged_buf[d][len++] = v;
            if (len == B) {
                store_block_fast<U, B>(data + w, merged_buf[d]);
                w += B;
                full_blocks[d]++;
                len = 0;
            }
            merged_len[d] = len;
        }
        total_M = w / B;
    } else {
        int T = std::min(num_threads, 16);
        struct ThreadState {
            alignas(64) U buf[256][B];
            uint32_t len[256] = {0};
            size_t full_blk[256] = {0};
            size_t lo = 0, hi = 0, w = 0;
        };
        std::vector<ThreadState> st(static_cast<size_t>(T));
        size_t total_blocks_cap = n / B;
        size_t blk_per_t = (total_blocks_cap + static_cast<size_t>(T) - 1) / static_cast<size_t>(T);

        for (int t = 0; t < T; ++t) {
            st[t].lo = std::min(n, static_cast<size_t>(t) * blk_per_t * B);
            st[t].hi = (t == T - 1) ? n : std::min(n, static_cast<size_t>(t + 1) * blk_per_t * B);
        }

        // FIX 7: Use warm ThreadPool instead of spawning fresh OS threads
        ThreadPool::instance().parallel_for(T, [&](int t) {
            auto& s = st[t];
            size_t w = s.lo;
            for (size_t r = s.lo; r < s.hi; ++r) {
#if defined(KDR_X86_SIMD_AVAILABLE)
                if (r + 32 < s.hi) {
                    _mm_prefetch(reinterpret_cast<const char*>(data + r + 32), _MM_HINT_T0);
                }
#endif
                U v = data[r];
                uint8_t d = static_cast<uint8_t>((v >> shift) & 0xFF);
                uint32_t len = s.len[d];
                s.buf[d][len++] = v;
                if (len == B) {
                    store_block_fast<U, B>(data + w, s.buf[d]);
                    w += B;
                    s.full_blk[d]++;
                    len = 0;
                }
                s.len[d] = len;
            }
            s.w = w;
        });

        for (int t = 0; t < T; ++t)
            for (int b = 0; b < 256; ++b) full_blocks[b] += st[t].full_blk[b];

        int src_t = T - 1;
        size_t src_pos = st[src_t].w;
        for (int dst_t = 0; dst_t < T; ++dst_t) {
            size_t gap_cur = st[dst_t].w;
            size_t gap_end = (dst_t == T - 1) ? st[dst_t].w : st[dst_t + 1].lo;
            while (gap_cur < gap_end) {
                while (src_t > dst_t && src_pos <= st[src_t].lo) { src_t--; src_pos = st[src_t].w; }
                if (src_t <= dst_t) break;
                src_pos -= B;
                store_block_fast<U, B>(data + gap_cur, data + src_pos);
                gap_cur += B;
                st[src_t].w = src_pos;
            }
            if (src_t <= dst_t) { total_M = gap_cur / B; break; }
        }
        if (total_M == 0) for (int b = 0; b < 256; ++b) total_M += full_blocks[b];

        size_t w = total_M * B;
        for (int b = 0; b < 256; ++b) {
            uint32_t mlen = 0;
            for (int t = 0; t < T; ++t) {
                for (uint32_t k = 0; k < st[t].len[b]; ++k) {
                    merged_buf[b][mlen++] = st[t].buf[b][k];
                    if (mlen == B) {
                        store_block_fast<U, B>(data + w, merged_buf[b]);
                        w += B;
                        full_blocks[b]++;
                        mlen = 0;
                    }
                }
            }
            merged_len[b] = mlen;
        }
        total_M = w / B;
    }

    size_t blk_start[256], blk_next[256], blk_end[256];
    size_t b_sum = 0, f_sum = 0;
    int non_empty_blk = 0;
    for (int b = 0; b < 256; ++b) {
        count[b] = full_blocks[b] * B + merged_len[b];
        final_start[b] = f_sum; f_sum += count[b];
        blk_start[b] = b_sum; blk_next[b] = b_sum; b_sum += full_blocks[b]; blk_end[b] = b_sum;
        if (full_blocks[b] > 0) non_empty_blk++;
    }

    if (non_empty_blk > 1) permute_blocks_avx2<U, B>(data, shift, blk_next, blk_end);

    for (int b = 255; b >= 0; --b) {
        size_t F = full_blocks[b] * B, L = blk_start[b] * B, R = L + F;
        size_t f_start = final_start[b], f_end = f_start + count[b];
        if (merged_len[b] > 0) {
            std::memcpy(data + (f_end - merged_len[b]), merged_buf[b], merged_len[b] * sizeof(U));
        }
        size_t delta = f_start - L;
        if (F > 0 && delta > 0) {
            // FIX: Use std::memmove to safely handle overlapping memory regions without Undefined Behavior
            if (delta >= F) std::memmove(data + f_start, data + L, F * sizeof(U));
            else            std::memmove(data + R, data + L, delta * sizeof(U));
        }
    }
}

template <typename U>
inline void l2_lsd_24bit(U* first, size_t n, U* scratch) noexcept {
    if (n <= INSERTION_CUTOFF) { insertion_sort(first, first + n); return; }
    // OPTIMIZATION 3: Three 8-bit passes (256 counters = 1 KB) instead of two 12-bit passes (4096 counters = 32 KB)
    // Completely eliminates store-forwarding stalls and cache thrashing!
    uint32_t c0[256] = {0};
    for (size_t i = 0; i < n; ++i) c0[first[i] & 0xFF]++;
    uint32_t s0 = 0;
    for (int i = 0; i < 256; ++i) { uint32_t t0 = c0[i]; c0[i] = s0; s0 += t0; }
    for (size_t i = 0; i < n; ++i) scratch[c0[first[i] & 0xFF]++] = first[i];

    uint32_t c1[256] = {0};
    for (size_t i = 0; i < n; ++i) c1[(scratch[i] >> 8) & 0xFF]++;
    uint32_t s1 = 0;
    for (int i = 0; i < 256; ++i) { uint32_t t1 = c1[i]; c1[i] = s1; s1 += t1; }
    for (size_t i = 0; i < n; ++i) first[c1[(scratch[i] >> 8) & 0xFF]++] = scratch[i];

    uint32_t c2[256] = {0};
    for (size_t i = 0; i < n; ++i) c2[(first[i] >> 16) & 0xFF]++;
    uint32_t s2 = 0;
    for (int i = 0; i < 256; ++i) { uint32_t t2 = c2[i]; c2[i] = s2; s2 += t2; }
    for (size_t i = 0; i < n; ++i) scratch[c2[(first[i] >> 16) & 0xFF]++] = first[i];
    std::memcpy(first, scratch, n * sizeof(U));
}

template <typename U>
inline void l2_lsd_16bit(U* first, size_t n, U* scratch) noexcept {
    if (n <= INSERTION_CUTOFF) { insertion_sort(first, first + n); return; }
    uint32_t c0[256] = {0}, c1[256] = {0};
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32_t v0 = static_cast<uint32_t>(first[i]);
        uint32_t v1 = static_cast<uint32_t>(first[i+1]);
        uint32_t v2 = static_cast<uint32_t>(first[i+2]);
        uint32_t v3 = static_cast<uint32_t>(first[i+3]);
        c0[v0 & 0xFF]++; c1[(v0 >> 8) & 0xFF]++;
        c0[v1 & 0xFF]++; c1[(v1 >> 8) & 0xFF]++;
        c0[v2 & 0xFF]++; c1[(v2 >> 8) & 0xFF]++;
        c0[v3 & 0xFF]++; c1[(v3 >> 8) & 0xFF]++;
    }
    for (; i < n; ++i) {
        uint32_t v = static_cast<uint32_t>(first[i]);
        c0[v & 0xFF]++; c1[(v >> 8) & 0xFF]++;
    }
    uint32_t s0 = 0, s1 = 0;
    for (int j = 0; j < 256; ++j) {
        uint32_t t0 = c0[j]; c0[j] = s0; s0 += t0;
        uint32_t t1 = c1[j]; c1[j] = s1; s1 += t1;
    }
    for (size_t k = 0; k < n; ++k) scratch[c0[first[k] & 0xFF]++] = first[k];
    for (size_t k = 0; k < n; ++k) first[c1[(scratch[k] >> 8) & 0xFF]++] = scratch[k];
}

template <typename U>
void l2_msd_oop(U* first, size_t n, U* scratch, int shift) noexcept {
    if (n <= INSERTION_CUTOFF) { insertion_sort(first, first + n); return; }
    if (shift == 16) { l2_lsd_24bit(first, n, scratch); return; }
    if (shift == 8)  { l2_lsd_16bit(first, n, scratch); return; }
    uint32_t count[256] = {0};
    for (size_t i = 0; i < n; ++i) count[(first[i] >> shift) & 0xFF]++;
    uint32_t start[256], next[256], sum = 0;
    int non_empty = 0;
    for (int b = 0; b < 256; ++b) {
        start[b] = sum; next[b] = sum; sum += count[b];
        if (count[b] > 0) non_empty++;
    }
    if (non_empty > 1) {
        for (size_t i = 0; i < n; ++i) scratch[next[(first[i] >> shift) & 0xFF]++] = first[i];
        std::memcpy(first, scratch, n * sizeof(U));
    }
    if (shift == 0) return;
    for (int b = 0; b < 256; ++b) {
        if (count[b] > 1) l2_msd_oop(first + start[b], count[b], scratch + start[b], shift - 8);
    }
}

template <typename U>
void numeric_msd_rec(U* first, size_t n, U* scratch, int shift) {
    if (n <= INSERTION_CUTOFF) { insertion_sort(first, first + n); return; }
    if (n <= SCRATCH_CAP) { l2_msd_oop(first, n, scratch, shift); return; }
    size_t start[256], count[256];
    block_partition_inplace<U, BLOCK_SIZE>(first, n, shift, start, count, 1);
    if (shift == 0) return;
    for (int b = 0; b < 256; ++b) {
        if (count[b] > 1) numeric_msd_rec(first + start[b], count[b], scratch, shift - 8);
    }
}

// ============================================================================
// FIX 2: STRICT ALIASING SAFE NUMERIC DISPATCHER & TRANSFORMATIONS
// ============================================================================
template <typename T>
void sort_numeric(T* first, size_t n, int num_threads) {
    using Traits = RadixTraits<T>;
    using U = typename Traits::U;
    if (n <= 1) return;

    // Fast 10-ns monotonic pre-scan
    int mono = quick_monotonic_check(first, n);
    if (mono == 1) return;
    if (mono == 2) { std::reverse(first, first + n); return; }

    // FIX 2: Strict-Aliasing Safe Transformation to Unsigned Keys
    U* data = reinterpret_cast<U*>(static_cast<void*>(first));
    constexpr bool needs_xform = !std::is_same_v<T, U>;
    if constexpr (needs_xform) {
        for (size_t i = 0; i < n; ++i) {
            T val;
            std::memcpy(&val, first + i, sizeof(T));
            U k = Traits::to_key(val);
            std::memcpy(data + i, &k, sizeof(U));
        }
    }

    // Special Ultra-Fast 8-bit Counting Sort (256 buckets, O(N) time, O(1) space)
    if constexpr (sizeof(U) == 1) {
        size_t counts[256] = {0};
        for (size_t i = 0; i < n; ++i) counts[data[i]]++;
        size_t out_idx = 0;
        for (int b = 0; b < 256; ++b) {
            size_t c = counts[b];
            if (c > 0) {
                std::memset(data + out_idx, b, c);
                out_idx += c;
            }
        }
    }
    // Special Ultra-Fast 16-bit Counting Sort (65536 buckets, O(N) time)
    else if constexpr (sizeof(U) == 2) {
        if (n <= INSERTION_CUTOFF) {
            insertion_sort(data, data + n);
        } else {
            std::vector<uint32_t> counts(65536, 0);
            for (size_t i = 0; i < n; ++i) counts[data[i]]++;
            size_t out_idx = 0;
            for (uint32_t b = 0; b < 65536; ++b) {
                uint32_t c = counts[b];
                if (c > 0) {
                    std::fill_n(data + out_idx, c, static_cast<uint16_t>(b));
                    out_idx += c;
                }
            }
        }
    }
    else {
    // High-bit skip & Range tracking (for FIX 9 Counting Sort)
    U first_val = data[0];
    U d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    U min_val = first_val, max_val = first_val;

    size_t i = 1;
    for (; i + 4 <= n; i += 4) {
        d0 |= (data[i] ^ first_val);
        d1 |= (data[i + 1] ^ first_val);
        d2 |= (data[i + 2] ^ first_val);
        d3 |= (data[i + 3] ^ first_val);
        min_val = std::min({min_val, data[i], data[i+1], data[i+2], data[i+3]});
        max_val = std::max({max_val, data[i], data[i+1], data[i+2], data[i+3]});
    }
    for (; i < n; ++i) {
        d0 |= (data[i] ^ first_val);
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }
    U diff_mask = d0 | d1 | d2 | d3;

    if (diff_mask != 0) {
        // FIX 9: Small Range Fast-Path (< 65536 range)
        bool handled_by_counting = false;
        if constexpr (sizeof(U) <= 4) {
            handled_by_counting = try_counting_sort(data, n, min_val, max_val);
        }

        if (!handled_by_counting) {
#ifdef __AVX512F__
            int top_bit = 63 - kdr_clz64(static_cast<uint64_t>(diff_mask));
            if (num_threads <= 1) {
                bit_msd_avx512_rec(data, data + n, top_bit);
            } else {
                bit_msd_avx512_parallel(data, data + n, top_bit, std::min(num_threads, 16));
            }
#else
            int highest_bit = 63 - kdr_clz64(static_cast<uint64_t>(diff_mask));
            int top_shift = (highest_bit / 8) * 8;

            if (n <= SCRATCH_CAP || num_threads <= 1) {
                // OPTIMIZATION 4: Zero heap allocation via Thread-Local static scratch
                U* scratch = get_tls_scratch<U>();
                numeric_msd_rec(data, n, scratch, top_shift);
            } else {
                size_t start[256], count[256];
                block_partition_inplace<U, BLOCK_SIZE>(data, n, top_shift, start, count, num_threads);
                if (top_shift > 0) {
                    int T_cnt = std::min(num_threads, 16);
                    std::atomic<int> next_bucket{0};
                    ThreadPool::instance().parallel_for(T_cnt, [&](int) {
                        U* scratch = get_tls_scratch<U>();
                        while (true) {
                            int b = next_bucket.fetch_add(1, std::memory_order_relaxed);
                            if (b >= 256) break;
                            if (count[b] > 1) numeric_msd_rec(data + start[b], count[b], scratch, top_shift - 8);
                        }
                    });
                }
            }
#endif
        }
    }
    } // close else block for sizeof(U) > 2

    // FIX 2: Strict-Aliasing Safe Transformation from Unsigned Keys
    if constexpr (needs_xform) {
        for (size_t k = 0; k < n; ++k) {
            U key;
            std::memcpy(&key, data + k, sizeof(U));
            T val = Traits::from_key(key);
            std::memcpy(first + k, &val, sizeof(T));
        }
    }
}

// ============================================================================
// FIX 8: ROBUST 64-BIT STRING ENGINE (NO 32-BIT OVERFLOW, WORD-CACHED MSD)
// ============================================================================
inline uint64_t load_be64(const std::string& s, size_t depth) noexcept {
    size_t len = s.size();
    if (depth >= len) return 0;
    size_t rem = len - depth;
    const char* p = s.data() + depth;
    if (rem >= 8) {
        uint64_t v;
        std::memcpy(&v, p, 8);
        return kdr_bswap64(v);
    }
    uint64_t v = 0;
    std::memcpy(&v, p, rem);
    return kdr_bswap64(v);
}

inline bool string_less_from(const std::string& a, const std::string& b, size_t depth) noexcept {
    size_t la = (a.size() > depth) ? (a.size() - depth) : 0;
    size_t lb = (b.size() > depth) ? (b.size() - depth) : 0;
    size_t m = std::min(la, lb);
    int cmp = (m > 0) ? std::memcmp(a.data() + depth, b.data() + depth, m) : 0;
    return cmp < 0 || (cmp == 0 && la < lb);
}

// FIX 8: 64-bit index to prevent integer overflow beyond 4.29 billion strings
struct StrRec {
    uint64_t key;
    size_t idx;
};

inline void strrec_insertion(StrRec* first, size_t n, const std::string* orig, size_t depth) noexcept {
    for (size_t i = 1; i < n; ++i) {
        StrRec val = first[i];
        size_t j = i;
        while (j > 0) {
            if (first[j - 1].key > val.key ||
               (first[j - 1].key == val.key && string_less_from(orig[val.idx], orig[first[j - 1].idx], depth))) {
                first[j] = first[j - 1];
                --j;
            } else break;
        }
        first[j] = val;
    }
}

inline void strrec_msd_rec(StrRec* first, StrRec* buf, size_t n, const std::string* orig, size_t depth, int shift) {
    if (n <= 1) return;
    if (n <= 32) { strrec_insertion(first, n, orig, depth); return; }

    if (shift < 0) {
        size_t done = 0;
        for (size_t i = 0; i < n; ++i) {
            if (orig[first[i].idx].size() <= depth) {
                if (i != done) std::swap(first[i], first[done]);
                done++;
            }
        }
        first += done; buf += done; n -= done;
        if (n <= 1) return;
        if (n <= 32) {
            for (size_t i = 0; i < n; ++i) first[i].key = load_be64(orig[first[i].idx], depth);
            strrec_insertion(first, n, orig, depth);
            return;
        }
        uint64_t k0 = load_be64(orig[first[0].idx], depth);
        first[0].key = k0;
        uint64_t diff = 0;
        for (size_t i = 1; i < n; ++i) {
            uint64_t k = load_be64(orig[first[i].idx], depth);
            first[i].key = k;
            diff |= (k ^ k0);
        }
        if (diff == 0) { strrec_msd_rec(first, buf, n, orig, depth + 8, -1); return; }
        shift = ((63 - kdr_clz64(diff)) / 8) * 8;
    }

    uint32_t count[256] = {0};
    for (size_t i = 0; i < n; ++i) count[(first[i].key >> shift) & 0xFF]++;
    uint32_t start[256], next[256], sum = 0;
    int non_empty = 0;
    for (int b = 0; b < 256; ++b) {
        start[b] = sum; next[b] = sum; sum += count[b];
        if (count[b] > 0) non_empty++;
    }
    if (non_empty > 1) {
        for (size_t i = 0; i < n; ++i) buf[next[(first[i].key >> shift) & 0xFF]++] = first[i];
        std::memcpy(first, buf, n * sizeof(StrRec));
    }

    int next_shift = shift - 8;
    size_t next_depth = (next_shift < 0) ? (depth + 8) : depth;
    for (int b = 0; b < 256; ++b) {
        if (count[b] > 1) {
            strrec_msd_rec(first + start[b], buf + start[b], count[b], orig, next_depth, next_shift);
        }
    }
}

inline void sort_strings(std::string* first, size_t n, int num_threads) {
    if (n <= 1) return;
    if (n <= 32) { std::sort(first, first + n); return; }

    // OPTIMIZATION 5: Thread-local reusable buffers eliminating 32 MB heap allocation per sort
    thread_local std::vector<StrRec> tls_recs;
    thread_local std::vector<StrRec> tls_buf;
    thread_local std::vector<uint8_t> tls_visited;
    if (tls_recs.size() < n) tls_recs.resize(n);
    if (tls_buf.size() < n) tls_buf.resize(n);
    if (tls_visited.size() < n) tls_visited.resize(n);
    StrRec* recs = tls_recs.data();
    StrRec* buf = tls_buf.data();
    uint8_t* visited = tls_visited.data();
    std::fill_n(visited, n, static_cast<uint8_t>(0));

    size_t depth = 0;
    uint64_t diff = 0;
    while (true) {
        uint64_t k0 = load_be64(first[0], depth);
        diff = 0;
        bool any_longer = (first[0].size() > depth + 8);
        for (size_t i = 0; i < n; ++i) {
            uint64_t k = load_be64(first[i], depth);
            recs[i] = {k, i};
            diff |= (k ^ k0);
            any_longer |= (first[i].size() > depth + 8);
        }
        if (diff != 0 || !any_longer) break;
        depth += 8;
    }

    if (diff != 0) {
        int top_shift = ((63 - kdr_clz64(diff)) / 8) * 8;
        if (num_threads <= 1 || n < 32768) {
            strrec_msd_rec(recs, buf, n, first, depth, top_shift);
        } else {
            uint32_t count[256] = {0};
            for (size_t i = 0; i < n; ++i) count[(recs[i].key >> top_shift) & 0xFF]++;
            uint32_t start[256], next[256], sum = 0;
            for (int b = 0; b < 256; ++b) { start[b] = sum; next[b] = sum; sum += count[b]; }
            for (size_t i = 0; i < n; ++i) buf[next[(recs[i].key >> top_shift) & 0xFF]++] = recs[i];
            std::memcpy(recs, buf, n * sizeof(StrRec));

            int next_shift = top_shift - 8;
            size_t next_depth = (next_shift < 0) ? (depth + 8) : depth;
            std::atomic<int> next_bucket{0};
            int T = std::min(num_threads, 16);

            ThreadPool::instance().parallel_for(T, [&](int) {
                while (true) {
                    int b = next_bucket.fetch_add(1, std::memory_order_relaxed);
                    if (b >= 256) break;
                    if (count[b] > 1) {
                        strrec_msd_rec(recs + start[b], buf + start[b],
                                       count[b], first, next_depth, next_shift);
                    }
                }
            });
        }
    } else {
        strrec_insertion(recs, n, first, 0);
    }

    for (size_t i = 0; i < n; ++i) {
        if (visited[i] || recs[i].idx == i) continue;
        std::string tmp = std::move(first[i]);
        size_t cur = i;
        while (true) {
            visited[cur] = 1;
            size_t nxt = recs[cur].idx;
            if (nxt >= n) break; // Defensive bound check against out-of-bounds corruption
            if (nxt == i) { first[cur] = std::move(tmp); break; }
            first[cur] = std::move(first[nxt]);
            cur = nxt;
        }
    }
}

} // namespace detail

// ============================================================================
// FIX 6: PUBLIC API (UNSTABLE SORTING NOTICE & CONVENIENT ENTRYPOINTS)
// ============================================================================
/**
 * kdr::sort — In-Place High-Performance Radix Sort (UNSTABLE)
 * NOTE: This algorithm is strictly unstable; relative order of equal keys is not preserved.
 */
template <typename T>
inline void sort(T* first, T* last, int num_threads = 1) {
    if (first >= last) return;
    size_t n = static_cast<size_t>(last - first);
    if constexpr (std::is_same_v<T, std::string>) {
        detail::sort_strings(first, n, num_threads);
    } else {
        detail::sort_numeric(first, n, num_threads);
    }
}

template <typename T>
inline void sort(std::vector<T>& data, int num_threads = 1) {
    if (data.size() <= 1) return;
    sort(data.data(), data.data() + data.size(), num_threads);
}

template <typename T>
inline void parallel_sort(std::vector<T>& data) {
    int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    sort(data, hw);
}

template <typename T>
inline void parallel_sort(T* first, T* last) {
    int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    sort(first, last, hw);
}

} // namespace kdr
