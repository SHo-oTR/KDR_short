#include <kdr/kdr_sort.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

// 1. Branchless Block-Quicksort
inline uint32_t* branchless_partition(uint32_t* first, uint32_t* last) {
    uint32_t* mid = first + (last - first) / 2;
    if (*mid < *first) std::swap(*mid, *first);
    if (*(last - 1) < *first) std::swap(*(last - 1), *first);
    if (*(last - 1) < *mid) std::swap(*(last - 1), *mid);
    uint32_t pivot = *mid;
    std::swap(*mid, *(last - 1));

    constexpr int B = 64;
    uint8_t off_l[B], off_r[B];
    uint32_t* l = first;
    uint32_t* r = last - 1;
    int num_l = 0, num_r = 0, start_l = 0, start_r = 0;

    while (r - l >= 2 * B) {
        if (num_l == 0) {
            start_l = 0;
            for (int i = 0; i < B; ++i) {
                off_l[num_l] = static_cast<uint8_t>(i);
                num_l += (l[i] >= pivot);
            }
        }
        if (num_r == 0) {
            start_r = 0;
            for (int i = 0; i < B; ++i) {
                off_r[num_r] = static_cast<uint8_t>(i + 1);
                num_r += (r[-(i + 1)] <= pivot);
            }
        }
        int num = std::min(num_l, num_r);
        for (int i = 0; i < num; ++i) {
            std::swap(l[off_l[start_l + i]], r[-off_r[start_r + i]]);
        }
        num_l -= num; num_r -= num;
        start_l += num; start_r += num;
        if (num_l == 0) l += B;
        if (num_r == 0) r -= B;
    }
    while (l < r) {
        while (l < r && *l < pivot) ++l;
        while (l < r && *(r - 1) > pivot) --r;
        if (l < r) { --r; std::swap(*l, *r); ++l; }
    }
    std::swap(*l, *(last - 1));
    return l;
}

void block_qsort_rec(uint32_t* first, uint32_t* last) {
    while (last - first > 32) {
        uint32_t* p = branchless_partition(first, last);
        if (p - first < last - (p + 1)) {
            block_qsort_rec(first, p);
            first = p + 1;
        } else {
            block_qsort_rec(p + 1, last);
            last = p;
        }
    }
    kdr::detail::insertion_sort(first, last);
}

void block_qsort(std::vector<uint32_t>& a) {
    if (a.size() > 1) block_qsort_rec(a.data(), a.data() + a.size());
}

// 2. LSD Radix 11-11-10 bit
void lsd_11bit(std::vector<uint32_t>& a) {
    size_t n = a.size();
    std::vector<uint32_t> tmp(n);
    size_t c0[2048] = {0}, c1[2048] = {0}, c2[1024] = {0};
    for (size_t i = 0; i < n; ++i) {
        uint32_t v = a[i];
        c0[v & 0x7FF]++; c1[(v >> 11) & 0x7FF]++; c2[v >> 22]++;
    }
    size_t s0 = 0, s1 = 0, s2 = 0;
    for (int i = 0; i < 2048; ++i) { size_t t = c0[i]; c0[i] = s0; s0 += t; }
    for (int i = 0; i < 2048; ++i) { size_t t = c1[i]; c1[i] = s1; s1 += t; }
    for (int i = 0; i < 1024; ++i) { size_t t = c2[i]; c2[i] = s2; s2 += t; }
    for (size_t i = 0; i < n; ++i) tmp[c0[a[i] & 0x7FF]++] = a[i];
    for (size_t i = 0; i < n; ++i) a[c1[(tmp[i] >> 11) & 0x7FF]++] = tmp[i];
    for (size_t i = 0; i < n; ++i) tmp[c2[a[i] >> 22]++] = a[i];
    a.swap(tmp);
}

template <typename T, typename Fn>
double bench_one(const std::vector<T>& orig, Fn&& fn, int reps = 2) {
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        std::vector<T> d = orig;
        auto t0 = Clock::now();
        fn(d);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

int main() {
    std::mt19937_64 rng(777);
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "JSON_START\n";
    std::cout << "[\n";

    bool first_entry = true;
    auto emit_result = [&](const std::string& type, const std::string& dist, size_t n, 
                           const std::string& algo, double ms) {
        if (!first_entry) std::cout << ",\n";
        first_entry = false;
        double rate_m = (static_cast<double>(n) / 1e6) / (ms / 1000.0);
        std::cout << "  {\"type\": \"" << type << "\", \"dist\": \"" << dist 
                  << "\", \"n\": " << n << ", \"algo\": \"" << algo 
                  << "\", \"ms\": " << ms << ", \"rate_m_per_sec\": " << rate_m << "}";
        std::cout.flush();
    };

    // Sizes: 20M, 50M, 100M
    for (size_t N : {20000000ULL, 50000000ULL, 100000000ULL}) {
        std::cerr << "Running N=" << N << "...\n";

        // Distributions:
        // 1. Uniform
        {
            std::vector<uint32_t> data(N);
            for (auto& x : data) x = static_cast<uint32_t>(rng());

            double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); }, 1);
            emit_result("uint32", "Uniform Random", N, "std::sort", ms_std);

            double ms_bq = bench_one(data, block_qsort, 1);
            emit_result("uint32", "Uniform Random", N, "Block-Quicksort", ms_bq);

            double ms_lsd = bench_one(data, lsd_11bit, 1);
            emit_result("uint32", "Uniform Random", N, "LSD-11bit", ms_lsd);

            double ms_k1 = bench_one(data, [](auto& v){ kdr::sort(v, 1); }, 1);
            emit_result("uint32", "Uniform Random", N, "KDR (1T)", ms_k1);

            double ms_k6 = bench_one(data, [](auto& v){ kdr::sort(v, 6); }, 1);
            emit_result("uint32", "Uniform Random", N, "KDR (6T)", ms_k6);

            double ms_k12 = bench_one(data, [](auto& v){ kdr::sort(v, 12); }, 1);
            emit_result("uint32", "Uniform Random", N, "KDR (12T)", ms_k12);
        }

        // 2. Already Sorted
        {
            std::vector<uint32_t> data(N);
            for (size_t i = 0; i < N; ++i) data[i] = static_cast<uint32_t>(i);

            double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); }, 1);
            emit_result("uint32", "Already Sorted", N, "std::sort", ms_std);

            double ms_bq = bench_one(data, block_qsort, 1);
            emit_result("uint32", "Already Sorted", N, "Block-Quicksort", ms_bq);

            double ms_k1 = bench_one(data, [](auto& v){ kdr::sort(v, 1); }, 1);
            emit_result("uint32", "Already Sorted", N, "KDR (1T)", ms_k1);

            double ms_k12 = bench_one(data, [](auto& v){ kdr::sort(v, 12); }, 1);
            emit_result("uint32", "Already Sorted", N, "KDR (12T)", ms_k12);
        }

        // 3. Reverse Sorted
        {
            std::vector<uint32_t> data(N);
            for (size_t i = 0; i < N; ++i) data[i] = static_cast<uint32_t>(N - i);

            double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); }, 1);
            emit_result("uint32", "Reverse Sorted", N, "std::sort", ms_std);

            double ms_bq = bench_one(data, block_qsort, 1);
            emit_result("uint32", "Reverse Sorted", N, "Block-Quicksort", ms_bq);

            double ms_k1 = bench_one(data, [](auto& v){ kdr::sort(v, 1); }, 1);
            emit_result("uint32", "Reverse Sorted", N, "KDR (1T)", ms_k1);

            double ms_k12 = bench_one(data, [](auto& v){ kdr::sort(v, 12); }, 1);
            emit_result("uint32", "Reverse Sorted", N, "KDR (12T)", ms_k12);
        }

        // 4. Low Cardinality (16 unique values)
        {
            std::vector<uint32_t> data(N);
            for (auto& x : data) x = static_cast<uint32_t>(rng() % 16);

            double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); }, 1);
            emit_result("uint32", "16 Unique Values", N, "std::sort", ms_std);

            double ms_bq = bench_one(data, block_qsort, 1);
            emit_result("uint32", "16 Unique Values", N, "Block-Quicksort", ms_bq);

            double ms_k1 = bench_one(data, [](auto& v){ kdr::sort(v, 1); }, 1);
            emit_result("uint32", "16 Unique Values", N, "KDR (1T)", ms_k1);

            double ms_k12 = bench_one(data, [](auto& v){ kdr::sort(v, 12); }, 1);
            emit_result("uint32", "16 Unique Values", N, "KDR (12T)", ms_k12);
        }

        // 5. Gaussian / Normal Distribution
        {
            std::vector<uint32_t> data(N);
            std::normal_distribution<double> norm(2147483648.0, 268435456.0);
            for (auto& x : data) {
                double val = norm(rng);
                if (val < 0.0) val = 0.0;
                if (val > 4294967295.0) val = 4294967295.0;
                x = static_cast<uint32_t>(val);
            }

            double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); }, 1);
            emit_result("uint32", "Gaussian Normal", N, "std::sort", ms_std);

            double ms_bq = bench_one(data, block_qsort, 1);
            emit_result("uint32", "Gaussian Normal", N, "Block-Quicksort", ms_bq);

            double ms_lsd = bench_one(data, lsd_11bit, 1);
            emit_result("uint32", "Gaussian Normal", N, "LSD-11bit", ms_lsd);

            double ms_k1 = bench_one(data, [](auto& v){ kdr::sort(v, 1); }, 1);
            emit_result("uint32", "Gaussian Normal", N, "KDR (1T)", ms_k1);

            double ms_k12 = bench_one(data, [](auto& v){ kdr::sort(v, 12); }, 1);
            emit_result("uint32", "Gaussian Normal", N, "KDR (12T)", ms_k12);
        }

        // 6. Sawtooth (Ramps of length 10,000)
        {
            std::vector<uint32_t> data(N);
            for (size_t i = 0; i < N; ++i) data[i] = static_cast<uint32_t>(i % 10000);

            double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); }, 1);
            emit_result("uint32", "Sawtooth Ramps", N, "std::sort", ms_std);

            double ms_bq = bench_one(data, block_qsort, 1);
            emit_result("uint32", "Sawtooth Ramps", N, "Block-Quicksort", ms_bq);

            double ms_k1 = bench_one(data, [](auto& v){ kdr::sort(v, 1); }, 1);
            emit_result("uint32", "Sawtooth Ramps", N, "KDR (1T)", ms_k1);

            double ms_k12 = bench_one(data, [](auto& v){ kdr::sort(v, 12); }, 1);
            emit_result("uint32", "Sawtooth Ramps", N, "KDR (12T)", ms_k12);
        }
    }

    // 64-bit uint64_t at large scale (20M and 50M)
    for (size_t N : {20000000ULL, 50000000ULL}) {
        std::cerr << "Running uint64 N=" << N << "...\n";
        std::vector<uint64_t> data(N);
        for (auto& x : data) x = rng();

        double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); }, 1);
        emit_result("uint64", "Uniform Random", N, "std::sort", ms_std);

        double ms_k1 = bench_one(data, [](auto& v){ kdr::sort(v, 1); }, 1);
        emit_result("uint64", "Uniform Random", N, "KDR (1T)", ms_k1);

        double ms_k6 = bench_one(data, [](auto& v){ kdr::sort(v, 6); }, 1);
        emit_result("uint64", "Uniform Random", N, "KDR (6T)", ms_k6);

        double ms_k12 = bench_one(data, [](auto& v){ kdr::sort(v, 12); }, 1);
        emit_result("uint64", "Uniform Random", N, "KDR (12T)", ms_k12);
    }

    std::cout << "\n]\n";
    std::cout << "JSON_END\n";
    return 0;
}
