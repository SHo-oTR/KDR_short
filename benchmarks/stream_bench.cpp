#include <kdr/kdr_sort.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

// Branchless Block-Quicksort
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

// LSD Radix 11-11-10
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
double bench_one(const std::vector<T>& orig, Fn&& fn) {
    std::vector<T> d = orig;
    auto t0 = Clock::now();
    fn(d);
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::ofstream g_out;

void record_result(const std::string& type_name, const std::string& dist, size_t n, 
                   const std::string& algo, double ms) {
    double rate = (static_cast<double>(n) / 1e6) / (ms / 1000.0);
    g_out << "{\"type\":\"" << type_name << "\",\"distribution\":\"" << dist 
          << "\",\"n\":" << n << ",\"algo\":\"" << algo 
          << "\",\"ms\":" << ms << ",\"rate_m_per_sec\":" << rate << "}\n";
    g_out.flush();
    std::cout << "[" << type_name << " " << dist << " N=" << n / 1000000 << "M] "
              << std::left << std::setw(20) << algo << ": " 
              << std::right << std::setw(8) << std::fixed << std::setprecision(2) << ms << " ms ("
              << rate << " M/s)\n";
    std::cout.flush();
}

int main(int argc, char** argv) {
    std::string suite = (argc > 1) ? argv[1] : "all";
    g_out.open("stream_results.jsonl", std::ios::app);
    std::mt19937_64 rng(13579);

    if (suite == "scale_u32" || suite == "all") {
        for (size_t N : {20000000ULL, 50000000ULL, 100000000ULL}) {
            std::cout << "\n=== Scale Test N = " << N / 1000000 << "M ===\n";
            std::vector<uint32_t> d(N);
            for (auto& x : d) x = static_cast<uint32_t>(rng());

            double ms_std = bench_one(d, [](auto& v){ std::sort(v.begin(), v.end()); });
            record_result("uint32", "Uniform", N, "std::sort", ms_std);

            double ms_bq = bench_one(d, block_qsort);
            record_result("uint32", "Uniform", N, "Block-Quicksort", ms_bq);

            double ms_lsd = bench_one(d, lsd_11bit);
            record_result("uint32", "Uniform", N, "LSD-11bit", ms_lsd);

            double ms_k1 = bench_one(d, [](auto& v){ kdr::sort(v, 1); });
            record_result("uint32", "Uniform", N, "KDR (1T)", ms_k1);

            double ms_k2 = bench_one(d, [](auto& v){ kdr::sort(v, 2); });
            record_result("uint32", "Uniform", N, "KDR (2T)", ms_k2);

            double ms_k6 = bench_one(d, [](auto& v){ kdr::sort(v, 6); });
            record_result("uint32", "Uniform", N, "KDR (6T)", ms_k6);

            double ms_k12 = bench_one(d, [](auto& v){ kdr::sort(v, 12); });
            record_result("uint32", "Uniform", N, "KDR (12T)", ms_k12);
        }
    }

    if (suite == "distributions" || suite == "all") {
        size_t N = 50000000ULL;
        std::cout << "\n=== Diverse Distributions N = 50M ===\n";

        auto run_dist = [&](const std::string& name, const std::vector<uint32_t>& d) {
            double ms_std = bench_one(d, [](auto& v){ std::sort(v.begin(), v.end()); });
            record_result("uint32", name, N, "std::sort", ms_std);

            double ms_k1 = bench_one(d, [](auto& v){ kdr::sort(v, 1); });
            record_result("uint32", name, N, "KDR (1T)", ms_k1);

            double ms_k6 = bench_one(d, [](auto& v){ kdr::sort(v, 6); });
            record_result("uint32", name, N, "KDR (6T)", ms_k6);

            double ms_k12 = bench_one(d, [](auto& v){ kdr::sort(v, 12); });
            record_result("uint32", name, N, "KDR (12T)", ms_k12);
        };

        // 1. Gaussian (Normal)
        {
            std::vector<uint32_t> d(N);
            std::normal_distribution<double> dist(2147483648.0, 400000000.0);
            for (auto& x : d) x = static_cast<uint32_t>(std::clamp(dist(rng), 0.0, 4294967295.0));
            run_dist("Gaussian", d);
        }

        // 2. Nearly Sorted (99% sorted, 1% local shuffle)
        {
            std::vector<uint32_t> d(N);
            for (size_t i = 0; i < N; ++i) d[i] = static_cast<uint32_t>(i);
            size_t swaps = N / 100;
            for (size_t i = 0; i < swaps; ++i) {
                size_t p = rng() % (N - 10);
                std::swap(d[p], d[p + (rng() % 10)]);
            }
            run_dist("Nearly-Sorted-99%", d);
        }

        // 3. Organ-Pipe
        {
            std::vector<uint32_t> d(N);
            size_t half = N / 2;
            for (size_t i = 0; i < half; ++i) d[i] = static_cast<uint32_t>(i * 2);
            for (size_t i = half; i < N; ++i) d[i] = static_cast<uint32_t>((N - i) * 2);
            run_dist("Organ-Pipe", d);
        }

        // 4. Low Cardinality (4 values)
        {
            std::vector<uint32_t> d(N);
            for (auto& x : d) x = static_cast<uint32_t>((rng() % 4) * 1000000);
            run_dist("Low-Card-4", d);
        }

        // 5. Reverse Sorted
        {
            std::vector<uint32_t> d(N);
            for (size_t i = 0; i < N; ++i) d[i] = static_cast<uint32_t>(N - i);
            run_dist("Reverse-Sorted", d);
        }
    }

    if (suite == "types" || suite == "all") {
        size_t N = 50000000ULL;
        std::cout << "\n=== Diverse Types N = 50M ===\n";

        // int32
        {
            std::vector<int32_t> d(N);
            for (auto& x : d) x = static_cast<int32_t>(rng());
            double ms_std = bench_one(d, [](auto& v){ std::sort(v.begin(), v.end()); });
            record_result("int32", "Uniform", N, "std::sort", ms_std);
            double ms_k1 = bench_one(d, [](auto& v){ kdr::sort(v, 1); });
            record_result("int32", "Uniform", N, "KDR (1T)", ms_k1);
            double ms_k12 = bench_one(d, [](auto& v){ kdr::sort(v, 12); });
            record_result("int32", "Uniform", N, "KDR (12T)", ms_k12);
        }

        // uint64
        {
            std::vector<uint64_t> d(N);
            for (auto& x : d) x = rng();
            double ms_std = bench_one(d, [](auto& v){ std::sort(v.begin(), v.end()); });
            record_result("uint64", "Uniform", N, "std::sort", ms_std);
            double ms_k1 = bench_one(d, [](auto& v){ kdr::sort(v, 1); });
            record_result("uint64", "Uniform", N, "KDR (1T)", ms_k1);
            double ms_k12 = bench_one(d, [](auto& v){ kdr::sort(v, 12); });
            record_result("uint64", "Uniform", N, "KDR (12T)", ms_k12);
        }

        // float
        {
            std::vector<float> d(N);
            std::uniform_real_distribution<float> dist(-1e6f, 1e6f);
            for (auto& x : d) x = dist(rng);
            double ms_std = bench_one(d, [](auto& v){ std::sort(v.begin(), v.end()); });
            record_result("float", "Uniform", N, "std::sort", ms_std);
            double ms_k1 = bench_one(d, [](auto& v){ kdr::sort(v, 1); });
            record_result("float", "Uniform", N, "KDR (1T)", ms_k1);
            double ms_k12 = bench_one(d, [](auto& v){ kdr::sort(v, 12); });
            record_result("float", "Uniform", N, "KDR (12T)", ms_k12);
        }

        // double
        {
            std::vector<double> d(N);
            std::uniform_real_distribution<double> dist(-1e12, 1e12);
            for (auto& x : d) x = dist(rng);
            double ms_std = bench_one(d, [](auto& v){ std::sort(v.begin(), v.end()); });
            record_result("double", "Uniform", N, "std::sort", ms_std);
            double ms_k1 = bench_one(d, [](auto& v){ kdr::sort(v, 1); });
            record_result("double", "Uniform", N, "KDR (1T)", ms_k1);
            double ms_k12 = bench_one(d, [](auto& v){ kdr::sort(v, 12); });
            record_result("double", "Uniform", N, "KDR (12T)", ms_k12);
        }
    }
    return 0;
}
