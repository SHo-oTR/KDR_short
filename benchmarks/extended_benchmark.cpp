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

// 1. Heap Sort
template <typename T>
void heap_sort(std::vector<T>& a) {
    std::make_heap(a.begin(), a.end());
    std::sort_heap(a.begin(), a.end());
}

// 2. Shellsort (Ciura)
template <typename T>
void shell_sort(std::vector<T>& a) {
    static const size_t gaps[] = { 701, 301, 132, 57, 23, 10, 4, 1 };
    size_t n = a.size();
    std::vector<size_t> seq;
    for (size_t g = 1750; g < n; g = static_cast<size_t>(g * 2.25)) seq.push_back(g);
    std::reverse(seq.begin(), seq.end());
    for (size_t g : gaps) if (g < n) seq.push_back(g);
    for (size_t gap : seq) {
        for (size_t i = gap; i < n; ++i) {
            T tmp = a[i];
            size_t j = i;
            while (j >= gap && a[j - gap] > tmp) {
                a[j] = a[j - gap];
                j -= gap;
            }
            a[j] = tmp;
        }
    }
}

// 3. Branchless Block-Quicksort
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

// 4. LSD Radix 11-11-10 bit
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

// 5. American Flag Sort
void american_flag_rec(uint32_t* first, size_t n, int shift) {
    if (n <= 32) { kdr::detail::insertion_sort(first, first + n); return; }
    size_t count[256] = {0};
    for (size_t i = 0; i < n; ++i) count[(first[i] >> shift) & 0xFF]++;
    size_t start[256], next[256], sum = 0; int non_empty = 0;
    for (int b = 0; b < 256; ++b) {
        start[b] = sum; next[b] = sum; sum += count[b];
        if (count[b] > 0) non_empty++;
    }
    if (non_empty > 1) {
        for (int b = 0; b < 256; ++b) {
            size_t end_b = start[b] + count[b];
            while (next[b] < end_b) {
                uint32_t val = first[next[b]];
                uint8_t dest = (val >> shift) & 0xFF;
                if (dest == b) next[b]++;
                else {
                    do {
                        std::swap(val, first[next[dest]++]);
                        dest = (val >> shift) & 0xFF;
                    } while (dest != b);
                    first[next[b]++] = val;
                }
            }
        }
    }
    if (shift == 0) return;
    for (int b = 0; b < 256; ++b)
        if (count[b] > 1) american_flag_rec(first + start[b], count[b], shift - 8);
}

template <typename T, typename Fn>
double bench_one(const std::vector<T>& orig, Fn&& fn, int reps = 3) {
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
    std::mt19937_64 rng(9999);
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "JSON_START\n";
    std::cout << "[\n";

    bool first_entry = true;
    auto emit_result = [&](const std::string& cat, const std::string& dist, size_t n, 
                           const std::string& algo, double ms, double m_elem_per_sec) {
        if (!first_entry) std::cout << ",\n";
        first_entry = false;
        std::cout << "  {\"category\": \"" << cat << "\", \"distribution\": \"" << dist 
                  << "\", \"n\": " << n << ", \"algo\": \"" << algo 
                  << "\", \"ms\": " << ms << ", \"rate_m_per_sec\": " << m_elem_per_sec << "}";
    };

    // 1. Comprehensive uint32_t Scale Test (1M, 5M, 10M)
    for (size_t N : {1000000ULL, 5000000ULL, 10000000ULL}) {
        std::vector<uint32_t> uniform(N);
        for (auto& x : uniform) x = static_cast<uint32_t>(rng());

        std::vector<uint32_t> sorted(N);
        for (size_t i = 0; i < N; ++i) sorted[i] = static_cast<uint32_t>(i);

        std::vector<uint32_t> reverse(N);
        for (size_t i = 0; i < N; ++i) reverse[i] = static_cast<uint32_t>(N - i);

        std::vector<uint32_t> low_card(N);
        for (auto& x : low_card) x = static_cast<uint32_t>(rng() % 16);

        auto run_all_u32 = [&](const std::string& dist, const std::vector<uint32_t>& data) {
            double rate = static_cast<double>(N) / 1e6;
            
            double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); });
            emit_result("uint32", dist, N, "std::sort", ms_std, rate / (ms_std / 1000.0));

            double ms_stable = bench_one(data, [](auto& v){ std::stable_sort(v.begin(), v.end()); });
            emit_result("uint32", dist, N, "std::stable_sort", ms_stable, rate / (ms_stable / 1000.0));

            double ms_bq = bench_one(data, block_qsort);
            emit_result("uint32", dist, N, "Block-Quicksort", ms_bq, rate / (ms_bq / 1000.0));

            if (dist == "Uniform") {
                double ms_lsd = bench_one(data, lsd_11bit);
                emit_result("uint32", dist, N, "LSD-11bit", ms_lsd, rate / (ms_lsd / 1000.0));

                double ms_flag = bench_one(data, [](auto& v){ american_flag_rec(v.data(), v.size(), 24); });
                emit_result("uint32", dist, N, "American Flag Sort", ms_flag, rate / (ms_flag / 1000.0));
            }

            for (int t : {1, 2, 4, 6, 12}) {
                std::string name = "KDR (" + std::to_string(t) + "T)";
                double ms_kdr = bench_one(data, [t](auto& v){ kdr::sort(v, t); });
                emit_result("uint32", dist, N, name, ms_kdr, rate / (ms_kdr / 1000.0));
            }
        };

        run_all_u32("Uniform", uniform);
        run_all_u32("Already-Sorted", sorted);
        run_all_u32("Reverse-Sorted", reverse);
        run_all_u32("Low-Cardinality-16", low_card);
    }

    // 2. Different Types Test (N=5M)
    {
        size_t N = 5000000;
        double rate = static_cast<double>(N) / 1e6;

        // int32_t
        {
            std::vector<int32_t> data(N);
            for (auto& x : data) x = static_cast<int32_t>(rng());
            double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); });
            emit_result("int32", "Uniform", N, "std::sort", ms_std, rate / (ms_std / 1000.0));
            double ms_k1 = bench_one(data, [](auto& v){ kdr::sort(v, 1); });
            emit_result("int32", "Uniform", N, "KDR (1T)", ms_k1, rate / (ms_k1 / 1000.0));
            double ms_k6 = bench_one(data, [](auto& v){ kdr::sort(v, 6); });
            emit_result("int32", "Uniform", N, "KDR (6T)", ms_k6, rate / (ms_k6 / 1000.0));
        }

        // uint64_t
        {
            std::vector<uint64_t> data(N);
            for (auto& x : data) x = rng();
            double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); });
            emit_result("uint64", "Uniform", N, "std::sort", ms_std, rate / (ms_std / 1000.0));
            double ms_k1 = bench_one(data, [](auto& v){ kdr::sort(v, 1); });
            emit_result("uint64", "Uniform", N, "KDR (1T)", ms_k1, rate / (ms_k1 / 1000.0));
            double ms_k6 = bench_one(data, [](auto& v){ kdr::sort(v, 6); });
            emit_result("uint64", "Uniform", N, "KDR (6T)", ms_k6, rate / (ms_k6 / 1000.0));
        }

        // float
        {
            std::vector<float> data(N);
            std::uniform_real_distribution<float> d(-1e6f, 1e6f);
            for (auto& x : data) x = d(rng);
            double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); });
            emit_result("float", "Uniform", N, "std::sort", ms_std, rate / (ms_std / 1000.0));
            double ms_k1 = bench_one(data, [](auto& v){ kdr::sort(v, 1); });
            emit_result("float", "Uniform", N, "KDR (1T)", ms_k1, rate / (ms_k1 / 1000.0));
            double ms_k6 = bench_one(data, [](auto& v){ kdr::sort(v, 6); });
            emit_result("float", "Uniform", N, "KDR (6T)", ms_k6, rate / (ms_k6 / 1000.0));
        }

        // double
        {
            std::vector<double> data(N);
            std::uniform_real_distribution<double> d(-1e12, 1e12);
            for (auto& x : data) x = d(rng);
            double ms_std = bench_one(data, [](auto& v){ std::sort(v.begin(), v.end()); });
            emit_result("double", "Uniform", N, "std::sort", ms_std, rate / (ms_std / 1000.0));
            double ms_k1 = bench_one(data, [](auto& v){ kdr::sort(v, 1); });
            emit_result("double", "Uniform", N, "KDR (1T)", ms_k1, rate / (ms_k1 / 1000.0));
            double ms_k6 = bench_one(data, [](auto& v){ kdr::sort(v, 6); });
            emit_result("double", "Uniform", N, "KDR (6T)", ms_k6, rate / (ms_k6 / 1000.0));
        }
    }

    // 3. String Engine Tests (N=1M URLs with 24-byte common prefix)
    {
        size_t N = 1000000;
        double rate = static_cast<double>(N) / 1e6;
        std::vector<std::string> urls(N);
        for (size_t i = 0; i < N; ++i) {
            urls[i] = "https://kdr.bench/v13/" + std::to_string(rng() % 5000000);
        }
        double ms_std = bench_one(urls, [](auto& v){ std::sort(v.begin(), v.end()); }, 2);
        emit_result("string", "Shared-Prefix URLs", N, "std::sort", ms_std, rate / (ms_std / 1000.0));

        double ms_k1 = bench_one(urls, [](auto& v){ kdr::sort(v, 1); }, 2);
        emit_result("string", "Shared-Prefix URLs", N, "KDR (1T)", ms_k1, rate / (ms_k1 / 1000.0));

        double ms_k2 = bench_one(urls, [](auto& v){ kdr::sort(v, 2); }, 2);
        emit_result("string", "Shared-Prefix URLs", N, "KDR (2T)", ms_k2, rate / (ms_k2 / 1000.0));

        double ms_k6 = bench_one(urls, [](auto& v){ kdr::sort(v, 6); }, 2);
        emit_result("string", "Shared-Prefix URLs", N, "KDR (6T)", ms_k6, rate / (ms_k6 / 1000.0));
    }

    std::cout << "\n]\n";
    std::cout << "JSON_END\n";
    return 0;
}
