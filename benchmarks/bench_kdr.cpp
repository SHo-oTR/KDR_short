#include <kdr/kdr_sort.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

// 1. Heap Sort
void heap_sort(std::vector<uint32_t>& a) {
    std::make_heap(a.begin(), a.end());
    std::sort_heap(a.begin(), a.end());
}

// 2. Shellsort (Ciura extended gap sequence)
void shell_sort(std::vector<uint32_t>& a) {
    static const size_t gaps[] = {
        701, 301, 132, 57, 23, 10, 4, 1
    };
    size_t n = a.size();
    std::vector<size_t> seq;
    for (size_t g = 1750; g < n; g = static_cast<size_t>(g * 2.25)) seq.push_back(g);
    std::reverse(seq.begin(), seq.end());
    for (size_t g : gaps) if (g < n) seq.push_back(g);
    for (size_t gap : seq) {
        for (size_t i = gap; i < n; ++i) {
            uint32_t tmp = a[i];
            size_t j = i;
            while (j >= gap && a[j - gap] > tmp) {
                a[j] = a[j - gap];
                j -= gap;
            }
            a[j] = tmp;
        }
    }
}

// 3. Branchless Block-Quicksort (Edelkamp & Weiss / pdqsort core idea)
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
    // Scalar cleanup
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

// 4. LSD Radix 8-bit
void lsd_8bit(std::vector<uint32_t>& a) {
    size_t n = a.size();
    std::vector<uint32_t> tmp(n);
    for (int p = 0; p < 4; ++p) {
        size_t c[256] = {0}; int sh = p * 8;
        for (size_t i = 0; i < n; ++i) c[(a[i] >> sh) & 0xFF]++;
        bool skip = false; for (int b = 0; b < 256; ++b) if (c[b] == n) { skip = true; break; }
        if (skip) continue;
        size_t s = 0; for (int b = 0; b < 256; ++b) { size_t t = c[b]; c[b] = s; s += t; }
        for (size_t i = 0; i < n; ++i) tmp[c[(a[i] >> sh) & 0xFF]++] = a[i];
        a.swap(tmp);
    }
}

// 5. LSD Radix 11-11-10 bit
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

// 6. American Flag Sort (In-Place MSD + Insertion <= 32)
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

// 7. Counting Sort (only for small range [0..K])
void counting_sort_small(std::vector<uint32_t>& a, uint32_t max_val) {
    std::vector<uint32_t> cnt(max_val + 1, 0);
    for (uint32_t x : a) cnt[x]++;
    size_t idx = 0;
    for (uint32_t v = 0; v <= max_val; ++v) {
        uint32_t c = cnt[v];
        while (c--) a[idx++] = v;
    }
}

template <typename Fn>
double measure(const std::vector<uint32_t>& orig, const std::vector<uint32_t>& ref, Fn&& fn, int reps = 2) {
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        std::vector<uint32_t> d = orig;
        auto t0 = Clock::now();
        fn(d);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
        if (d != ref) return -1.0;
    }
    return best;
}

void run_bench(const std::string& title, const std::vector<uint32_t>& data, bool test_counting = false, uint32_t max_v = 0) {
    std::vector<uint32_t> ref = data;
    std::sort(ref.begin(), ref.end());
    std::cout << "\n=== " << title << " (N=" << data.size() << ") ===\n";
    auto show = [&](const std::string& name, auto fn) {
        double ms = measure(data, ref, fn, 2);
        std::cout << std::left << std::setw(28) << name
                  << std::right << std::setw(9) << std::fixed << std::setprecision(2) << ms << " ms\n";
    };
    show("Heap Sort", heap_sort);
    show("Shellsort (Ciura)", shell_sort);
    show("Merge Sort (stable_sort)", [](std::vector<uint32_t>& v){ std::stable_sort(v.begin(), v.end()); });
    show("Introsort (std::sort)", [](std::vector<uint32_t>& v){ std::sort(v.begin(), v.end()); });
    show("Block-Quicksort (branchless)", block_qsort);
    show("American Flag Sort (MSD)", [](std::vector<uint32_t>& v){ american_flag_rec(v.data(), v.size(), 24); });
    show("LSD Radix (8-bit)", lsd_8bit);
    show("LSD Radix (11-11-10 bit)", lsd_11bit);
    if (test_counting) {
        show("Counting Sort (K=" + std::to_string(max_v) + ")", [max_v](std::vector<uint32_t>& v){ counting_sort_small(v, max_v); });
    }
    show("KDR (1T, O(1))", [](std::vector<uint32_t>& v){ kdr::sort(v); });
    show("KDR (2T, O(1))", [](std::vector<uint32_t>& v){ kdr::parallel_sort(v); });
}

int main() {
    const size_t N = 5'000'000;
    std::mt19937_64 rng(42);

    {
        std::vector<uint32_t> d(N);
        for (auto& x : d) x = static_cast<uint32_t>(rng());
        run_bench("Uniform uint32_t", d);
    }
    {
        std::vector<uint32_t> d(N);
        for (auto& x : d) x = static_cast<uint32_t>(rng() % 10000);
        run_bench("Small-Range [0..9999]", d, true, 9999);
    }
    {
        std::vector<uint32_t> d(N);
        for (size_t i = 0; i < N; ++i) d[i] = static_cast<uint32_t>(i);
        run_bench("Already-Sorted", d);
    }
    return 0;
}
