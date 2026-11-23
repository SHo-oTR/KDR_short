#include <kdr/kdr_sort.hpp>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

template <typename T>
void verify(const std::string& name, const std::vector<T>& orig) {
    std::vector<T> ref = orig;
    std::sort(ref.begin(), ref.end());

    std::vector<T> a = orig;
    kdr::sort(a);
    assert(a == ref);

    std::vector<T> b = orig;
    kdr::parallel_sort(b);
    assert(b == ref);

    std::cout << "[PASS] " << name << " (N=" << orig.size() << ")\n";
}

int main() {
    std::mt19937_64 rng(12345);

    // Edge cases
    verify("empty uint32", std::vector<uint32_t>{});
    verify("single uint32", std::vector<uint32_t>{42});
    verify("all-equal uint64", std::vector<uint64_t>(100000, 999999ULL));
    verify("sorted int32", [](){ std::vector<int32_t> v(100000); for(int i=0;i<100000;++i) v[i]=i-50000; return v; }());
    verify("reverse int32", [](){ std::vector<int32_t> v(100000); for(int i=0;i<100000;++i) v[i]=50000-i; return v; }());

    // Random numeric types (both < SCRATCH_CAP and > SCRATCH_CAP)
    {
        std::vector<uint32_t> v(600000);
        for (auto& x : v) x = static_cast<uint32_t>(rng());
        verify("random uint32_t", v);
    }
    {
        std::vector<int32_t> v(600000);
        for (auto& x : v) x = static_cast<int32_t>(rng());
        verify("random int32_t", v);
    }
    {
        std::vector<int64_t> v(600000);
        for (auto& x : v) x = static_cast<int64_t>(rng());
        verify("random int64_t", v);
    }
    {
        std::vector<float> v(600000);
        std::uniform_real_distribution<float> d(-1e6f, 1e6f);
        for (auto& x : v) x = d(rng);
        v[0] = -0.0f; v[1] = +0.0f;
        v[2] = -std::numeric_limits<float>::infinity();
        v[3] = std::numeric_limits<float>::infinity();
        verify("random float (+inf/-inf/-0/+0)", v);
    }
    {
        std::vector<double> v(600000);
        std::uniform_real_distribution<double> d(-1e12, 1e12);
        for (auto& x : v) x = d(rng);
        verify("random double", v);
    }
    {
        std::vector<std::string> v(200000);
        for (size_t i = 0; i < v.size(); ++i) {
            if (i % 1000 == 0) { v[i] = ""; continue; }
            std::string s = "https://kdr.sort/v6/";
            int len = static_cast<int>(rng() % 25);
            for (int k = 0; k < len; ++k) s.push_back(static_cast<char>('a' + (rng() % 26)));
            v[i] = std::move(s);
        }
        verify("random std::string (with empty & shared prefix)", v);
    }
    std::cout << "ALL KDR TESTS PASSED!\n";
    return 0;
}
