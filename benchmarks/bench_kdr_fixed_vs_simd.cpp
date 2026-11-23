#include <kdr/kdr_sort.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

template <typename T, typename Fn>
double bench(const std::vector<T>& orig, Fn&& fn, int reps = 2) {
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
    std::mt19937_64 rng(13579);
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "JSON_START\n[\n";
    bool first_entry = true;

    auto emit = [&](const std::string& cat, const std::string& dist, size_t n,
                     const std::string& algo, double ms) {
        if (!first_entry) std::cout << ",\n";
        first_entry = false;
        double rate_m = (static_cast<double>(n) / 1e6) / (ms / 1000.0);
        std::cout << "  {\"category\": \"" << cat << "\", \"dist\": \"" << dist 
                  << "\", \"n\": " << n << ", \"algo\": \"" << algo 
                  << "\", \"ms\": " << ms << ", \"rate_m_per_sec\": " << rate_m << "}";
        std::cout.flush();
    };

    // 1. Uniform uint32_t (1M, 5M, 10M, 20M)
    for (size_t N : {1000000ULL, 5000000ULL, 10000000ULL, 20000000ULL}) {
        std::vector<uint32_t> data(N);
        for (auto& x : data) x = static_cast<uint32_t>(rng());

        double ms_std = bench(data, [](auto& v){ std::sort(v.begin(), v.end()); });
        emit("uint32", "Uniform", N, "std::sort", ms_std);

        double ms_k1 = bench(data, [](auto& v){ kdr::sort(v, 1); });
        emit("uint32", "Uniform", N, "KDR Fixed (1T)", ms_k1);

        double ms_k2 = bench(data, [](auto& v){ kdr::sort(v, 2); });
        emit("uint32", "Uniform", N, "KDR Fixed (2T ThreadPool)", ms_k2);

        double ms_k6 = bench(data, [](auto& v){ kdr::sort(v, 6); });
        emit("uint32", "Uniform", N, "KDR Fixed (6T ThreadPool)", ms_k6);

        double ms_k12 = bench(data, [](auto& v){ kdr::sort(v, 12); });
        emit("uint32", "Uniform", N, "KDR Fixed (12T ThreadPool)", ms_k12);
    }

    // 2. Small Range [0..9999] uint32_t (5M) — Testing NEW Counting Sort Fast-Path!
    {
        size_t N = 5000000;
        std::vector<uint32_t> data(N);
        for (auto& x : data) x = static_cast<uint32_t>(rng() % 10000);

        double ms_std = bench(data, [](auto& v){ std::sort(v.begin(), v.end()); });
        emit("uint32", "Small-Range [0..9999]", N, "std::sort", ms_std);

        double ms_k = bench(data, [](auto& v){ kdr::sort(v); });
        emit("uint32", "Small-Range [0..9999]", N, "KDR Fixed Fast-Path", ms_k);
    }

    // 3. NEW 16-bit uint16_t (5M)
    {
        size_t N = 5000000;
        std::vector<uint16_t> data(N);
        for (auto& x : data) x = static_cast<uint16_t>(rng());

        double ms_std = bench(data, [](auto& v){ std::sort(v.begin(), v.end()); });
        emit("uint16", "Uniform Random", N, "std::sort", ms_std);

        double ms_k = bench(data, [](auto& v){ kdr::sort(v); });
        emit("uint16", "Uniform Random", N, "KDR Fixed 16-bit Engine", ms_k);
    }

    // 4. NEW 8-bit uint8_t (10M)
    {
        size_t N = 10000000;
        std::vector<uint8_t> data(N);
        for (auto& x : data) x = static_cast<uint8_t>(rng());

        double ms_std = bench(data, [](auto& v){ std::sort(v.begin(), v.end()); });
        emit("uint8", "Uniform Random", N, "std::sort", ms_std);

        double ms_k = bench(data, [](auto& v){ kdr::sort(v); });
        emit("uint8", "Uniform Random", N, "KDR Fixed 8-bit Engine", ms_k);
    }

    // 5. Float (5M) & Double (5M) with strict-aliasing safe pipeline
    {
        size_t N = 5000000;
        std::vector<float> data_f(N);
        std::uniform_real_distribution<float> df(-1e6f, 1e6f);
        for (auto& x : data_f) x = df(rng);

        double ms_std_f = bench(data_f, [](auto& v){ std::sort(v.begin(), v.end()); });
        emit("float", "Uniform Random", N, "std::sort", ms_std_f);

        double ms_k_f = bench(data_f, [](auto& v){ kdr::parallel_sort(v); });
        emit("float", "Uniform Random", N, "KDR Fixed (ThreadPool)", ms_k_f);

        std::vector<double> data_d(N);
        std::uniform_real_distribution<double> dd(-1e12, 1e12);
        for (auto& x : data_d) x = dd(rng);

        double ms_std_d = bench(data_d, [](auto& v){ std::sort(v.begin(), v.end()); });
        emit("double", "Uniform Random", N, "std::sort", ms_std_d);

        double ms_k_d = bench(data_d, [](auto& v){ kdr::parallel_sort(v); });
        emit("double", "Uniform Random", N, "KDR Fixed (ThreadPool)", ms_k_d);
    }

    // 6. 64-bit uint64_t (10M)
    {
        size_t N = 10000000;
        std::vector<uint64_t> data(N);
        for (auto& x : data) x = rng();

        double ms_std = bench(data, [](auto& v){ std::sort(v.begin(), v.end()); });
        emit("uint64", "Uniform Random", N, "std::sort", ms_std);

        double ms_k1 = bench(data, [](auto& v){ kdr::sort(v, 1); });
        emit("uint64", "Uniform Random", N, "KDR Fixed (1T)", ms_k1);

        double ms_k12 = bench(data, [](auto& v){ kdr::parallel_sort(v); });
        emit("uint64", "Uniform Random", N, "KDR Fixed (12T ThreadPool)", ms_k12);
    }

    // 7. Already-Sorted (10M uint32)
    {
        size_t N = 10000000;
        std::vector<uint32_t> data(N);
        for (size_t i = 0; i < N; ++i) data[i] = static_cast<uint32_t>(i);

        double ms_std = bench(data, [](auto& v){ std::sort(v.begin(), v.end()); });
        emit("uint32", "Already-Sorted", N, "std::sort", ms_std);

        double ms_k = bench(data, [](auto& v){ kdr::sort(v); });
        emit("uint32", "Already-Sorted", N, "KDR Fixed Fast-Path", ms_k);
    }

    // 8. std::string (1M with 64-bit StrRec index & LCP skipping)
    {
        size_t N = 1000000;
        std::vector<std::string> urls(N);
        for (size_t i = 0; i < N; ++i) {
            urls[i] = "https://kdr.fixed.bench/v13/" + std::to_string(rng() % 5000000);
        }

        double ms_std = bench(urls, [](auto& v){ std::sort(v.begin(), v.end()); });
        emit("string", "Shared-Prefix URLs", N, "std::sort", ms_std);

        double ms_k = bench(urls, [](auto& v){ kdr::parallel_sort(v); });
        emit("string", "Shared-Prefix URLs", N, "KDR Fixed (ThreadPool)", ms_k);
    }

    std::cout << "\n]\nJSON_END\n";
    return 0;
}
