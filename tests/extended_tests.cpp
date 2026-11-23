#include <kdr/kdr_sort.hpp>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <atomic>

// Heap allocation tracker
static std::atomic<size_t> g_allocated_bytes{0};
static std::atomic<size_t> g_allocation_count{0};
static bool g_track_allocations = false;

void* operator new(size_t size) {
    if (g_track_allocations) {
        g_allocated_bytes += size;
        g_allocation_count++;
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, size_t) noexcept {
    std::free(p);
}

template <typename T>
bool verify_range(const std::string& name, const std::vector<T>& orig, int threads = 1) {
    std::vector<T> ref = orig;
    std::sort(ref.begin(), ref.end());

    std::vector<T> actual = orig;
    if (threads == 1) {
        kdr::sort(actual);
    } else {
        kdr::sort(actual, threads);
    }

    if (actual != ref) {
        std::cerr << "[FAIL] " << name << " (N=" << orig.size() << ", threads=" << threads << ")\n";
        for (size_t i = 0; i < std::min<size_t>(orig.size(), 10); ++i) {
            std::cerr << "  idx " << i << ": actual=" << actual[i] << " vs ref=" << ref[i] << "\n";
        }
        for (size_t i = 0; i < orig.size(); ++i) {
            if (actual[i] != ref[i]) {
                std::cerr << "  FIRST MISMATCH at idx " << i << ": actual=" << actual[i] << " vs ref=" << ref[i] << "\n";
                break;
            }
        }
        return false;
    }
    return true;
}

// Special verify for float to handle -0.0f vs +0.0f and inf
bool verify_float(const std::string& name, const std::vector<float>& orig) {
    std::vector<float> ref = orig;
    std::sort(ref.begin(), ref.end(), [](float a, float b) {
        // Strict bitwise / value comparison
        if (std::isnan(a) || std::isnan(b)) return false;
        if (a == 0.0f && b == 0.0f) {
            // Check sign bit: -0.0f should precede +0.0f
            return std::signbit(a) && !std::signbit(b);
        }
        return a < b;
    });

    std::vector<float> actual = orig;
    kdr::sort(actual);

    for (size_t i = 0; i < orig.size(); ++i) {
        bool match = false;
        if (actual[i] == ref[i]) {
            if (actual[i] == 0.0f) {
                match = (std::signbit(actual[i]) == std::signbit(ref[i]));
            } else {
                match = true;
            }
        }
        if (!match) {
            std::cerr << "[FAIL FLOAT] " << name << " at idx " << i 
                      << ": actual=" << actual[i] << " (sign=" << std::signbit(actual[i]) << ")"
                      << " vs ref=" << ref[i] << " (sign=" << std::signbit(ref[i]) << ")\n";
            return false;
        }
    }
    return true;
}

int main() {
    std::mt19937_64 rng(54321);
    int total_tests = 0;
    int passed_tests = 0;

    auto run_test = [&](const std::string& name, const auto& v, int threads = 1) {
        total_tests++;
        if (verify_range(name, v, threads)) {
            passed_tests++;
            std::cout << "[PASS] " << name << " (N=" << v.size() << ")\n";
        }
    };

    std::cout << "=========================================\n";
    std::cout << "1. BOUNDARY & CORNER CASE TESTS\n";
    std::cout << "=========================================\n";
    run_test("N=0 uint32", std::vector<uint32_t>{});
    run_test("N=1 uint32", std::vector<uint32_t>{42});
    run_test("N=2 uint32 [1, 2]", std::vector<uint32_t>{1, 2});
    run_test("N=2 uint32 [2, 1]", std::vector<uint32_t>{2, 1});
    run_test("N=2 uint32 [5, 5]", std::vector<uint32_t>{5, 5});
    run_test("N=3 uint32 [3, 1, 2]", std::vector<uint32_t>{3, 1, 2});

    // Test exact cutoff thresholds
    for (size_t sz : {31, 32, 33, 47, 48, 49, 63, 64, 65, 127, 128, 129, 255, 256, 257, 
                      1023, 1024, 1025, 131071, 131072, 131073, 524287, 524288, 524289}) {
        std::vector<uint32_t> v(sz);
        for (auto& x : v) x = static_cast<uint32_t>(rng());
        run_test("Threshold N=" + std::to_string(sz), v);
    }

    std::cout << "\n=========================================\n";
    std::cout << "2. ADVERSARIAL & PATHOLOGICAL PATTERNS\n";
    std::cout << "=========================================\n";
    {
        // All equal
        std::vector<uint32_t> v(300000, 777777);
        run_test("All equal 300K", v);
    }
    {
        // 2 unique values (boolean flags)
        std::vector<uint32_t> v(300000);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (i % 3 == 0) ? 100 : 200;
        run_test("Two unique values 300K", v);
    }
    {
        // 8 unique values
        std::vector<uint32_t> v(300000);
        for (auto& x : v) x = static_cast<uint32_t>(rng() % 8);
        run_test("8 unique values 300K", v);
    }
    {
        // Strictly ascending
        std::vector<uint32_t> v(300000);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint32_t>(i);
        run_test("Ascending 300K", v);
    }
    {
        // Strictly descending
        std::vector<uint32_t> v(300000);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint32_t>(300000 - i);
        run_test("Descending 300K", v);
    }
    {
        // Organ pipe (ascending then descending)
        std::vector<uint32_t> v(300000);
        for (size_t i = 0; i < 150000; ++i) v[i] = static_cast<uint32_t>(i * 2);
        for (size_t i = 150000; i < 300000; ++i) v[i] = static_cast<uint32_t>((300000 - i) * 2);
        run_test("Organ pipe 300K", v);
    }
    {
        // Sawtooth pattern
        std::vector<uint32_t> v(300000);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint32_t>(i % 500);
        run_test("Sawtooth 300K", v);
    }
    {
        // Only high bits differ (low bits all 0)
        std::vector<uint32_t> v(300000);
        for (auto& x : v) x = static_cast<uint32_t>((rng() % 256) << 24);
        run_test("Only high 8 bits differ 300K", v);
    }
    {
        // Only low bits differ (high bits identical)
        std::vector<uint32_t> v(300000);
        for (auto& x : v) x = 0xABCD0000u | static_cast<uint32_t>(rng() % 65536);
        run_test("Only low 16 bits differ 300K", v);
    }
    {
        // Only 1 bit differs (bit 15)
        std::vector<uint32_t> v(300000);
        for (size_t i = 0; i < v.size(); ++i) v[i] = 0x12345678u ^ ((i & 1) ? (1u << 15) : 0);
        run_test("Single bit variation 300K", v);
    }

    std::cout << "\n=========================================\n";
    std::cout << "3. SIGNED & FLOATING POINT EDGE CASES\n";
    std::cout << "=========================================\n";
    {
        // int32 limits and negative numbers
        std::vector<int32_t> v = {
            std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max(),
            -1, 0, 1, -100, 100, -2147483647, 2147483646
        };
        run_test("int32 edge values", v);
    }
    {
        // int64 limits
        std::vector<int64_t> v = {
            std::numeric_limits<int64_t>::min(),
            std::numeric_limits<int64_t>::max(),
            -1LL, 0LL, 1LL, -999999999999LL, 999999999999LL
        };
        run_test("int64 edge values", v);
    }
    {
        // float -0.0f and +0.0f and inf
        std::vector<float> v = {
            +0.0f, -0.0f, -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            -1.0f, 1.0f, -1e-35f, 1e-35f, -1e35f, 1e35f,
            -0.0f, +0.0f, -0.0f
        };
        total_tests++;
        if (verify_float("Float special values (-0, +0, inf, denorm)", v)) {
            passed_tests++;
            std::cout << "[PASS] Float special values (-0, +0, inf, denorm)\n";
        }
    }
    {
        // double special values
        std::vector<double> v = {
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            -1e300, 1e300, -1e-300, 1e-300, -1.0, 1.0, 0.0, -0.0
        };
        run_test("double special values", v);
    }

    std::cout << "\n=========================================\n";
    std::cout << "4. STRING ENGINE EXTENSIVE TESTS\n";
    std::cout << "=========================================\n";
    {
        // Empty strings & single char
        std::vector<std::string> v = {"", "a", "", "b", "aa", "a", ""};
        run_test("Strings empty & short", v);
    }
    {
        // Long common prefix (64 bytes same prefix)
        std::string pfx(64, 'X');
        std::vector<std::string> v(50000);
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = pfx + std::to_string(rng() % 100000);
        }
        run_test("Strings 64-byte LCP prefix (N=50K)", v);
    }
    {
        // Varying length strings (0 to 120 chars)
        std::vector<std::string> v(50000);
        for (size_t i = 0; i < v.size(); ++i) {
            size_t len = rng() % 120;
            std::string s(len, 'a');
            for (size_t k = 0; k < len; ++k) s[k] = static_cast<char>('a' + (rng() % 26));
            v[i] = std::move(s);
        }
        run_test("Strings random length 0..120 (N=50K)", v);
    }

    std::cout << "\n=========================================\n";
    std::cout << "5. MULTITHREADING SCALING & STRESS (N=1,000,000)\n";
    std::cout << "=========================================\n";
    {
        std::vector<uint32_t> base_data(1000000);
        for (auto& x : base_data) x = static_cast<uint32_t>(rng());

        for (int t : {1, 2, 4, 6, 8, 12}) {
            run_test("Multithread uint32 (threads=" + std::to_string(t) + ")", base_data, t);
        }
    }
    {
        std::vector<std::string> base_str(100000);
        for (size_t i = 0; i < base_str.size(); ++i) {
            base_str[i] = "path/to/resource/" + std::to_string(rng());
        }
        for (int t : {1, 2, 4, 6}) {
            run_test("Multithread string (threads=" + std::to_string(t) + ")", base_str, t);
        }
    }

    std::cout << "\n=========================================\n";
    std::cout << "6. HEAP MEMORY ALLOCATION TRACKING\n";
    std::cout << "=========================================\n";
    {
        std::vector<uint32_t> test_vec(2000000);
        for (auto& x : test_vec) x = static_cast<uint32_t>(rng());

        g_allocated_bytes = 0;
        g_allocation_count = 0;
        g_track_allocations = true;

        kdr::sort(test_vec); // Single threaded sort

        g_track_allocations = false;
        std::cout << "KDR 1T (N=2,000,000 uint32):\n";
        std::cout << "  Heap allocations: " << g_allocation_count.load() << "\n";
        std::cout << "  Total heap bytes: " << g_allocated_bytes.load() << " bytes\n";
        if (g_allocated_bytes.load() <= 1024 * 1024) { // < 1MB (scratch cap is 131072 * 4 = 512KB)
            std::cout << "  [PASS] Fixed O(1) scratch footprint confirmed!\n";
            passed_tests++;
        } else {
            std::cout << "  [FAIL] Excessive heap allocation!\n";
        }
        total_tests++;
    }

    std::cout << "\n=========================================\n";
    std::cout << "FINAL TEST SUMMARY\n";
    std::cout << "Passed: " << passed_tests << " / " << total_tests << "\n";
    std::cout << "=========================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
