#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include "problem2_header.hpp"

using std::cout;
using std::endl;
using std::string;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::filesystem::path;

static constexpr uint64_t RANDOM_SEED = 42;
static const uint32_t bucket_count = 1000;
static constexpr uint64_t MAX_OPERATIONS = 1e+15;
static const uint32_t SENTINEL_KEY = 0;
static const uint32_t SENTINEL_VALUE = 0;
static const uint32_t PROBING_RETRIES = (1 << 20);
static const uint32_t TOMBSTONE_KEY = UINT32_MAX;

inline uint64_t packKeyValue(uint32_t key, uint32_t val) {
    return (static_cast<uint64_t>(key) << 32) |
            (static_cast<uint32_t>(val) & 0xFFFFFFFF);
}

// Function to unpack a 64-bit integer into two 32-bit integers
inline void unpackKeyValue(uint64_t value, uint32_t& key, uint32_t& val) {
    key = static_cast<uint32_t>(value >> 32);
    val = static_cast<uint32_t>(value & 0xFFFFFFFF);
}

void create_file(path pth, const uint32_t* data, uint64_t size) {
    FILE* fptr = NULL;
    fptr = fopen(pth.string().c_str(), "wb+");
    fwrite(data, sizeof(uint32_t), size, fptr);
    fclose(fptr);
}

/** Read n integer data from file given by pth and fill in the output variable
    data */
void read_data(path pth, uint64_t n, uint32_t* data) {
    FILE* fptr = fopen(pth.string().c_str(), "rb");
    string fname = pth.string();
    if (!fptr) {
        string error_msg = "Unable to open file: " + fname;
        perror(error_msg.c_str());
    }
    int freadStatus = fread(data, sizeof(uint32_t), n, fptr);
    if (freadStatus == 0) {
        string error_string = "Unable to read the file " + fname;
        perror(error_string.c_str());
    }
    fclose(fptr);
}

void validFlagsDescription() {
    cout << "add: specify number of push operations\n";
    cout << "rns: the number of iterations\n";
    cout << "rem: specify number of pop operations\n";
}


// These variables may get overwritten after parsing the CLI arguments
/** total number of operations */
uint64_t NUM_OPS = 1000000;
/** percentage of insert queries */
uint64_t INSERT = 600000;
/** percentage of delete queries */
uint64_t DELETE = 400000;
/** number of iterations */
uint64_t runs = 2;

// Code snippet to parse command line flags and initialize the variables
int parse_args(char* arg) {
  string s = string(arg);
  string s1;
  uint64_t val;

  try {
        s1 = s.substr(0, 4);
        string s2 = s.substr(5);
        val = stol(s2);
    } catch (...) {
        cout << "Supported: " << std::endl;
        cout << "-*=[], where * is:" << std::endl;
        validFlagsDescription();
        return 1;
    }

    if (s1 == "-add") {
        INSERT = val;
    } else if (s1 == "-rns") {
        runs = val;
    } else if (s1 == "-rem") {
        DELETE = val;
    } else {
        std::cout << "Unsupported flag:" << s1 << "\n";
        std::cout << "Use the below list flags:\n";
        validFlagsDescription();
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        int error = parse_args(argv[i]);
        if (error == 1) {
        cout << "Argument error, terminatih_keys_delng run.\n";
        exit(EXIT_FAILURE);
        }
    }

    uint64_t ADD = INSERT;
    uint64_t REM = DELETE;
    uint64_t NUM_OPS = ADD + REM;

    cout << "NUM OPS: " << NUM_OPS << " ADD: " << ADD << " REM: " << REM << "\n";

    auto* h_keys_insert = new uint32_t[ADD];
    memset(h_keys_insert, 0, sizeof(uint32_t) * ADD);

    // Use shared files filled with random numbers
    path cwd = std::filesystem::current_path();
    path path_insert_keys = cwd / "random_values_insert.bin";

    assert(std::filesystem::exists(path_insert_keys));

    // Read data from file
    read_data(path_insert_keys, ADD, h_keys_insert);

    // Max limit of the uint32_t: 4,294,967,295
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_int_distribution<uint32_t> dist_int(1, NUM_OPS);

    float total_time = 0.0F;

    HRTimer start, end;
    Stack s;

    for (int i = 0; i < runs; i++) {
        start = HR::now();
        solve(ADD, NUM_OPS, h_keys_insert, &s);
        end = HR::now();
        float iter_time = duration_cast<milliseconds>(end - start).count();

        total_time += iter_time;
    }

    cout << "Time taken by(ms): " << total_time / runs << " Threads : " << NUM_THREADS << "\n";

    delete[] h_keys_insert;

    return EXIT_SUCCESS;
}
