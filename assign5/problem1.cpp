#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include "problem1_header.hpp"
// #include <tbb/tbb.h>
// #include <tbb/concurrent_hash_map.h>

// using namespace tbb;

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

typedef struct {
  uint32_t key;
  uint32_t value;
} KeyValue;

// Pack key-value into a 64-bit integer
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

// These variables may get overwritten after parsing the CLI arguments
/** total number of operations */
uint64_t NUM_OPS = 1e3;
/** percentage of insert queries */
uint64_t INSERT = 60;
/** percentage of delete queries */
uint64_t DELETE = 30;
/** number of iterations */
uint64_t runs = 2;

// List of valid flags and description
void validFlagsDescription() {
  cout << "ops: specify total number of operations\n";
  cout << "rns: the number of iterations\n";
  cout << "add: percentage of insert queries\n";
  cout << "rem: percentage of delete queries\n";
}

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

  if (s1 == "-ops") {
    NUM_OPS = val;
  } else if (s1 == "-rns") {
    runs = val;
  } else if (s1 == "-add") {
    INSERT = val;
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


void batch_insert(hashTable* hm, int64_t n, KeyValue* pairs, bool* result, int64_t hash_type)
{
    if (hm -> size() + n  >= LOAD_FACTOR * hm -> get_map_size())
    {
        int64_t tmp = (hm -> size() + n) * 2; 
        hm -> resize(tmp,hash_type);
    }

    omp_set_num_threads(NUM_THREADS);
    
    #pragma omp parallel for
    for(int i=0;i<n;i++)
    {
        hm-> insert(pairs[i].key, pairs[i].value, &result[i], hash_type);
    }
    return;
}

void batch_delete(hashTable* hm, int64_t n, uint32_t* keys, bool* result, int64_t hash_type)
{
    hm -> batch_number += 1;
    omp_set_num_threads(NUM_THREADS);
    
    #pragma omp parallel for
    for(int i=0;i<n;i++)
    {
        hm-> remove(keys[i], &result[i], hash_type);
    }
    return;
}

void batch_lookup(hashTable* hm, int64_t n, uint32_t* keys, int64_t* result, int64_t hash_type)
{
    omp_set_num_threads(NUM_THREADS);
    
    #pragma omp parallel for
    for(int i=0;i<n;i++)
    {
        hm-> look_up(keys[i], &result[i], hash_type);
    }
    return;
}

// void batch_insert(tbb::concurrent_hash_map<uint32_t, uint32_t>* hm, int64_t n, KeyValue* pairs, bool* result, int64_t hash_type)
// {
//     omp_set_num_threads(NUM_THREADS);
//     #pragma omp parallel for
//     for (int i = 0; i < n; i++) {
//         tbb::concurrent_hash_map<uint32_t, uint32_t>::accessor accessor;
//         bool inserted = hm -> insert(accessor, pairs[i].key);
//         if (inserted) {
//             accessor->second = pairs[i].value;
//         }
//         result[i] = inserted;
//     }
// }

// void batch_delete(tbb::concurrent_hash_map<uint32_t, uint32_t>* hm, int64_t n, uint32_t* keys, bool* result, int64_t hash_type)
// {
//     omp_set_num_threads(NUM_THREADS);
    
//     #pragma omp parallel for
//     for (int i = 0; i < n; i++) {
//         result[i] = hm -> erase(keys[i]);
//     }
// }

// void batch_lookup(tbb::concurrent_hash_map<uint32_t, uint32_t>* hm, int64_t n, uint32_t* keys, int64_t* result, int64_t hash_type)
// {
//     omp_set_num_threads(NUM_THREADS);
    
//     #pragma omp parallel for
//     for (int64_t i = 0; i < n; i++)
//     {
//         tbb::concurrent_hash_map<uint32_t, uint32_t>::const_accessor accessor;
//         result[i] = hm->find(accessor, keys[i]) ? accessor->second : -1;
//     }
// }

void run (auto* hm, uint64_t ADD, uint64_t REM, uint64_t FIND, KeyValue* h_kvs_insert, uint32_t* h_keys_del, uint32_t* h_keys_lookup,int64_t hash_type = 1 )
{
    float total_insert_time = 0.0F;
    float total_delete_time = 0.0F;
    float total_search_time = 0.0F;
    HRTimer start, end;
    bool* result_add = new bool[ADD];
    bool* result_rem = new bool[REM];
    int64_t* result_find = new int64_t[FIND];

    for (int i = 0; i < runs; i++) {
        start = HR::now();
        batch_insert(hm, ADD, h_kvs_insert, result_add, hash_type);
        end = HR::now();
        float iter_insert_time = duration_cast<milliseconds>(end - start).count();

        // hm->print();

        start = HR::now();
        batch_delete(hm, REM, h_keys_del, result_rem, hash_type);
        end = HR::now();
        float iter_delete_time = duration_cast<milliseconds>(end - start).count();
        
        // hm->print();

        start = HR::now();
        batch_lookup(hm, FIND, h_keys_lookup, result_find, hash_type);
        end = HR::now();
        float iter_search_time = duration_cast<milliseconds>(end - start).count();
        
        // hm->print();

        total_insert_time += iter_insert_time;
        total_delete_time += iter_delete_time;
        total_search_time += iter_search_time;

        hm -> clear();
    }

    cout << "Time taken by insert kernel (ms): " << total_insert_time / runs
        << "\nTime taken by delete kernel (ms): " << total_delete_time / runs
        << "\nTime taken by search kernel (ms): " << total_search_time / runs
        << "\n\n";
    
    delete[] result_add;
    delete[] result_rem;
    delete[] result_find;
}
// ----------------Unit test cases------------------//
// test for insertions for all combinations like same key , same value, different keys etc;
void test1(auto* hm)
{
    int64_t n = 1e4;
    KeyValue* temp = new KeyValue[n];
    bool* ans = new bool[n];

    for(int i=0;i<n;i++)
    {
        temp[i].key = i/3;
        temp[i].value = i/3+1;
    }

    batch_insert(hm, n, temp, ans,1);

    if(hm -> size() != (n-1)/3 + 1){
      cout << "TEST 1:FAILED" << endl;
      hm -> clear();
      delete[] ans;
      delete[] temp;
      return;
    } 

    hm -> clear();
    delete[] ans;
    delete[] temp;

    cout << "TEST 1: PASSED" << endl;
}


// test for deletion
void test2(auto* hm)
{
    int64_t n = 1e4;
    KeyValue* temp = new KeyValue[n];
    bool* ans = new bool[n];
    uint32_t* temp_del = new uint32_t[n/10];
    bool* ans_del = new bool[n/10];

    for(int i=0;i<n;i++)
    {
        temp[i].key = i/3;
        temp[i].value = i/3+1;
        if(i<n/10)
          temp_del[i] = i;
    }

    batch_insert(hm, n, temp, ans,1);


    if(hm -> size() != (n-1)/3 + 1){
      cout << "TEST 2:FAILED" << endl;
      hm -> clear();
      delete[] ans_del;
      delete[] temp_del;
      delete[] ans;
      delete[] temp;
      return;
    } 

    batch_delete(hm, n/10, temp_del, ans_del,1);

    for(int i=0;i<n/10;i++)
    {
        if (!ans_del[i]){
          cout << "TEST 2:FAILED" << endl;
          hm -> clear();
          delete[] ans_del;
          delete[] temp_del;
          delete[] ans;
          delete[] temp;
          return;
        }
    }

    if(hm -> size() != ((n-1)/3 + 1 - n/10)){
      cout << "TEST 2:FAILED" << endl;
      hm -> clear();
      delete[] ans_del;
      delete[] temp_del;
      delete[] ans;
      delete[] temp;
      return;
    } 


    hm -> clear();
    delete[] ans_del;
    delete[] temp_del;
    delete[] ans;
    delete[] temp;

    cout << "TEST 2: PASSED" << endl;
}

// for lookups
void test3(auto* hm)
{
    int64_t n = 1e4;
    KeyValue* temp = new KeyValue[n];
    bool* ans = new bool[n];
    uint32_t* temp_del = new uint32_t[n/10];
    bool* ans_del = new bool[n/10];
    uint32_t* temp_lookup = new uint32_t[n/3];
    int64_t* ans_lookup = new int64_t[n/3];

    for(int i=0;i<n;i++)
    {
        temp[i].key = i/3;
        temp[i].value = i/3+1;
        if(i<n/10)
          temp_del[i] = i;
        if(i<n/3)
          temp_lookup[i] = i;
    }

    batch_insert(hm, n, temp, ans,1);

    if(hm -> size() != (n-1)/3 + 1){
      cout << "TEST 3:FAILED" << endl;
      hm -> clear();
      delete[] ans_lookup;
      delete[] temp_lookup;
      delete[] ans_del;
      delete[] temp_del;
      delete[] ans;
      delete[] temp;
      return;
    } 

    batch_delete(hm, n/10, temp_del, ans_del,1);

    for(int i=0;i<n/10;i++)
    {
        if (!ans_del[i]){
          cout << "TEST 3:FAILED" << endl;
          hm -> clear();
          delete[] ans_lookup;
          delete[] temp_lookup;
          delete[] ans_del;
          delete[] temp_del;
          delete[] ans;
          delete[] temp;
          return;
        }
    }

    if(hm -> size() != ((n-1)/3 + 1 - n/10)){
      cout << "TEST 3:FAILED" << endl;
      hm -> clear();
      delete[] ans_lookup;
      delete[] temp_lookup;
      delete[] ans_del;
      delete[] temp_del;
      delete[] ans;
      delete[] temp;
      return;
    } 

    batch_lookup(hm, n/3, temp_lookup, ans_lookup,1);


    if(hm -> size() != ((n-1)/3 + 1 - n/10)){
      cout << "TEST 3:FAILED" << endl;
      hm -> clear();
      delete[] ans_lookup;
      delete[] temp_lookup;
      delete[] ans_del;
      delete[] temp_del;
      delete[] ans;
      delete[] temp;
      return;
    }


    hm -> clear();
    delete[] ans_lookup;
    delete[] temp_lookup;
    delete[] ans_del;
    delete[] temp_del;
    delete[] ans;
    delete[] temp;

    cout << "TEST 3: PASSED" << endl;
}

// ---------------------- Unit test cases end-----------------//

int main(int argc, char* argv[]) {
  for (int i = 1; i < argc; i++) {
    int error = parse_args(argv[i]);
    if (error == 1) {
      cout << "Argument error, terminating run.\n";
      exit(EXIT_FAILURE);
    }
  }

  uint64_t ADD = NUM_OPS * (INSERT / 100.0);
  uint64_t REM = NUM_OPS * (DELETE / 100.0);
  uint64_t FIND = NUM_OPS - (ADD + REM);

  cout << "NUM OPS: " << NUM_OPS << " ADD: " << ADD << " REM: " << REM
       << " FIND: " << FIND << "\n";

  assert(ADD > 0);

  auto* h_kvs_insert = new KeyValue[ADD];
  memset(h_kvs_insert, 0, sizeof(KeyValue) * ADD);
  auto* h_keys_del = new uint32_t[REM];
  memset(h_keys_del, 0, sizeof(uint32_t) * REM);
  auto* h_keys_lookup = new uint32_t[FIND];
  memset(h_keys_lookup, 0, sizeof(uint32_t) * FIND);

  // Use shared files filled with random numbers
  path cwd = std::filesystem::current_path();
  path path_insert_keys = cwd / "random_keys_insert.bin";
  path path_insert_values = cwd / "random_values_insert.bin";
  path path_delete = cwd / "random_keys_delete.bin";
  path path_search = cwd / "random_keys_search.bin";

  assert(std::filesystem::exists(path_insert_keys));
  assert(std::filesystem::exists(path_insert_values));
  assert(std::filesystem::exists(path_delete));
  assert(std::filesystem::exists(path_search));

  // Read data from file
  auto* tmp_keys_insert = new uint32_t[ADD];
  read_data(path_insert_keys, ADD, tmp_keys_insert);
  auto* tmp_values_insert = new uint32_t[ADD];
  read_data(path_insert_values, ADD, tmp_values_insert);
  for (int i = 0; i < ADD; i++) {
    h_kvs_insert[i].key = tmp_keys_insert[i];
    h_kvs_insert[i].value = tmp_values_insert[i];
  }
  delete[] tmp_keys_insert;
  delete[] tmp_values_insert;

  if (REM > 0) {
    auto* tmp_keys_delete = new uint32_t[REM];
    read_data(path_delete, REM, tmp_keys_delete);
    for (int i = 0; i < REM; i++) {
      h_keys_del[i] = tmp_keys_delete[i];
    }
    delete[] tmp_keys_delete;
  }

  if (FIND > 0) {
    auto* tmp_keys_search = new uint32_t[FIND];
    read_data(path_search, FIND, tmp_keys_search);
    for (int i = 0; i < FIND; i++) {
      h_keys_lookup[i] = tmp_keys_search[i];
    }
    delete[] tmp_keys_search;
  }

  // Max limit of the uint32_t: 4,294,967,295
  std::mt19937 gen(RANDOM_SEED);
  std::uniform_int_distribution<uint32_t> dist_int(1, NUM_OPS);

  cout << "---------- hashTable ------------- " << endl;
  cout << "---------- Testcases ------------- " << endl;
  hashTable* hm  = new hashTable;
  test1(hm);
  test2(hm);
  test3(hm);

  cout << "---------- Time details ------------- " << endl;
  cout << "---------- Time details for hash1 ------------- " << endl;
  run(hm, ADD, REM, FIND, h_kvs_insert, h_keys_del, h_keys_lookup,1);
  cout << "---------- Time details for hash2 ------------- " << endl;
  run(hm, ADD, REM, FIND, h_kvs_insert, h_keys_del, h_keys_lookup,2);
  cout << "---------- Time details for hash3 ------------- " << endl;
  run(hm, ADD, REM, FIND, h_kvs_insert, h_keys_del, h_keys_lookup,3);
  cout << "---------- Time details for hash4 ------------- " << endl;
  run(hm, ADD, REM, FIND, h_kvs_insert, h_keys_del, h_keys_lookup,4);

  cout << endl;

  // tbb::concurrent_hash_map<uint32_t, uint32_t>* thm = new tbb::concurrent_hash_map<uint32_t, uint32_t>;
  // cout << "------------- TBB hashTable --------------" << endl;
  // cout << "---------- Testcases ------------- " << endl;
  // test1(thm);
  // test2(thm);
  // test3(thm);
  // cout << "---------- Time details ------------- " << endl;
  // run(thm, ADD, REM, FIND, h_kvs_insert, h_keys_del, h_keys_lookup,1);
  // delete thm;

  delete[] h_keys_del;
  delete[] h_keys_lookup;
  delete[] h_kvs_insert;

  return EXIT_SUCCESS;
}
