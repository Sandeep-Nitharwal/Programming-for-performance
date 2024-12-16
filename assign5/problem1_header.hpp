#include <bits/stdc++.h>
#include <pthread.h>
#include <omp.h>

using namespace std;

#define DEFAULT_MAP_SIZE (1000000)
#define LARGE_PRIME (1000000007) 
#define NUM_THREADS (4)
#define LOAD_FACTOR (0.8)

class my_pair
{
  public:
    int64_t key;
    int64_t value;
    pthread_mutex_t lock;

    my_pair(int64_t _key = -1, int64_t _value = -1) 
        : key(_key), value(_value)
    {
        int64_t ret_val = pthread_mutex_init(&lock, NULL);
        if (ret_val)
        {
            cerr << "Mutex init failed" << endl;
            exit(1);
        }
    }

    void acquire_lock()
    {
        pthread_mutex_lock(&lock);
    }

    void release_lock()
    {
        pthread_mutex_unlock(&lock);
    }

    ~my_pair()
    {
        pthread_mutex_destroy(&lock);
    }
};

class hashTable
{
public:
    atomic<int64_t> curr_size;
    my_pair* data;
    int64_t map_size;
    int64_t batch_number;

    pair<int64_t,int64_t> hash1(int64_t key);
    pair<int64_t,int64_t> hash2(int64_t key);
    pair<int64_t,int64_t> hash3(int64_t key,int64_t val1, int64_t val2);
    pair<int64_t,int64_t> hash4(int64_t key,int64_t val1, int64_t val2);


    hashTable(int64_t default_size = DEFAULT_MAP_SIZE)
    {
        batch_number = 0;
        curr_size = 0;
        map_size = DEFAULT_MAP_SIZE;
        data = new my_pair[default_size];
    }

    void resize(int64_t new_size, int64_t hash_type);
    void insert(int64_t key, int64_t val, bool *result,int64_t hash_type);
    void remove(int64_t key, bool *result,int64_t hash_type);
    void look_up(int64_t key, int64_t *result,int64_t hash_type);
    void print();
    void clear();
    int64_t size();
    int64_t get_map_size();

    ~hashTable()
    {
        delete[] data;
    }
};


int64_t hashTable::size()
{
    int64_t curr = curr_size.load();
    return curr;
}

int64_t hashTable::get_map_size()
{
    int64_t curr = map_size;
    return curr;
}

pair<int64_t,int64_t> hashTable::hash1(int64_t key)
{
    return {key % map_size , (LARGE_PRIME - (key) % LARGE_PRIME) % map_size};
}

pair<int64_t,int64_t> hashTable::hash2(int64_t key)
{
    return {key % map_size , ((key*key) % LARGE_PRIME + LARGE_PRIME ) % map_size};
}

pair<int64_t,int64_t> hashTable::hash3(int64_t key,int64_t val1, int64_t val2)
{
    return {key % map_size , ((val1*key + val2) % LARGE_PRIME + LARGE_PRIME ) % map_size};
}

pair<int64_t,int64_t> hashTable::hash4(int64_t key,int64_t val1, int64_t val2)
{
    return {key % map_size , ((((val1*val1)%LARGE_PRIME)*key + val2) % LARGE_PRIME + LARGE_PRIME) % map_size};
}


void hashTable::look_up(int64_t key, int64_t* result,int64_t hash_type)
{
    pair<int64_t,int64_t> temp;
    if(hash_type == 1){
        temp = hash1(key);
    }
    else if(hash_type == 2){
        temp = hash2(key);
    }
    else if(hash_type == 3){
        temp = hash3(key,2,987654323);
    }
    else{
        temp = hash4(key,2,987654323);
    }
    int64_t index = temp.first;
    int64_t first = temp.first;
    int64_t count = 0;
    int64_t secondary_index = temp.second;

    data[index].acquire_lock();
    while ((data[index].key == -1*(batch_number+1) || data[index].key >= 0) && data[index].key != key)
    {
        data[index].release_lock();
        index = (index + secondary_index) % map_size;
        data[index].acquire_lock();

        if (index == first || count > map_size) break;
        count++;
    } 

    if (data[index].key == key)
    {
        *result = data[index].value;
        data[index].release_lock();
        return;
    }

    *result = -1;
    data[index].release_lock();
    return; 
}

void hashTable::insert(int64_t key, int64_t value, bool* result,int64_t hash_type)
{
    pair<int64_t,int64_t> temp;
    if(hash_type == 1){
        temp = hash1(key);
    }
    else if(hash_type == 2){
        temp = hash2(key);
    }
    else if(hash_type == 3){
        temp = hash3(key,2,987654323);
    }
    else{
        temp = hash4(key,2,987654323);
    }
    int64_t index = temp.first;
    int64_t first = temp.first;
    int64_t count = 0;
    int64_t secondary_index = temp.second;

    data[index].acquire_lock();
    while ((data[index].key == -1*(batch_number+1) || data[index].key >= 0) && data[index].key != key)
    {
        data[index].release_lock();
        index = (index + secondary_index) % map_size;
        data[index].acquire_lock();

        if (index == first || count > map_size) break;
        count++;
    } 

    if (data[index].key == key)
    {
        *result = false;
        data[index].release_lock();
        return;
    }

    *result = true;
    data[index].value = value;
    data[index].key = key;
    data[index].release_lock();
    curr_size.fetch_add(1);
    return;
}

void hashTable::remove(int64_t key, bool* result,int64_t hash_type)
{
    pair<int64_t,int64_t> temp;
    if(hash_type == 1){
        temp = hash1(key);
    }
    else if(hash_type == 2){
        temp = hash2(key);
    }
    else if(hash_type == 3){
        temp = hash3(key,2,987654323);
    }
    else{
        temp = hash4(key,2,987654323);
    }
    int64_t index = temp.first;
    int64_t first = temp.first;
    int64_t count = 0;
    int64_t secondary_index = temp.second;

    data[index].acquire_lock();

    while ((data[index].key == -1*(batch_number+1) || data[index].key >= 0) && data[index].key != key)
    {
        data[index].release_lock();
        index = (index + secondary_index) % map_size;
        data[index].acquire_lock();

        if (index == first || count > map_size) break;
        count++;
    } 

    if (data[index].key == key)
    {
        *result = true;
        data[index].key = -1*(batch_number+1);
        data[index].release_lock();
        curr_size.fetch_add(-1);
        return;
    }

    *result = false;
    data[index].release_lock();
    return;
}

void hashTable::print()
{
    cout << "Map Size = " << curr_size << endl;
    for(int i=0;i<map_size;i++)
    {
        if (data[i].key >= 0)
        {
            cout << "Key: " << data[i].key << " Value: " << data[i].value << endl; 
        }
    }
}

void hashTable::clear()
{
    for(int i=0;i<map_size;i++)
    {
        data[i].key = -1;
        data[i].value = -1;
    }
    curr_size = 0;
    batch_number = 0;
}

void hashTable::resize(int64_t new_size, int64_t hash_type)
{
    my_pair* tmp = data;
    int64_t old_size = map_size;
    batch_number = 0;

    map_size = new_size;
    data = new my_pair[new_size];

    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for
    for(int i=0;i<old_size;i++)
    {
        if (tmp[i].key >= 0)
        {
            bool result;
            insert(tmp[i].key, tmp[i].value, &result,hash_type);
            assert(result);
        }
    }
}

