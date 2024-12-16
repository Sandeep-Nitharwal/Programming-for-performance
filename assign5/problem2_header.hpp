#include <atomic>
#include <iostream>
#include <omp.h>

#include <cassert>
#include <cstdint>
#include <cstdio>

#include <thread>
#include <atomic>
using namespace std;

#define NUM_THREADS (16)

class Node
{
public:
    Node* next_node;
    uint32_t val;
    Node (uint32_t _val, Node* _next, uint64_t tag);
    Node* next();
    void set(Node* current, uint64_t tag);
};

Node::Node (uint32_t _val = 0, Node* _next = nullptr, uint64_t tag = 0)
{
    val = _val;
    next_node = _next;

    tag = tag % 16;
    next_node = (Node*) ((uint64_t)next_node | tag);
}

Node* Node::next()
{
    uint64_t tmp = (uint64_t) next_node;
    tmp = ((tmp >> 4) << 4);
    return (Node*) tmp;
}

void Node::set(Node* current, uint64_t tag)
{
    tag = tag % 16;
    next_node = (Node*) ((uint64_t)current | tag);
}

class Stack
{
 public:
    atomic <Node*> top;
    atomic <uint64_t> pop_count;
    int pop();
    void push(uint32_t val);
    void print();

    Stack () : top (nullptr), pop_count (0) {}
};

void Stack::push(uint32_t val)
{
    bool ret = false;
    Node* new_node = new(std::align_val_t(16)) Node(val);
    while (!ret)
    {
        Node* old = top.load();
        new_node -> set(old, pop_count);
        ret = top.compare_exchange_strong(old, new_node);
    }
    
}

int Stack::pop()
{
    bool ret = false;
    Node* temp;

    while (!ret)
    {
        Node* old = top.load();
        if (!old) return -1;

        Node* new_top = old -> next();
        ret = top.compare_exchange_strong(old, new_top);
        if (ret) temp = old;
    }

    pop_count.fetch_add(1);
    uint32_t ret_val = temp -> val;
    delete temp;
    return ret_val;
}

void Stack::print()
{
    cout << "Stack is: ";
    Node* tmp = top.load();
    while(tmp)
    {
        cout << tmp -> val << " ";
        tmp = tmp -> next();
    }
    cout << endl;
}

void solve(uint32_t ADD, uint32_t NUM_OPS, uint32_t* keys, Stack* s)
{

    omp_set_num_threads(NUM_THREADS);
    atomic<uint32_t> add_ops = 0;

    #pragma omp parallel for schedule(static, NUM_OPS / NUM_THREADS)
    for(int i=0; i< NUM_OPS; i++)
    {
        uint32_t tid = omp_get_thread_num();
        int val;
        string operation = "PUSH";
        if (i < ADD)
        {
            val = keys[i];
            s -> push(val);
        }

        else{
            operation = "POP";
            val = s -> pop();
        }

        // #pragma omp critical
        // {
        //     cout << "THREAD "<< tid << " does a " << operation << " with value = " << val << endl;
        // }
    }
}