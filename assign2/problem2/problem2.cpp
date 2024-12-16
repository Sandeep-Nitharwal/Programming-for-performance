#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>
#include <string>
#include <map>
#include <atomic>

using namespace std;

// Global variables for synchronization
mutex buffer_mutex;
condition_variable buffer_not_empty;
condition_variable buffer_not_full;
queue<string> shared_buffer;
atomic<int> producers_done(0);
int num_producers;
int lines_per_thread;
int buffer_size;
string input_file_path;
string output_file_path;

// Function to read lines from the input file and push to the buffer
void producer_thread(ifstream& infile, int id) {
    vector<string> lines;
    string line;
    int lines_processed = 0;

    while (true) {
        // Reading lines from file
        if(lines_processed == lines_per_thread) break;
        {
            lock_guard<mutex> lock(buffer_mutex);
            lines.clear();
            for (int i = 0; i < lines_per_thread && getline(infile, line); i++) {
                lines.push_back(line);
                lines_processed++;
            }
            if (lines.empty()) break;  // No more lines to process
        }

        // Push lines to buffer
        {
            unique_lock<mutex> lock(buffer_mutex);
            buffer_not_full.wait(lock, [] { return shared_buffer.size() < buffer_size; });

            for (const auto& l : lines) {
                shared_buffer.push(l);
            }

            buffer_not_empty.notify_one();
        }
    }

    producers_done++;
}

// Function to consume lines from the buffer and write to the output file
void consumer_thread(ofstream& outfile) {
    while (true) {
        vector<string> lines_to_write;
        {
            unique_lock<mutex> lock(buffer_mutex);
            buffer_not_empty.wait(lock, [] { return !shared_buffer.empty() || producers_done == num_producers; });

            if (shared_buffer.empty() && producers_done == num_producers) break;

            while (!shared_buffer.empty()) {
                lines_to_write.push_back(shared_buffer.front());
                shared_buffer.pop();
            }

            buffer_not_full.notify_all();
        }

        // Write to the output file outside the critical section
        for (const auto& line : lines_to_write) {
            outfile << line << endl;
        }
    }
}

map<string, string> parse_args(int argc, char* argv[]) {
    map<string, string> args;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        size_t pos = arg.find('=');
        if (pos != string::npos) {
            string key = arg.substr(1, pos - 1); // Removes the '-' and gets the flag name
            string value = arg.substr(pos + 1);   // Gets the value after '='
            args[key] = value;
        }
    }
    return args;
}

int main(int argc, char* argv[]) {
    // Parse the command-line arguments
    map<string, string> args = parse_args(argc, argv);

    // Validate and extract required arguments
    if (args.find("inp") == args.end() || args.find("thr") == args.end() || 
        args.find("lns") == args.end() || args.find("buf") == args.end() || 
        args.find("out") == args.end()) {
        cerr << "Usage: " << argv[0] << " -inp=<input_file> -thr=<threads> -lns=<lines_per_thread> -buf=<buffer_size> -out=<output_file>" << endl;
        return 1;
    }

    input_file_path = args["inp"];
    num_producers = stoi(args["thr"]);
    lines_per_thread = stoi(args["lns"]);
    buffer_size = stoi(args["buf"]);
    output_file_path = args["out"];

    ifstream infile(input_file_path);
    if (!infile.is_open()) {
        cerr << "Error opening input file: " << input_file_path << endl;
        return 1;
    }

    ofstream outfile(output_file_path);
    if (!outfile.is_open()) {
        cerr << "Error opening output file: " << output_file_path << endl;
        return 1;
    }

    if(num_producers < 0){
		cerr << "Number of producer threads must be non-negative" << std::endl;
		exit(EXIT_FAILURE);
	}

	if(lines_per_thread < 0){
		cerr << "Number of lines per thread must be non-negative" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	if(buffer_size <= 0){
		cerr << "Buffer size must be positive" << std::endl;
		exit(EXIT_FAILURE);
	}

    // Launch producer threads
    vector<thread> producers;
    for (int i = 0; i < num_producers; ++i) {
        producers.emplace_back(producer_thread, ref(infile), i);
    }

    // Launch the consumer thread
    thread consumer(consumer_thread, ref(outfile));

    // Join all producer threads
    for (auto& producer : producers) {
        producer.join();
    }

    // Notify consumer thread to finish
    {
        unique_lock<mutex> lock(buffer_mutex);
        buffer_not_empty.notify_one();
    }

    // Join the consumer thread
    consumer.join();

    infile.close();
    outfile.close();

    return 0;
}
