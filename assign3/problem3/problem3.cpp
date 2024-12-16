
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <queue>
#include <string>
#include <unistd.h>
#include <atomic>
#include <map>
#include <omp.h>

using namespace std;


queue<string> buffer;

pthread_cond_t empty_cond = PTHREAD_COND_INITIALIZER;
pthread_cond_t fill_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t buffer_mutex = PTHREAD_MUTEX_INITIALIZER;

int num_full = 0;
int num_producers;
int lines_per_thread;
int buffer_size;

int all_done = 0;
volatile int num_round_done = 0;
int num_producer_done = 0;

string input_file_path;
string output_file_path;
fstream infile;
fstream outfile;

void producer(int id) {
    string line;
    vector<string> lines;
    uint64_t lines_processed;
    int curr_round = 0;
    bool local_done = false;

    while(!all_done) {
        lines_processed = 0;
        lines.clear();

        // Avoid busy waiting with a small sleep
        while(curr_round != num_round_done && !all_done) {
            usleep(100);  // Add small sleep to prevent CPU spinning
        }

        if (all_done) break;  // Check if we should exit

        #pragma omp critical(file)
        {
            while(lines_processed < lines_per_thread && !infile.eof() && getline(infile, line)) {
                lines.push_back(line);
                lines_processed++;
            }
            local_done = infile.eof() && lines.empty();
        }
        
        // Handle both empty and non-empty lines cases
        pthread_mutex_lock(&buffer_mutex);
        
        if (!lines.empty()) {
            for(int i = 0; i < lines.size(); i++) {
                while(num_full == buffer_size && !all_done) {
                    pthread_cond_wait(&empty_cond, &buffer_mutex);
                }
                
                if (all_done) break;  // Check if we should exit

                buffer.push(lines[i]);
                num_full++;
                pthread_cond_signal(&fill_cond);
            }
        }
        
        pthread_mutex_unlock(&buffer_mutex);

        #pragma omp atomic
        num_producer_done++;

        if(num_producer_done == num_producers) {
            #pragma omp critical(reset)
            {
                num_round_done++;
                num_producer_done = 0;
            }
            
            if(local_done) {
                pthread_mutex_lock(&buffer_mutex);
                all_done = true;
                pthread_cond_broadcast(&fill_cond);
                pthread_mutex_unlock(&buffer_mutex);
            }
        }
        curr_round++;
    }
}

void consumer(void) {
    string line;

    while(true) {
        pthread_mutex_lock(&buffer_mutex);
        
        while(num_full == 0 && !all_done) {
            pthread_cond_wait(&fill_cond, &buffer_mutex);
        }
        
        // If buffer is empty and all producers are done, exit
        if(num_full == 0 && all_done) {
            pthread_mutex_unlock(&buffer_mutex);
            break;
        }
        
        // Process all available items in buffer
        while(!buffer.empty()) {
            line = buffer.front();
            buffer.pop();
            outfile << line << endl;
            num_full--;
        }
        
        pthread_cond_broadcast(&empty_cond);
        pthread_mutex_unlock(&buffer_mutex);
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

    infile.open(input_file_path.c_str(), ios::in);
    if (!infile.is_open()) {
        cerr << "Error opening input file: " << input_file_path << endl;
        return 1;
    }

    outfile.open(output_file_path.c_str(), ios::out);
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

    omp_set_num_threads(num_producers + 1);
	#pragma omp parallel for
	for(int i = 0; i < num_producers + 1; i++){
		int id = omp_get_thread_num();
		if(id == 0)
			consumer();
		else
			producer(id);
	}

    infile.close();
    outfile.close();

    return 0;
}

