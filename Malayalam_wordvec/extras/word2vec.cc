#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <locale>

// Platform-specific includes
#ifdef _MSC_VER
    #define _CRT_SECURE_NO_WARNINGS
    #include <time.h>
    #include <windows.h>
#else
    #include <pthread.h>
#endif

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#include <unordered_map>

// Namespace usage for clarity
using namespace std;


const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
using real = float;  // Precision of float numbers
struct vocab_word {
    long long cn;  // Frequency of the word
    std::vector<int> point;  // Huffman tree points
    std::string word;  // Word
    std::string code;  // Huffman code
    char codelen;  // Length of Huffman code
};
std::string train_file, output_file;
std::vector<vocab_word> vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
std::unordered_map<std::string, int> vocab_hash;  // C++ replacement for vocab_hash
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
real alpha = 0.025, starting_alpha, sample = 1e-3;
std::vector<real> syn0, syn1, syn1neg, expTable;
