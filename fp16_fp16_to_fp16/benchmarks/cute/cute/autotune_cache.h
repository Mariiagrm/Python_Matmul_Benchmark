#ifndef AUTOTUNE_CACHE_H
#define AUTOTUNE_CACHE_H


#include <iostream>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>

// 1. The Key structure
struct MatrixDim {
    int m, n, k;

    // We must define the '==' operator so the map can handle collisions
    bool operator==(const MatrixDim& other) const {
        return m == other.m && n == other.n && k == other.k;
    }
};

// 2. The Custom Hash Function
struct MatrixDimHash {
    std::size_t operator()(const MatrixDim& dim) const {
        // A standard way to combine multiple integer hashes
        std::size_t h1 = std::hash<int>()(dim.m);
        std::size_t h2 = std::hash<int>()(dim.n);
        std::size_t h3 = std::hash<int>()(dim.k);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};


class AutotuneCache {
private:
    std::unordered_map<MatrixDim, int, MatrixDimHash> cache;
    std::string filepath;
    bool file_found; // <--- NEW: Track if the file exists

    void load_from_disk() {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            file_found = false;
            return;
        }
        file_found = true;

        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string item;
            int values[4];
            int i = 0;
            while (std::getline(ss, item, ',') && i < 4) {
                values[i++] = std::stoi(item);
            }
            if (i == 4) {
                MatrixDim dim = {values[0], values[1], values[2]};
                cache[dim] = values[3];
            }
        }
        file.close();
    }

public:
    AutotuneCache(const std::string& path) : filepath(path), file_found(false) {
        load_from_disk();
    }

    // NEW: Methods to let main() check the status
    bool did_file_exist() const { return file_found; }
    int get_size() const { return cache.size(); }

    int get_best_kernel(int m, int n, int k) {
        MatrixDim dim = {m, n, k};
        if (cache.find(dim) != cache.end()) return cache[dim];
        return -1; 
    }

    void save_heuristic(int m, int n, int k, int best_idx) {
        MatrixDim dim = {m, n, k};
        cache[dim] = best_idx;
        std::ofstream file(filepath, std::ios::app);
        if (file.is_open()) {
            file << m << "," << n << "," << k << "," << best_idx << "\n";
            file.close();
        }
    }
};


#endif // AUTOTUNE_CACHE_H