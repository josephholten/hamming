#include "hamming.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>

std::vector<Base> read_seqdata(const char* path) {
    std::vector<Base> data;

    std::ifstream stream(path);
    std::cout << path << std::endl;
    std::string line;
    size_t i = 0;
    while (getline(stream, line)) {
        if (line[0] == '>')
            continue;
        for (size_t i = 0; i < line.size(); i++) {
            data.push_back(line[i]);
        }
        i++;
    }
    return data;
}

std::vector<Base> random_seqdata(size_t n) {
    std::vector<Base> data(n, 0);
    std::random_device r;
    std::default_random_engine engine(r());
    std::uniform_int_distribution<Base> dist(0,3);
    std::array<Base,4> map = {'A', 'C', 'G', 'T'};
    for (size_t i = 0; i < n; i++)
        data[i] = map[dist(engine)];
    return data;
}

std::vector<std::vector<Base>> random_seqdata(size_t n, size_t m) {
    std::vector<std::vector<Base>> data;
    for (size_t i = 0; i < m; i++) {
        data.push_back(random_seqdata(n));
    }
    return data;
}
