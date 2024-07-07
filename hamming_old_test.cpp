int test1(int argc, char** argv) {
    size_t n = 10;
    std::vector<Base> random = random_seqdata(n);
    for (size_t i = 0; i < n; i++) 
        printf("%d ", random[i]);
    printf("\n");

    std::vector<Base> data = read_seqdata(argv[1]);
    for (size_t i = 0; i < n; i++) {
        printf("'%c' ", data[i]);
    printf("\n");
    }

    return 0;
}

void print_vec(const std::vector<Base>& v) {
    for (size_t i = 0; i < v.size(); i++) {
        printf("%c ", v[i]);
    }
    printf("\n");
}

int test2(int argc, char** argv) {
    std::vector<Base> x = random_seqdata(10);
    std::vector<Base> y = random_seqdata(10);
    print_vec(x);
    print_vec(y);
    std::cout << hamming_seq_naive(10, x.data(), y.data()) << std::endl;
    std::cout << hamming_seq_branchless(10, x.data(), y.data()) << std::endl;
    return 0;
}

int test3(int argc, char** argv) {
    uint8_t a = 0xFF;
    uint8_t b = 3*a;
    uint8_t s = -b;
    printf("%x\n", b);
    printf("%d\n", b);
    printf("%d\n", s);
    return 0;
}

int main(int argc, char** argv) {
    test3(argc, argv);
}
