#include <iostream>
#include <math.h>

int main() {
    int N = 5000;
    size_t nearestN = powf(2, ceilf(log2f(N)));
    printf("%d\n", nearestN);
    for (size_t stride = nearestN >> 1ul; stride > 0; stride >>= 1ul) {
        printf("%d\n", stride);
    }
    for (int i = nearestN / 2; i > 0; i /= 2) {
        printf("%d\n", i);
    }
    return 0;
}