#include "tensor/tensor.h"



int main() {
    

    int n = 3;


    Tensor t = Tensor::zeros({n, n}, 2);
    t(2,2) = 42.f;

    t.print();


    printf("Creating a tensor of shape (%d, %d) filled with zeros...\n", n, n);
    

    return 0;
}