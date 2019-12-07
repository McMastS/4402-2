#include <iostream>
#include <string>
#include <cassert>
#include <ctime>
#include <stdio.h>

using namespace std;

struct cuda_exception 
{
    explicit cuda_exception(const char *err) : error_info(err) {}
    explicit cuda_exception(const string &err) : error_info(err) {}
    string what() const throw() { return error_info; }

    private:
    string error_info;
};

void checkCudaError(const char *msg) 
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        string error_info(msg);
        error_info += " : ";
        error_info += cudaGetErrorString(err);
        throw cuda_exception(error_info);
    }
}

template<typename T>
void random_matrices(T *M, T *N, size_t height, size_t width, int p = 2) 
{
    for(size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            int random = rand() % p;
            M[i * width + j] = random;
            N[i * width + j] = random; 
        }
    }
}

template<typename T>
void print_matrix(const T *M, size_t height, size_t width)
{
    if (height >= 32 || width >= 32) {
        cout << "a matrix of height " << height << ", of width " << width << endl;
        return;
    }

    for(size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            cout << M[i * width + j] << "   ";
        }
        cout << endl;
    }
    cout << endl;
}

void serial_min_plus(int *A, size_t n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                const unsigned int kj = k *n + j;
                const unsigned int ij = i*n + j;
                const unsigned int ik = i*n + k;

                int t1 = A[ik] + A[kj];
                int t2 = A[ij];
                A[ij] = (t1 < t2) ? t1: t2;
            }
        }
    }
}

#define BLOCK_SIZE 4

__global__ void min_plus_kernel(int *C, size_t n, size_t k) 
{
    const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= n) || (j >= n) || (k >= n)) return;

    const unsigned int kj = k *n + j;
    const unsigned int ij = i*n + j;
    const unsigned int ik = i*n + k;

    int t1 = C[ik] + C[kj];
    int t2 = C[ij];
    C[ij] = (t1 < t2) ? t1: t2;
}

void min_plus_gpu(int *C, size_t n)
{
    size_t mem_size = n * n * sizeof(int);

    int *Cd;
    cudaMalloc((void **)&Cd, mem_size);
    checkCudaError("allocating GPU memory for matrix");
    cudaMemcpy(Cd, C, mem_size, cudaMemcpyHostToDevice);

    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x,
        (n + block_dim.y - 1) / block_dim.y);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
    for (int k = 0; k < n; k++) {
        min_plus_kernel<<<grid_dim, block_dim>>>(Cd, n, k);
        cudaThreadSynchronize();
    }
    cudaMemcpy(C, Cd, mem_size, cudaMemcpyDeviceToHost);

    cudaFree(Cd);
}

int main()
{
    // int *W, *serial_W;
    int n = 4;
    int W [n * n] = {0, 3, 100, 7, 8, 0, 2, 100, 5, 100, 0, 1, 2, 100, 100, 0};
    int serial_W [n*n] = {0, 3, 100, 7, 8, 0, 2, 100, 5, 100, 0, 1, 2, 100, 100, 0};
    // cout << "Please enter a value for n: " << endl;
    // cin >> n;
    
    // assert(n % BLOCK_SIZE == 0);

    try {
        
        // random_matrices(W, serial_W, n, n, 10);

        print_matrix(W, n, n);
        print_matrix(serial_W, n, n);

        min_plus_gpu(W, n);
        serial_min_plus(serial_W, n);
    } catch (cuda_exception &err) {
        cout << err.what() << endl;
        // delete [] W;
        // delete [] serial_W;
        return EXIT_FAILURE;
    } catch (...) {
        // delete [] W;
        // delete [] serial_W;
        cout << "unknown exeception" << endl;
        return EXIT_FAILURE;
    }

    print_matrix(W, n, n);
    print_matrix(serial_W, n, n);

    // delete [] W;
    // delete [] serial_W;
    return 0;
}
