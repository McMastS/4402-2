#include <iostream>
#include <string>
#include <cassert>
#include <ctime>

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
void random_matrix(T *M, size_t height, size_t width, int p = 2) 
{
    for(size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            M[i * width + j] = rand() % p; 
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

#define BLOCK_SIZE 16

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

int main()
{
    int *W;
    int n;
    cout << "Please enter a value for n: " << endl;
    cin >> n;
    
    assert(n % BLOCK_SIZE == 0);

    size_t mem_size = n * n * sizeof(int);

    try {
        W = new int[n * n];
        random_matrix(W, n, n);
    } catch (cuda_exception &err) {
        cout << err.what() << endl;
        delete [] W;
        return EXIT_FAILURE;
    } catch (...) {
        delete [] W;
        cout << "unknown exeception" << endl;
        return EXIT_FAILURE;
    }

    print_matrix(W, n, n);

    delete [] W;
    return 0;
}
