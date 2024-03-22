#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
/* 
Program estimates Pi via Monte Carlo simulation of a set amount of random points inside or outside a circle.
Each CUDA thread generates a sequence of random points within the unit square.
The estimate of pi is calculated using the ratio of points inside the circle to the total number of points generated.
*/


// CUDA kernel to perform Monte Carlo simulation
__global__ void monte_carlo_estimate(unsigned long long *result, unsigned long long num_points) {
    unsigned long long thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;
    curandState_t state;    // seed for random #, each thread has its own to avoid repeats
    curand_init(clock64(), thread_id, 0, &state);   // Init state based on clock

    unsigned long long points_in_circle = 0;

    for (unsigned long long i = thread_id; i < num_points; i += stride) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        float distance = x * x + y * y;
        if (distance <= 1.0f) {        // Check inside or outside of circle
            points_in_circle++;
        }
    }

    atomicAdd(result, points_in_circle);    // Total number of points in circle avaible in result at end
}

int main(int argc, char **argv) {
    unsigned long long num_points = 1000000000; // Random points to generate
    unsigned long long *result;
    cudaMallocManaged(&result, sizeof(unsigned long long)); // Accessible via CPU and GPU
    *result = 0;

    unsigned int blocks = 256; 
    unsigned int threads_per_block = 256; 

    // Timing for benchmark purposes
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
       

    // Launch CUDA kernel
    monte_carlo_estimate<<<blocks, threads_per_block>>>(result, num_points);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    double pi_estimate = 4.0 * (*result / (double)num_points);  // Estimate does not get accurate enough to need bigger than double
    printf("Estimated value of pi: %lf\n", pi_estimate);
    printf("Calculation time: %.3f milliseconds\n", milliseconds);

    // Handle destruction/freeing
    cudaFree(result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}