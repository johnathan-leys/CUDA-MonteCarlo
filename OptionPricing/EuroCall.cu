#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>

// Basic Monte carlo simulation for European option pricing

// Option pricing parameters: __constant__ is seen on device, d_ for device_variable
// Used in the geometric Brownian motion model
__constant__ float d_S0;
__constant__ float d_K;
__constant__ float d_r;
__constant__ float d_sigma;
__constant__ float d_T;

// Monte Carlo simulation parameters
const int NUM_PATHS = 1000000;
const int NUM_THREADS = 256;

// CUDA kernel for Monte Carlo simulation
__global__ void monte_carlo_kernel(float* option_prices, curandState* rand_states) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    // Initialize random number generator
    curand_init(0, thread_id, 0, &rand_states[thread_id]);

    /* 
    Perform Monte Carlo simulation: For each path, generate random #, compute stock price at maturity using
    geometric Brownian motion model, calculate payoff
    */
    for (int path_id = thread_id; path_id < NUM_PATHS; path_id += num_threads) {
        float rand_normal = curand_normal(&rand_states[thread_id]);
        float S_T = d_S0 * exp((d_r - 0.5f * d_sigma * d_sigma) * d_T + d_sigma * sqrt(d_T) * rand_normal);
        option_prices[path_id] = fmaxf(S_T - d_K, 0.0f);
    }

}

int main() {
    float option_price;
    float* option_prices;
    curandState* rand_states;

    /*
    Option pricing parameters:
    S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)\
    S_T is the stock price at time T (maturity)
    Z is a random number (from normal distribution)
    */
    float S0 = 100.0f;  // initial stock price (d_S0)
    float K = 105.0f;   // strike price of option (d_K)
    float r = 0.05f;    // risk-free interest rate (d_r)
    float sigma = 0.2f; // volatility of the stock (d_sigma)
    float T = 1.0f;     // time to maturity (d_T)

    // Copy option pricing parameters to constant memory
    cudaMemcpyToSymbol(d_S0, &S0, sizeof(float));
    cudaMemcpyToSymbol(d_K, &K, sizeof(float));
    cudaMemcpyToSymbol(d_r, &r, sizeof(float));
    cudaMemcpyToSymbol(d_sigma, &sigma, sizeof(float));
    cudaMemcpyToSymbol(d_T, &T, sizeof(float));

    // Allocate memory on GPU
    cudaMalloc(&option_prices, NUM_PATHS * sizeof(float));
    cudaMalloc(&rand_states, NUM_PATHS * sizeof(curandState));

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch Monte Carlo kernel
    int num_blocks = (NUM_PATHS + NUM_THREADS - 1) / NUM_THREADS;
    monte_carlo_kernel<<<num_blocks, NUM_THREADS>>>(option_prices, rand_states);

    // Wait for kernel to finish
    cudaDeviceSynchronize();
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> execution_time = end_time - start_time;

    // Compute option price by averaging the payoffs
    option_price = 0.0f;
    float* temp_prices = new float[NUM_PATHS];
    cudaMemcpy(temp_prices, option_prices, NUM_PATHS * sizeof(float), cudaMemcpyDeviceToHost);

    // Average the payoffs and discount by the risk-free interest rate
    for (int i = 0; i < NUM_PATHS; i++) {
        option_price += temp_prices[i];
    }
    option_price /= NUM_PATHS;
    option_price *= exp(-r * T);

    std::cout << "Option price: " << option_price << std::endl;
    std::cout << "Execution time: " << execution_time.count() << " ms" << std::endl;

    // Free memory
    cudaFree(option_prices);
    cudaFree(rand_states);
    delete[] temp_prices;

    return 0;
}