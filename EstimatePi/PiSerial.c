/*
Serial Code that runs the same simulation for Pi on CPU. Take much longer to run,
>30 Seconds for me. Reduce the nuber of points to get faster runtime.
Meant to demonstrate just how fast CUDA can speed up certain processes.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    unsigned long long num_points = 1000000000; 
    unsigned long long points_in_circle = 0;

    // Seed the random number generator
    srand(time(NULL));

    // Start timing
    clock_t start = clock();

    // Generate random points and count those inside the unit circle
    for (int i = 0; i < num_points; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        double distance = x * x + y * y;
        if (distance <= 1.0) {
            points_in_circle++;
        }
    }

    // Stop timing
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    double pi_estimate = 4.0 * points_in_circle / (double)num_points;
    printf("Estimated value of pi: %lf\n", pi_estimate);
    printf("Calculation time: %.3f seconds\n", time_taken);

    return 0;
}