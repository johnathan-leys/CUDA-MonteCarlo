/*

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


// Global
unsigned long long num_points = 100000000; // Currently 1/10th the number of points as in CUDA example, otherwise takes incredibly long.
unsigned long long points_in_circle = 0;


int main(int argc, char **argv) {

    // Start timing
    double start = omp_get_wtime();

    #pragma omp parallel reduction(+:points_in_circle)  // Reduction, add values from each thread
    {
        unsigned int seed = (unsigned int)time(NULL) + omp_get_thread_num();    // This time seed with thread number. 
        srand(seed);                                                            // Not rubustly thread safe but should work for demo

        // Generate random points and count those inside the unit circle
        #pragma omp for
        for (int i = 0; i < num_points; i++) {
            double x = (double)rand() / RAND_MAX;
            double y = (double)rand() / RAND_MAX;
            double distance = x * x + y * y;
            if (distance <= 1.0) {
                points_in_circle++;
            }
        }
    }

    // Stop timing
    double end = omp_get_wtime();
    double time_taken = (end - start);

    double pi_estimate = 4.0 * points_in_circle / (double)num_points;
    printf("Estimated value of pi: %lf\n", pi_estimate);
    printf("Calculation time: %.3f seconds\n", time_taken);

    return 0;
}