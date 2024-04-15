# CUDA-MonteCarlo
Demonstration of Monte Carlo simulation(s) using CUDA. A basic simulation to estimate the value of Pi is performed with CUDA, OpenMP, and non-parallel C code.
A Monte Carlo simulation to estimate the price of a European Call Option is performed using CUDA.

# Results
## Pi Estimations
The Pi estimation programs are rather simple
The Pi simulations were ran and benchmarked on a Haswell-EP machine with an Intel Xeon CPU E5-2640 v3 and NVIDIA Quadro P400.
Results are as follows with 100000000 points simulated:
| Method        |   Estimation of Pi    | Time      |
| --------      | -------------------   | -----     |
| Serial        |   3.141350            | 3.629 s   |
| OpenMP        |   3.141700            |  94 ms    |
| CUDA          |   3.141350            | 28.367 ms |

CUDA is clearly the fastest, which is expected due to the 256 CUDA cores on the P400 vs the 8 on the CPU.
When increasing the number of points 10x, results were as follows:

| Method        |   Estimation of Pi    | Time      |
| --------      | -------------------   | -----     |
| Serial        |   3.141627            | 33.811 s  |
| OpenMP        |   3.141557            | 951 ms    |
| CUDA          |   3.141618            | 126.182 ms|


OpenMP takes ~10x as long, which is linear and expected, but the CUDA code only takes ~4.5 times as long, which 
could be due to grid/block layout being better optimized or something in the managed/shared memory.

For comparison, I ran the CUDA code on my desktop with a GTX 1070 Ti, which has 2432 CUDA cores, 9.5x as many as the P400.
This resulted in a calculation time of 3.647 milliseconds, about a 7.75x speedup.

## Option Pricing
The European Call Option program simulates many paths representing potential stock price movements, and calculates the potential payoff (profit or $0) of the call at expiration. After simulating these paths, the payoffs are averaged and adjusted (with risk free interest rate) to estimate the "fair value" of the option. This is a common way to estimate fair value of a stock option.

This is a more complex simulation, and involves each thread running a loop, so I was interested to see the results.
Running the simulation with 1M paths results in an execution time of 39.9639 ms on the 1070 Ti and 383.134 ms on the Quadro P400, both getting the same result of 
$8.03522 for the call price.