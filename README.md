# CUDA-MonteCarlo
Demonstration of Monte Carlo simulation(s) using CUDA. A basic simulation to estimate the value of Pi is performed with CUDA, OpenMP, and non-parallel C code.
A Monte Carlo simulation to estimate the price of a European Call Option is performed using CUDA.

# Results
The Pi simulations were ran and benchmarked on a Haswell-EP machine with an Intel(R) Xeon(R) CPU E5-2640 v3 and NVIDIA Quadro P400.
Results are as follows:
| Simulation    |   Estimation of Pi    | Time      |
| --------      | -------------------   | -----     |
| Serial        |   3.141350            | 3.629 s   |
| OpenMP        |   3.141700            |  94 ms    |
| CUDA          |    3.141350           | 28.367 ms |