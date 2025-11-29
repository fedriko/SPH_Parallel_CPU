# Real-Time SPH Fluid Simulation

## Project Overview

The goal of this project is to create an interactable fluid that runs in real-time using Smoothed Particle Hydrodynamics (SPH).

## Implemented Models and Methods

### SPH Models

We implemented two distinct SPH models:

1.  **Single-Pass Model:** Computes all forces in a single pass for better performance (ideal for games).
2.  **Predictive-Corrective Model:** A more physically robust approach that computes viscosity and external forces to predict velocity, then applies pressure forces to correct for fluid compressions.

### State Equations for Density

We implement two different state equations to compute densities:

*   **The Tait Equation:** Provides a more reactive fluid (exponential response to densities). 
      It is more prone to chaotic behavior but works well with density and velocity damping. Chaotic behaviors generally arrive when the user violently compresses or rotates the confinement box.

*   **A Linear Solver:** Generally more robust under heavy flows and provides a less violent response to compression.

### Numerical Integration (Symplectic Euler)

We use the Symplectic Euler method for numerical integration.

This is the algorithm of choice for oscillatory systems in physics simulations because, while it doesn't perfectly conserve energy at every time step, its energy error remains bounded and oscillates around the true initial energy value over long periods (by preserving the system's symplectic form).

Even though the simulation uses dissipative viscosity forces, their overall impact is relatively low in this low viscosity fluid simulation. For this reason, Symplectic Euler remains a good choice for this kind of application.

## Performance & Implementation Details 

### Uniform CSR Grid
I implemented a Uniform Grid for neighbor searching, using a data structure optimized for contiguous memory access, which is crucial for performance in a real-time fluid simulation.
Unlike a fixed grid, this approach allows the bounds of the simulation box to change dynamically in real-time without penalty. The structure is highly cache-efficient because elements are stored in a linear array. 
This data locality dramatically speeds up calculations during force computations, as the CPU can access neighboring particle data much faster.

### Parallelization and Burst compiler (unity)
I also leveraged multi-threading via a job system, ensuring most of the mathematics is compatible with the Unity Burst Compiler to maximize performance.
The system uses two main linear arrays: one for the grid cells and another that stores the particle indices within each cell. 
By setting the cell size equal to the kernel's support radius, we only need to check adjacent cells to find all necessary neighbors (27 neighbors cells).
This was the bigest gain in performance from my initial naive aproach.



## XSPH implementation (NON DISIPATIVE)
"An improved particle method for incompressible fluid flow" by J. J. Monaghan, published in the Journal of Computational Physics in 1989. 
