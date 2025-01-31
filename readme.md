# **Genetic Algorithm for Solving the Traveling Salesman Problem (TSP)**

## **Description**
This project implements a **Genetic Algorithm (GA)** to solve the **Traveling Salesman Problem (TSP)** and compares its performance against a **Greedy Algorithm** and **Random Search**. The project follows a structured approach, including dataset parsing, fitness evaluation, and evolutionary optimization.

## **Features**
- **TSP Data Parsing**: Reads `.tsp` files and extracts city coordinates.
- **Distance Calculation**: Computes Euclidean distances between cities.
- **Solution Representation**: Uses a permutation-based encoding to represent city routes.
- **Fitness Evaluation**: Calculates the total distance of a given tour.
- **Greedy Algorithm**: Implements a nearest-neighbor heuristic.
- **Genetic Algorithm Components**:
  - **Selection**: Tournament-based parent selection.
  - **Crossover**: Partially Mapped Crossover (PMX) for solution recombination.
  - **Mutation**: Swap mutation combined with a 2-opt local search.
  - **Elitism**: Retains the best solutions in each generation.
- **Performance Analysis**:
  - Compares GA, Greedy Algorithm, and Random Search.
  - Tests conducted on **berlin11, berlin52, kroA100, and kroA150** datasets.
  - Fitness progress and optimal routes visualized.