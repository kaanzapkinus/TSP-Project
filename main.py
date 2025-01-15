import pandas as pd
import math
import random

# Function to parse TSP file and return DataFrame and metadata
def parse_tsp_file(file_path):
    with open(file_path, 'r') as infile:
        name, file_type, comment, dimension, edge_weight_type = None, None, None, None, None

        while True:
            line = infile.readline().strip()
            if not line or line.startswith('NODE_COORD_SECTION'):
                break
            if line.startswith('NAME'):
                name = line.split(':')[1].strip()
            elif line.startswith('TYPE'):
                file_type = line.split(':')[1].strip()
            elif line.startswith('COMMENT'):
                comment = line.split(':')[1].strip()
            elif line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                edge_weight_type = line.split(':')[1].strip()

        if None in [name, file_type, comment, dimension, edge_weight_type]:
            raise ValueError("Missing header information in the TSP file!")

        city_list = []
        for _ in range(dimension):
            line = infile.readline().strip().split()
            city_list.append([int(line[0]), float(line[1]), float(line[2])])

        dataframe = pd.DataFrame(city_list, columns=['City_ID', 'X_Coordinate', 'Y_Coordinate'])
        return dataframe, name, file_type, comment, dimension, edge_weight_type

# Function to calculate Euclidean distance between two cities
def calculate_distance(city1, city2):
    return math.sqrt((city2['X_Coordinate'] - city1['X_Coordinate'])**2 +
                     (city2['Y_Coordinate'] - city1['Y_Coordinate'])**2)
    
# Function to create distance matrix
def create_distance_matrix(dataframe):
    distance_matrix = pd.DataFrame(index=dataframe['City_ID'], columns=dataframe['City_ID'], dtype=float)
    for city1_id in dataframe['City_ID']:
        for city2_id in dataframe['City_ID']:
            if city1_id != city2_id:
                city1 = dataframe[dataframe['City_ID'] == city1_id].iloc[0]
                city2 = dataframe[dataframe['City_ID'] == city2_id].iloc[0]
                distance_matrix.loc[city1_id, city2_id] = calculate_distance(city1, city2)
            else:
                distance_matrix.loc[city1_id, city2_id] = 0
    return distance_matrix

# Function to calculate fitness of a solution
def calculate_fitness(solution, distance_matrix):
    total_distance = sum(distance_matrix.loc[solution[i], solution[i + 1]] for i in range(len(solution) - 1))
    total_distance += distance_matrix.loc[solution[-1], solution[0]]  # Return to start
    return total_distance

# Function to generate a random solution
def generate_random_solution(dataframe):
    solution = dataframe['City_ID'].tolist()
    random.shuffle(solution)
    return solution

# Greedy algorithm for solving the TSP
def greedy_algorithm(dataframe, start_city=0):
    unvisited = dataframe['City_ID'].tolist()
    current_city_id = unvisited.pop(start_city)
    visited = [current_city_id]
    total_distance = 0

    while unvisited:
        current_city = dataframe[dataframe['City_ID'] == current_city_id].iloc[0]
        nearest_city_id = min(unvisited, key=lambda city_id: calculate_distance(
            current_city, dataframe[dataframe['City_ID'] == city_id].iloc[0]))
        nearest_city = dataframe[dataframe['City_ID'] == nearest_city_id].iloc[0]
        total_distance += calculate_distance(current_city, nearest_city)
        visited.append(nearest_city_id)
        current_city_id = nearest_city_id
        unvisited.remove(nearest_city_id)

    total_distance += calculate_distance(dataframe[dataframe['City_ID'] == current_city_id].iloc[0],
                                         dataframe[dataframe['City_ID'] == visited[0]].iloc[0])
    return visited, total_distance

# Function to perform tournament selection
def tournament_selection(population, tournament_size):
    """
    Perform tournament selection to choose one individual from the population.

    Args:
        population (pd.DataFrame): The population DataFrame with 'Solution' and 'Fitness' columns.
        tournament_size (int): The number of individuals to include in the tournament.

    Returns:
        dict: The selected individual as a dictionary with 'Solution' and 'Fitness'.
    """
    # Randomly select individuals for the tournament
    tournament = population.sample(n=tournament_size)

    # Find the individual with the best (lowest) fitness
    best_individual = tournament.loc[tournament['Fitness'].idxmin()]

    # Return the selected individual as a dictionary
    return {'Solution': best_individual['Solution'], 'Fitness': best_individual['Fitness']}

#cycle crossover
def pmx_crossover(parent1, parent2):
    """
    Perform Partially Mapped Crossover (PMX) on two parents to generate a child.

    Args:
        parent1 (list): The first parent solution.
        parent2 (list): The second parent solution.

    Returns:
        list: The child solution.
    """
    size = len(parent1)
    child = [None] * size

    # Select two random crossover points
    point1, point2 = sorted(random.sample(range(size), 2))

    # Copy the segment from parent1 to the child
    child[point1:point2 + 1] = parent1[point1:point2 + 1]

    # Map the values from parent2 to child, ensuring no duplicates
    for i in range(point1, point2 + 1):
        if parent2[i] not in child:
            val = parent2[i]
            idx = i

            # Find the corresponding position in parent1
            while True:
                corresponding_val = parent1[idx]
                idx = parent2.index(corresponding_val)
                if child[idx] is None:
                    break

            # Place the value in the correct position
            child[idx] = val

    # Fill in the remaining positions from parent2
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]

    return child

# mutation function
def mutation_function(individual, mutation_probability=0.1, mutation_type='swap'):
    """
    Apply mutation to a given solution (individual).

    Args:
        individual (list): A solution (list of city IDs) to mutate.
        mutation_probability (float): Probability of mutation occurring.
        mutation_type (str): Type of mutation ('swap' or 'inversion').

    Returns:
        list: Mutated individual (or the same individual if no mutation occurs).
    """
    mutated_individual = individual.copy()

    # Apply mutation based on probability
    

    if random.random() < mutation_probability:
        if mutation_type == 'swap':
            # Swap Mutation: Swap two random cities
            idx1, idx2 = random.sample(range(len(mutated_individual)), 2)
            mutated_individual[idx1], mutated_individual[idx2] = mutated_individual[idx2], mutated_individual[idx1]
        elif mutation_type == 'inversion':
            # Inversion Mutation: Reverse a random segment of the solution
            start, end = sorted(random.sample(range(len(mutated_individual)), 2))
            mutated_individual[start:end + 1] = reversed(mutated_individual[start:end + 1])
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

    return mutated_individual

    return mutated_individual
# Function to create initial population
def create_population(dataframe, num_individuals, include_greedy=True, greedy_function=True):
    """
    Create an initial population of solutions.
    """
    population_data = []
    city_ids = dataframe['City_ID'].tolist()

    # Add the greedy solution if required
    if include_greedy and greedy_function: 
        greedy_solution, greedy_fitness = greedy_function(dataframe)
        population_data.append({'Solution': greedy_solution, 'Fitness': greedy_fitness})
        num_individuals -= 1

    # Create random solutions for the rest of the population
    for _ in range(num_individuals):
        random.shuffle(city_ids)
        population_data.append({'Solution': city_ids.copy(), 'Fitness': None})

    # Return the population as a DataFrame
    return pd.DataFrame(population_data)

# Function to print population information
def print_population_info(population):
    if 'Fitness' not in population or population['Fitness'].isnull().all():
        print("No fitness values available in the population.")
        return

    size = len(population)
    best_score = population['Fitness'].min()
    median_score = population['Fitness'].median()
    worst_score = population['Fitness'].max()

    print("Population Information:")
    print(f" - Population size: {size}")
    print(f" - Best fitness score: {best_score}")
    print(f" - Median fitness score: {median_score}")
    print(f" - Worst fitness score: {worst_score}")

def create_new_epoch(previous_population, distance_matrix, mutation_probability=0.1, crossover_probability=0.8, pop_size=50):
    """
    Create a new population (epoch) from the previous population.

    Args:
        previous_population (pd.DataFrame): Current population with 'Solution' and 'Fitness'.
        distance_matrix (pd.DataFrame): Distance matrix for calculating fitness.
        mutation_probability (float): Probability of mutating a child.
        crossover_probability (float): Probability of performing crossover.
        pop_size (int): The desired size of the new population.

    Returns:
        pd.DataFrame: The new population with updated solutions and fitness values.
    """
    new_population = []
    
    # Ensure the best solution is carried forward (elitism)
    best_individual = previous_population.loc[previous_population['Fitness'].idxmin()]
    new_population.append({'Solution': best_individual['Solution'], 'Fitness': best_individual['Fitness']})

    while len(new_population) < pop_size:
        # Select parents
        parent1 = tournament_selection(previous_population, tournament_size=5)['Solution']
        parent2 = tournament_selection(previous_population, tournament_size=5)['Solution']

        # Perform crossover based on probability
        if random.random() < crossover_probability:
            child = pmx_crossover(parent1, parent2)
        else:
            child = parent1.copy()  # No crossover, copy parent1 as child

        # Apply mutation
        mutated_child = mutation_function(child, mutation_probability)

        # Calculate fitness for the mutated child
        child_fitness = calculate_fitness(mutated_child, distance_matrix)

        # Add the child to the new population
        new_population.append({'Solution': mutated_child, 'Fitness': child_fitness})

    # Convert to DataFrame
    return pd.DataFrame(new_population)

# Main logic
if __name__ == "__main__":
    dataframe, name, file_type, comment, dimension, edge_weight_type = parse_tsp_file('berlin52.tsp')

    distance_matrix = create_distance_matrix(dataframe)
    random_solution = generate_random_solution(dataframe)
    random_fitness = calculate_fitness(random_solution, distance_matrix)

    greedy_solution, greedy_fitness = greedy_algorithm(dataframe)

    # Include greedy solution in the population
    population = create_population(dataframe, num_individuals=25, include_greedy=True, greedy_function=greedy_algorithm)
    population['Fitness'] = population['Solution'].apply(lambda sol: calculate_fitness(sol, distance_matrix))
    selected_individual = tournament_selection(population, tournament_size=20)

    # Popülasyonu fitness değerine göre sırala
    sorted_population = population.sort_values(by="Fitness")

    # En iyi iki ebeveyni seç
    parent1 = sorted_population.iloc[0]["Solution"]
    parent2 = sorted_population.iloc[1]["Solution"]
    
    # Cycle Crossover ile yeni bir çocuk oluştur
    child = pmx_crossover(parent1, parent2)
    mutated_child = mutation_function(child, mutation_probability=0.2, mutation_type='swap')

    # Çocuğun fitness değerini hesapla
    child_fitness = calculate_fitness(child, distance_matrix)
    mutated_child_fitness = calculate_fitness(mutated_child, distance_matrix)
    
    # Çocuğu popülasyona ekle
    child_df = pd.DataFrame([{'Solution': mutated_child, 'Fitness': child_fitness}])
    population = pd.concat([population, child_df], ignore_index=True)    

    # İlk popülasyonu oluştur
    population = create_population(dataframe, num_individuals=50, include_greedy=1, greedy_function=greedy_algorithm)
    population['Fitness'] = population['Solution'].apply(lambda sol: calculate_fitness(sol, distance_matrix))

    # Epoch döngüsü başlat
    num_epochs = 1000
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        population = create_new_epoch(population, distance_matrix, mutation_probability=0.3, pop_size=100)
        print_population_info(population)

    # En iyi sonucu yazdır
    best_solution = population.loc[population['Fitness'].idxmin()]
    print("Best Solution:", best_solution)

    # Popülasyon bilgisini yazdır
    print_population_info(population)

    #Print results
    print(f"File Name: {name}")
    print(f"File Type: {file_type}")
    print(f"Comment: {comment}")
    print(f"Number of Cities (Dimension): {dimension}")
    print(f"Edge Weight Type: {edge_weight_type}\n")

    print("Random Solution:")
    print(f" - City Sequence: {random_solution}")
  
    print("Greedy Algorithm Solution:")
    print(f" - City Sequence: {greedy_solution}")
    
    # Print the child solution
    print("Cycle Crossover Result:")
    print(f"Parent 1: {parent1}\n")
    print(f"Parent 2: {parent2}\n")
    print(f"Child: {child}\n")
    
    print(f" - Random Fitness: {random_fitness}\n")
    print(f" - Greedy Fitness: {greedy_fitness}\n")
    print(f" - Child Fitness (Cycle Crossover ): {child_fitness}\n")
    print(f" - Mutated Child Fitness (Mutation): {mutated_child_fitness}\n")
    
    print("Tournament Selection Result:")
    print(f" - Selected Solution: {selected_individual['Solution']}")
    print(f" - Selected Fitness: {selected_individual['Fitness']}\n")
    
    print("Best Solution Across All Epochs:")
    print(best_solution)
    
    print_population_info(population)