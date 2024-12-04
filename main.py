import pandas as pd
import math
import random

# Open and read the TSP file
with open('berlin11_modified.tsp', 'r') as infile:
    # Initialize header variables
    name = None
    file_type = None
    comment = None
    dimension = None
    edge_weight_type = None

    # Dynamically read the header section
    while True:
        line = infile.readline().strip()
        if not line or line.startswith('NODE_COORD_SECTION'):
            break  # Stop reading when the node section starts
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

    # Validate the headers
    if None in [name, file_type, comment, dimension, edge_weight_type]:
        raise ValueError("Missing header information in the TSP file!")

    # Read the city coordinates
    node_list = []
    for _ in range(dimension):
        line = infile.readline().strip().split()
        city_id = int(line[0])
        x = float(line[1])
        y = float(line[2])
        node_list.append([city_id, x, y])

# Load the city data into a pandas DataFrame
df = pd.DataFrame(node_list, columns=['City_ID', 'X_Coordinate', 'Y_Coordinate'])

# Function to calculate Euclidean distance between two cities
def calculate_distance(city1, city2):
    """Calculate the Euclidean distance between two cities."""
    x1, y1 = city1['X_Coordinate'], city1['Y_Coordinate']
    x2, y2 = city2['X_Coordinate'], city2['Y_Coordinate']
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to create a distance matrix
def create_distance_matrix(df):
    """Create a matrix containing distances between all cities."""
    distance_matrix = pd.DataFrame(index=df['City_ID'], columns=df['City_ID'], dtype=float)
    for city1_id in df['City_ID']:
        for city2_id in df['City_ID']:
            if city1_id != city2_id:
                city1 = df[df['City_ID'] == city1_id].iloc[0]
                city2 = df[df['City_ID'] == city2_id].iloc[0]
                distance = calculate_distance(city1, city2)
                distance_matrix.loc[city1_id, city2_id] = distance
            else:
                distance_matrix.loc[city1_id, city2_id] = 0  # Distance to itself is 0
    return distance_matrix

# Function to calculate the fitness of a solution
def calculate_fitness(solution, distance_matrix):
    """Calculate the total distance of a solution (fitness)."""
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += distance_matrix.loc[solution[i], solution[i + 1]]
    total_distance += distance_matrix.loc[solution[-1], solution[0]]  # Return to start
    return total_distance

# Function to generate a random solution
def generate_random_solution(df):
    """Generate a random solution with all cities."""
    solution = df['City_ID'].tolist()
    random.shuffle(solution)
    return solution

# Greedy algorithm for solving the TSP
def greedy_algorithm(df):
    """Solve the TSP using a greedy algorithm."""
    unvisited = df['City_ID'].tolist()
    current_city_id = unvisited.pop(0)  # Start with the first city
    visited = [current_city_id]
    total_distance = 0

    while unvisited:
        nearest_city_id = None
        nearest_distance = float('inf')

        current_city = df[df['City_ID'] == current_city_id].iloc[0]
        for city_id in unvisited:
            city = df[df['City_ID'] == city_id].iloc[0]
            distance = calculate_distance(current_city, city)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_city_id = city_id

        visited.append(nearest_city_id)
        total_distance += nearest_distance
        current_city_id = nearest_city_id
        unvisited.remove(nearest_city_id)

    # Add the distance back to the starting city
    total_distance += calculate_distance(df[df['City_ID'] == current_city_id].iloc[0],
                                         df[df['City_ID'] == visited[0]].iloc[0])
    return visited, total_distance

# Create the distance matrix
distance_matrix = create_distance_matrix(df)

# Generate a random solution and calculate its fitness
random_solution = generate_random_solution(df)
random_fitness = calculate_fitness(random_solution, distance_matrix)

# Run the greedy algorithm
greedy_solution, greedy_fitness = greedy_algorithm(df)

# Print results
print(f"File Name: {name}")
print(f"File Type: {file_type}")
print(f"Comment: {comment}")
print(f"Number of Cities (Dimension): {dimension}")
print(f"Edge Weight Type: {edge_weight_type}")
print("\nRandom Solution:")
print("City Sequence:", " -> ".join(map(str, random_solution)))
print(f"Total Distance (Fitness): {random_fitness}")
print("\nGreedy Algorithm Solution:")
print("City Sequence:", " -> ".join(map(str, greedy_solution)))
print(f"Total Distance (Fitness): {greedy_fitness}")
