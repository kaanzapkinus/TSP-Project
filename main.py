import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np


# TSP dosyasını oku ve DataFrame döndür
def parse_tsp_file(file_path):
    with open(file_path, "r") as infile:
        name, file_type, comment, dimension, edge_weight_type = (
            None,
            None,
            None,
            None,
            None,
        )

        while True:
            line = infile.readline().strip()
            if not line or line.startswith("NODE_COORD_SECTION"):
                break
            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("TYPE"):
                file_type = line.split(":")[1].strip()
            elif line.startswith("COMMENT"):
                comment = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip()

        city_list = []
        for _ in range(dimension):
            line = infile.readline().strip().split()
            city_id = int(line[0])
            x_coord = float(line[1])
            y_coord = float(line[2])
            city_list.append([city_id, x_coord, y_coord])

        dataframe = pd.DataFrame(
            city_list, columns=["City_ID", "X_Coordinate", "Y_Coordinate"]
        )
        return dataframe, name, file_type, comment, dimension, edge_weight_type


# difference between 2 cities
def calculate_distance(city1, city2):
    return math.sqrt(
        (city2["X_Coordinate"] - city1["X_Coordinate"]) ** 2
        + (city2["Y_Coordinate"] - city1["Y_Coordinate"]) ** 2
    )


# a matrix storing the distances between all cities
def create_distance_matrix(dataframe):
    city_ids = dataframe["City_ID"].values
    city_to_idx = {cid: i for i, cid in enumerate(city_ids)}
    coords = dataframe[["X_Coordinate", "Y_Coordinate"]].values
    num_cities = len(city_ids)

    distance_matrix = np.zeros((num_cities, num_cities), dtype=float)
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dist = math.sqrt(
                (coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix, city_to_idx


# fitness calculator
def calculate_fitness(solution, distance_matrix, city_to_idx):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += distance_matrix[
            city_to_idx[solution[i]], city_to_idx[solution[i + 1]]
        ]
    total_distance += distance_matrix[
        city_to_idx[solution[-1]], city_to_idx[solution[0]]
    ]
    return total_distance


# greedy algorithm
def greedy_algorithm(dataframe, start_city_id=1):
    unvisited = dataframe["City_ID"].tolist()
    unvisited.remove(start_city_id)
    current_city_id = start_city_id
    visited = [current_city_id]
    total_distance = 0

    while unvisited:
        current_city = dataframe[dataframe["City_ID"] == current_city_id].iloc[0]
        nearest_city_id = min(
            unvisited,
            key=lambda cid: calculate_distance(
                current_city, dataframe[dataframe["City_ID"] == cid].iloc[0]
            ),
        )
        nearest_city = dataframe[dataframe["City_ID"] == nearest_city_id].iloc[0]
        total_distance += calculate_distance(current_city, nearest_city)
        visited.append(nearest_city_id)
        unvisited.remove(nearest_city_id)
        current_city_id = nearest_city_id

    total_distance += calculate_distance(
        dataframe[dataframe["City_ID"] == current_city_id].iloc[0],
        dataframe[dataframe["City_ID"] == visited[0]].iloc[0],
    )
    return visited, total_distance


# PMX Crossover
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    point1, point2 = sorted(random.sample(range(size), 2))

    child[point1 : point2 + 1] = parent1[point1 : point2 + 1]

    for i in range(point1, point2 + 1):
        if parent2[i] not in child:
            val = parent2[i]
            idx = i
            while True:
                corresponding_val = parent1[idx]
                idx = parent2.index(corresponding_val)
                if child[idx] is None:
                    break
            child[idx] = val

    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]

    return child


# swap mutation function
def mutation_function(individual, mutation_probability):
    if random.random() < mutation_probability:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


# precise mutation
def precise_mutation(individual, distance_matrix, city_to_idx):
    best_fitness = calculate_fitness(individual, distance_matrix, city_to_idx)
    best_individual = individual.copy()

    for i in range(len(individual) - 1):
        for j in range(i + 1, len(individual)):
            mutated = individual.copy()
            mutated[i : j + 1] = reversed(mutated[i : j + 1])
            fitness = calculate_fitness(mutated, distance_matrix, city_to_idx)
            if fitness < best_fitness:
                best_fitness = fitness
                best_individual = mutated
    return best_individual


# tournament selection
def tournament_selection(population, tournament_size=5):
    tournament = population.sample(n=tournament_size)
    best_index = tournament["Fitness"].values.argmin()
    best_individual = tournament.iloc[best_index]
    return {
        "Solution": best_individual["Solution"],
        "Fitness": best_individual["Fitness"],
    }


# epoch
def create_new_epoch(
    previous_population,
    distance_matrix,
    city_to_idx,
    mutation_probability,
    crossover_probability,
    pop_size,
    dataframe,
):
    new_population = []
    
    #elite selection
    elite_count = max(1, int(0.10 * pop_size)) # 10% of the population
    best_individuals = previous_population.nsmallest(elite_count, "Fitness")
    new_population.extend(best_individuals.to_dict("records"))

    for _ in range(pop_size - elite_count):
        parent1 = tournament_selection(previous_population)["Solution"]
        parent2 = tournament_selection(previous_population)["Solution"]

        if random.random() < crossover_probability:
            child = pmx_crossover(parent1, parent2)
        else:
            child = parent1.copy()

        mutated_child = mutation_function(child, mutation_probability)
        precise_child = precise_mutation(mutated_child, distance_matrix, city_to_idx)
        child_fitness = calculate_fitness(precise_child, distance_matrix, city_to_idx)

        new_population.append({"Solution": precise_child, "Fitness": child_fitness})

    return pd.DataFrame(new_population)


# create population
def create_population(dataframe, num_individuals):
    population_data = []
    city_ids = dataframe["City_ID"].tolist()

    for _ in range(num_individuals):
        individual = random.sample(city_ids, len(city_ids))
        population_data.append({"Solution": individual, "Fitness": None})

    return pd.DataFrame(population_data)


if __name__ == "__main__":
    random.seed(random.randint(50, 100))
    dataframe, name, file_type, comment, dimension, edge_weight_type = parse_tsp_file(
        "berlin52.tsp"
    )
    distance_matrix, city_to_idx = create_distance_matrix(dataframe)

    population = create_population(dataframe, num_individuals=100)
    population["Fitness"] = population["Solution"].apply(
        lambda sol: calculate_fitness(sol, distance_matrix, city_to_idx)
    )

    #settings for the genetic algorithms
    num_epochs = 15
    crossover_probability = 0.6
    pop_size = 200
    mutation_probability = max(0.1, 0.4 * (1 - epoch / num_epochs))
    
    best_fitness_over_time = []
    
    # Matplotlib için başlangıç ayarları
    plt.ion()  # Interactive mode enabled
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))  # Sadece fitness tablosu

    # Fitness grafiği başlangıç ayarları
    ax1.set_title(f"Real-Time Fitness Progress ({dimension} Cities - {name})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Fitness")
    initial_fitness = population["Fitness"].min()
    ax1.set_xlim(1, num_epochs)
    ax1.set_ylim(0, initial_fitness * 1.2)

    # Epoch döngüsü
    for epoch in range(num_epochs):
        mutation_probability = mutation_probability
        population = create_new_epoch(
            previous_population=population,
            distance_matrix=distance_matrix,
            city_to_idx=city_to_idx,
            mutation_probability=mutation_probability,
            crossover_probability =crossover_probability,
            pop_size=pop_size,
            dataframe=dataframe,
        )
        best_fitness = population["Fitness"].min()
        best_fitness_over_time.append(best_fitness)
    
        # Fitness tablosunu her epoch'ta güncelle
        ax1.clear()
        ax1.set_title(f"Real-Time Fitness Progress ({dimension} Cities - {name})")
        ax1.set_xlabel(f"Epoch (Best Fitness: {best_fitness:.2f})")
        ax1.set_ylabel("Fitness")
        ax1.set_xlim(1, num_epochs)
        ax1.set_ylim(0, initial_fitness * 1.2)
        ax1.plot(
            range(1, len(best_fitness_over_time) + 1),
            best_fitness_over_time,
            color="teal",
        )

        # Fitness tablosunu güncelle ve göster
        plt.draw()
        plt.pause(0.1)

        print(f"Epoch {epoch + 1}/{num_epochs}, Best Fitness: {best_fitness}")

    # Epoch döngüsü tamamlandıktan sonra rota tablosu oluştur
    plt.ioff()  # Interactive mode off

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # İki grafik yan yana
    ax1.set_title(f"Final Fitness Progress ({dimension} Cities - {name})")
    ax1.set_xlabel(f"Epoch (Best Fitness: {best_fitness:.2f})")
    ax1.set_ylabel("Fitness")
    ax1.set_xlim(1, num_epochs)
    ax1.set_ylim(0, initial_fitness * 1.2)
    ax1.plot(
        range(1, len(best_fitness_over_time) + 1),
        best_fitness_over_time,
        color="teal",
    )

    # Rota tablosunu çiz
    ax2.set_title("Best Route")
    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")
    ax2.grid(True)

    best_solution = population.loc[population["Fitness"].idxmin()]
    best_route = best_solution["Solution"]
    coords = (
        dataframe.set_index("City_ID")
        .loc[best_route][["X_Coordinate", "Y_Coordinate"]]
        .values
    )
    coords = np.vstack([coords, coords[0]])
    ax2.plot(coords[:, 0], coords[:, 1], marker="o", linestyle="-", color="b")

    for i, city_id in enumerate(best_route):
        ax2.text(
            coords[i, 0],
            coords[i, 1],
            str(city_id),
            fontsize=9,
            ha="right",
            va="bottom",
            color="red",
        )

    # Son tabloyu kaydet ve göster
    plt.tight_layout()
    final_fitness = round(best_fitness_over_time[-1], 2)
    plot_filename = f"{name}_{num_epochs}Epochs_Crossover={crossover_probability}_PopSize={pop_size}_BestFitness={final_fitness}.png"
    plt.savefig(plot_filename)
    plt.show()

    # Greedy çözümlerin en sonda yazılması
    print("\nGreedy Algorithm Solutions (After GA):")
    greedy_solution, greedy_fitness = greedy_algorithm(dataframe, start_city_id=1)
    print(f"Greedy Solution: {greedy_solution}")
    print(f"Greedy Fitness: {greedy_fitness}\n")
    
    print("Best Route (City IDs):", best_route)
    print("Best Fitness (Total Distance):", best_solution["Fitness"])


