import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np

COLOR_PALETTE = {
    'main': "#3498DB",
    'secondary': "#E74C3C",
    'background': "#F8F9F9",
    'text': "#2C3E50",
    'accent': "#1ABC9C"
}


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
def tournament_selection(population, tournament_size=5): # tournament size
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
    dataframe, name, file_type, comment, dimension, edge_weight_type = parse_tsp_file("berlin52.tsp")
    distance_matrix, city_to_idx = create_distance_matrix(dataframe)

    greedy_solution, greedy_fitness = greedy_algorithm(dataframe, start_city_id=1)
    population = create_population(dataframe, num_individuals=100)
    population["Fitness"] = population["Solution"].apply(
        lambda sol: calculate_fitness(sol, distance_matrix, city_to_idx)
    )

    # Genetic algorithm settings
    num_epochs = 10
    crossover_probability = 0.6
    pop_size = 100
    mutation_probability = max(0.1, 0.4 * num_epochs)
    
    best_fitness_over_time = []
    
    # Real-time plotting setup
    plt.ion()
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0])

    # Parametre metni
    settings_text = (
        "⚙️ GA Parameters:\n"
        f"• Epochs: {num_epochs}\n"
        f"• Crossover: {crossover_probability}\n"
        f"• Population: {pop_size}\n"
        f"• Mutation: {mutation_probability:.2f}\n"
        f"• Greedy Solution Fitness: {greedy_fitness:.2f}\n"
        f"• Initial Best Fitness: {population['Fitness'].min():.2f}"
    )

    # Başlangıç grafik konfigürasyonu
    ax1.set_title(f"Real-Time Optimization: {name} ({dimension} Cities)", color=COLOR_PALETTE['main'], pad=20)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Fitness Value", fontsize=12)
    ax1.set_xlim(1, num_epochs)
    ax1.set_facecolor(COLOR_PALETTE['background'])
    initial_fitness = None  # Zoom kontrolü için

    # Optimizasyon döngüsü
    for epoch in range(num_epochs):
        population = create_new_epoch(
            previous_population=population,
            distance_matrix=distance_matrix,
            city_to_idx=city_to_idx,
            mutation_probability=mutation_probability,
            crossover_probability=crossover_probability,
            pop_size=pop_size,
            dataframe=dataframe,
        )
        best_fitness = population["Fitness"].min()
        best_fitness_over_time.append(best_fitness)
        
        # İlk fitness değerini kaydet
        if initial_fitness is None:
            initial_fitness = best_fitness

        # Grafik güncelleme
        ax1.clear()
        current_title = f"Epoch: {epoch+1}/{num_epochs} | Best Fitness: {best_fitness:.2f}"
        ax1.set_title(current_title, color=COLOR_PALETTE['main'], pad=20)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Fitness Value", fontsize=12)
        ax1.set_xlim(1, num_epochs)
            
        settings_text = (
        "⚙️ GA Parameters:\n"
        f"• Epochs: {num_epochs}\n"
        f"• Crossover: {crossover_probability}\n"
        f"• Population: {pop_size}\n"
        f"• Mutation: {mutation_probability:.2f}\n"
        f"• Greedy Solution Fitness: {greedy_fitness:.2f}\n"
        f"• Initial Best Fitness: {best_fitness:.2f}\n"
        f"• Genetic Fitness vs Greedy Fitness {((greedy_fitness - best_fitness)/greedy_fitness)*100:.1f}%" 

    )
        # Dinamik y-ekseni sınırları
        y_lower = best_fitness * 0.95
        y_upper = initial_fitness * 1.05
        ax1.set_ylim(y_lower, y_upper)
        ax1.set_facecolor(COLOR_PALETTE['background'])
        
        ax1.text(
            0.68, 0.97, settings_text,
            transform=ax1.transAxes,
            ha='left', va='top',
            fontsize=11,
            linespacing=1.5,
            bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=COLOR_PALETTE['background'],
            edgecolor=COLOR_PALETTE['main'],
            alpha=0.95
            )
        )
        
        # Gelişmiş çizgi grafiği
        ax1.fill_between(
            range(1, len(best_fitness_over_time) + 1),
            best_fitness_over_time,
            color=COLOR_PALETTE['main'],
            alpha=0.1
        )
        
        ax1.plot(
            range(1, len(best_fitness_over_time) + 1),
            best_fitness_over_time,
            color=COLOR_PALETTE['main'],
            marker='o',
            markersize=8,
            markerfacecolor=COLOR_PALETTE['secondary'],
            markeredgecolor='white',
            linestyle='-',
            linewidth=2.5
        )

        plt.draw()
        plt.pause(0.1)
        print(f"Epoch {epoch + 1}/{num_epochs} \t|\t Best Fitness: {best_fitness:.2f}")

    # Final görseller
    plt.ioff()
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 2], wspace=0.3)

    # Fitness Grafiği
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title(f"Fitness Summary: File: {name}, {dimension} Cities", color=COLOR_PALETTE['main'], pad=20)
    ax1.plot(
        range(1, len(best_fitness_over_time) + 1),
        best_fitness_over_time,
        color=COLOR_PALETTE['main'],
        linewidth=3,
        marker='o',
        markersize=8,
        markerfacecolor=COLOR_PALETTE['secondary']
    )
    ax1.fill_between(
        range(1, len(best_fitness_over_time) + 1),
        best_fitness_over_time,
        color=COLOR_PALETTE['main'],
        alpha=0.1
    )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Fitness Value", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor(COLOR_PALETTE['background'])
    
    # Final grafiğe parametre kutusu ekleme
    ax1.text(
        0.5, 0.2, settings_text,
        transform=ax1.transAxes,
        ha='center', va='center',
        fontsize=11,
        linespacing=1.5,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=COLOR_PALETTE['background'],
            edgecolor=COLOR_PALETTE['main'],
            alpha=0.95
        )
    )

    # Rota Görselleştirme
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("Optimal Route Visualization", color=COLOR_PALETTE['main'], pad=20)
    
    best_solution = population.loc[population["Fitness"].idxmin()]
    best_route = best_solution["Solution"]
    coords = (
        dataframe.set_index("City_ID")
        .loc[best_route][["X_Coordinate", "Y_Coordinate"]]
        .values
    )
    coords = np.vstack([coords, coords[0]])
    
    ax2.plot(
        coords[:, 0], coords[:, 1],
        marker='o',
        markersize=10,
        markerfacecolor=COLOR_PALETTE['secondary'],
        markeredgecolor='white',
        linestyle='-',
        color=COLOR_PALETTE['main'],
        linewidth=2.5,
        alpha=0.8
    )
    
    for i, city_id in enumerate(best_route):
        ax2.text(
            coords[i, 0] + 0.5,
            coords[i, 1] + 0.5,
            str(city_id),
            fontsize=9,
            color=COLOR_PALETTE['text'],
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2')
        )
    
    ax2.set_xlabel("X Coordinate", fontsize=12)
    ax2.set_ylabel("Y Coordinate", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor(COLOR_PALETTE['background'])

    plt.tight_layout(pad=4.0)
    final_fitness = round(best_fitness_over_time[-1], 2)
    plot_filename = f"{name}_pop_{pop_size}_crossover_{crossover_probability}_fitness_{final_fitness}.png"
            
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    # final results
    print("\n" + "="*50)
    print(" Post-Optimization Analysis ".center(50, '='))
    print("="*50)
    
    print(f"\n🔍 Greedy Solution Fitness: {greedy_fitness:.2f}")
    print(f"🏆 GA Best Fitness: {best_solution['Fitness']:.2f}")
    print(f"💹 Improvement: {((greedy_fitness - best_solution['Fitness'])/greedy_fitness)*100:.1f}%")
    print("\nOptimal Route City IDs:")
    print(' → '.join(map(str, best_route)))