import time
import math
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

#matplotlib.use('Agg') #this makes the plot to be saved as a file instead of showing it in the screen

COLOR_PALETTE = {
    'main': "#3498DB",
    'secondary': "#E74C3C",
    'background': "#F8F9F9",
    'text': "#2C3E50",
    'accent': "#1ABC9C"
}

#reading the tsp file and parsing the data
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
        for _ in range(dimension): #reading the city coordinates
            line = infile.readline().strip().split()
            city_id = int(line[0])
            x_coord = float(line[1])
            y_coord = float(line[2])
            city_list.append([city_id, x_coord, y_coord])

        dataframe = pd.DataFrame( #creating a dataframe from the city list
            city_list, columns=["City_ID", "X_Coordinate", "Y_Coordinate"]
        )
        return dataframe, name, file_type, comment, dimension, edge_weight_type

def calculate_distance(city1, city2): #calculating the distance between two cities
    return math.sqrt(
        (city2["X_Coordinate"] - city1["X_Coordinate"]) ** 2
        + (city2["Y_Coordinate"] - city1["Y_Coordinate"]) ** 2
    )

def create_distance_matrix(dataframe): #creating a distance matrix for all cities
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

def calculate_fitness(solution, distance_matrix, city_to_idx): #calculating the fitness of a solution
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += distance_matrix[
            city_to_idx[solution[i]], city_to_idx[solution[i + 1]]
        ]
    total_distance += distance_matrix[
        city_to_idx[solution[-1]], city_to_idx[solution[0]]
    ]
    return total_distance

def greedy_algorithm(dataframe, start_city_id=1): #greedy algorithm to find the best solution
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

def pmx_crossover(parent1, parent2): #partially mapped crossover function
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

def mutation_function(individual, mutation_probability): #mutation function
    if random.random() < mutation_probability:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def precise_mutation(individual, distance_matrix, city_to_idx): #precise mutation function
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

def tournament_selection(population, tournament_size=5): #tournament selection function
    tournament = population.sample(n=tournament_size)
    best_index = tournament["Fitness"].values.argmin()
    best_individual = tournament.iloc[best_index]
    return {
        "Solution": best_individual["Solution"],
        "Fitness": best_individual["Fitness"],
    }

def create_new_epoch( #creating a new epoch for the genetic algorithm
    previous_population,
    distance_matrix,
    city_to_idx,
    mutation_probability,
    crossover_probability,
    pop_size,
    dataframe,
):
    new_population = []
    
    elite_count = max(1, int(0.10 * pop_size))
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

def create_population(dataframe, num_individuals): #creating the initial population
    population_data = []
    city_ids = dataframe["City_ID"].tolist()

    for _ in range(num_individuals):
        individual = random.sample(city_ids, len(city_ids))
        population_data.append({"Solution": individual, "Fitness": None})

    return pd.DataFrame(population_data)

def run_part3_comparison(dataframe, distance_matrix, city_to_idx, dimension, #running the performance comparison only for report's part 3 section
                        ga_epochs, pop_size, mutation_prob, crossover_prob):
    start_time = time.time()

    #genetic algorithm runs
    print("\n" + "="*60)
    print("=== PERFORMANCE ANALYSIS STARTED ===".center(60))
    print("="*60 + "\n")
    
    print(f"🔵 Running Genetic Algorithm (10 runs)")
    ga_results = []
    all_ga_progress = []

    for run in range(10):
        run_start = time.time()
        population = create_population(dataframe, pop_size)
        population["Fitness"] = population["Solution"].apply(
            lambda sol: calculate_fitness(sol, distance_matrix, city_to_idx)
        )
        
        run_progress = []
        for epoch in range(ga_epochs):
            population = create_new_epoch(
                population, distance_matrix, city_to_idx,
                mutation_prob, crossover_prob, pop_size, dataframe
            )
            current_best = population["Fitness"].min()
            run_progress.append(current_best)
            
            elapsed = time.time() - run_start
            progress = (epoch+1)/ga_epochs * 100
            print(f"\rRun {run+1}/10 | Epoch {epoch+1}/{ga_epochs} | Best: {current_best:.2f} | Time: {elapsed:.2f}s | Progress: {progress:.1f}%  ", end="", flush=True)
        
        ga_results.append(min(run_progress))
        all_ga_progress.append(run_progress)
        print(f"\n✅ Run {run+1} completed | Best: {ga_results[-1]:.2f} | Duration: {time.time()-run_start:.2f}s")

    # 2. Greedy Algorithm Runs
    print(f"\n🔴 Running Greedy Algorithm (100 runs)...")
    greedy_results = []
    city_ids = dataframe["City_ID"].tolist()
    
    for i in range(100):
        start_city = random.choice(city_ids)
        _, fitness = greedy_algorithm(dataframe, start_city)
        greedy_results.append(fitness)
        if (i+1) % 10 == 0:
            print(f"⏳ Progress: {i+1}/100 | Current best: {min(greedy_results):.2f}", end="\r", flush=True)

    # 3. Random Search
    print(f"\n\n⚪ Generating Random Solutions (1000 runs)...")
    population = create_population(dataframe, 1000)
    population["Fitness"] = population["Solution"].apply(
        lambda sol: calculate_fitness(sol, distance_matrix, city_to_idx)
    )
    random_results = population["Fitness"].tolist()
    print(f"✅ Random search completed | Best: {min(random_results):.2f}")

    # Statistics Calculation
    def calculate_stats(data):
        return {
            'Best': np.min(data),
            'Mean': np.mean(data),
            'Std': np.std(data),
            'Variance': np.var(data)
        }

    stats = {
        'GA': calculate_stats(ga_results),
        'Greedy': calculate_stats(greedy_results),
        'Random': calculate_stats(random_results)
    }

    # Performance Improvements
    improvement_greedy = ((stats['Greedy']['Best'] - stats['GA']['Best'])/stats['Greedy']['Best'])*100
    improvement_random = ((stats['Random']['Best'] - stats['GA']['Best'])/stats['Random']['Best'])*100

    # Terminal Output
    print("\n" + "="*80)
    print("📊 FINAL RESULTS SUMMARY".center(80))
    print("="*80)
    
    headers = ["Metric", "Genetic Algorithm", "Greedy Algorithm", "Random Search"]
    row_format = "{:<15} | {:^18} | {:^18} | {:^18}"
    print(row_format.format(*headers))
    print("-"*80)
    
    for metric in ['Best', 'Mean', 'Std', 'Variance']:
        ga_val = f"{stats['GA'][metric]:,.2f}" if metric == 'Variance' else f"{stats['GA'][metric]:.2f}"
        gr_val = f"{stats['Greedy'][metric]:,.2f}" if metric == 'Variance' else f"{stats['Greedy'][metric]:.2f}"
        rn_val = f"{stats['Random'][metric]:,.2f}" if metric == 'Variance' else f"{stats['Random'][metric]:.2f}"
        
        print(row_format.format(
            metric,
            ga_val,
            gr_val,
            rn_val
        ))

    print("\n" + "★ PERFORMANCE IMPROVEMENT".center(80))
    print(f"GA vs Greedy: +{improvement_greedy:.1f}% better")
    print(f"GA vs Random: +{improvement_random:.1f}% better")

    # Visualization
    plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 0.3, 0.7, 0.7], hspace=0.15)  # Boşluklar optimize edildi

    # 1. Fitness Progress Plot
    ax1 = plt.subplot(gs[0])
    avg_ga_progress = np.mean(all_ga_progress, axis=0)

    # Grafik çizgileri
    line1, = ax1.plot(range(1, ga_epochs+1), avg_ga_progress, 
                    color=COLOR_PALETTE['main'], linewidth=3,
                    label=f'Genetic Algorithm (Best: {stats["GA"]["Best"]:.2f})')

    line2 = ax1.axhline(y=stats['Greedy']['Best'], 
                        color=COLOR_PALETTE['secondary'], linestyle='--', linewidth=2.5)

    line3 = ax1.axhline(y=stats['Random']['Best'], 
                        color=COLOR_PALETTE['accent'], linestyle='-.', linewidth=2.5)

    # legend ve style
    ax1.legend(
        [line1, line2, line3],
        [f'Genetic Algorithm (Best: {stats["GA"]["Best"]:.2f})',
        f'Greedy Algorithm (Best: {stats["Greedy"]["Best"]:.2f})',
        f'Random Search (Best: {stats["Random"]["Best"]:.2f})'],
        loc='upper right',
        bbox_to_anchor=(0.97, 0.6),  
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        facecolor=COLOR_PALETTE['background']
    )

    ax1.set_title(f"Performance Comparison: GA, Greedy, and Random on {name} ({dimension} Cities)", 
                pad=20, color=COLOR_PALETTE['main'], fontsize=14)
    ax1.set_facecolor(COLOR_PALETTE['background'])

    # part3 text
    ax_note = plt.subplot(gs[1])
    ax_note.axis('off')
    note_text = (
        f"Part 3: Experiment Summary for {name} file. ({dimension} Cities)\n\n"
        " - Genetic Algorithm tested 10 times with statistical analysis.\n"
        " - Greedy Algorithm tested 100 times, best 5 results shown.\n"
        " - Random Search tested 1000 times, statistical summary provided."
    )
    ax_note.text(0.5, 0.5, note_text, ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='lightgray', edgecolor='gray', boxstyle='round,pad=0.5'))

    # 3. experiment parameters table 
    ax2 = plt.subplot(gs[2])
    ax2.axis('off')

    params_data = [
        ["Experiment Parameter", "Value", "Performance Metric", "Result"],
        ["Epochs", ga_epochs, "GA Fitness", f"{stats['GA']['Best']:.2f}"],
        ["Population Size", pop_size, "Greedy Fitness", f"{stats['Greedy']['Best']:.2f}"],
        ["Crossover Probability", crossover_prob, "Random Fitness", f"{stats['Random']['Best']:.2f}"],
        ["Mutation Probability", f"{mutation_prob:.2f}", "GA Improvement vs. Greedy", f"+{improvement_greedy:.1f}%"],
        ["Tournament Size", f"5", "GA Improvement vs. Random", f"+{improvement_random:.1f}%"]
    ]

    params_table = ax2.table(
        cellText=params_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.2, 0.3, 0.25],  
        bbox=[0.05, 0.1, 0.9, 0.8]  
    )

    for j in range(4):
        cell = params_table[0, j]
        cell.set_facecolor("#BDC3C7")
        cell.set_text_props(color="black", weight="bold")
        cell.set_fontsize(12)

    ax3 = plt.subplot(gs[3])
    ax3.axis('off')

    results_data = [
        ["Metric", "Genetic Algorithm", "Greedy Algorithm", "Random Search"],
        ["Best Fitness", 
        f"{stats['GA']['Best']:.2f}", 
        f"{stats['Greedy']['Best']:.2f}", 
        f"{stats['Random']['Best']:.2f}"],
        ["Mean Fitness", 
        f"{stats['GA']['Mean']:.2f}", 
        f"{stats['Greedy']['Mean']:.2f}", 
        f"{stats['Random']['Mean']:.2f}"],
        ["Standard Deviation", 
        f"{stats['GA']['Std']:.2f}", 
        f"{stats['Greedy']['Std']:.2f}", 
        f"{stats['Random']['Std']:.2f}"],
        ["Variance", 
        f"{stats['GA']['Variance']:,.2f}", 
        f"{stats['Greedy']['Variance']:,.2f}", 
        f"{stats['Random']['Variance']:,.2f}"]
    ]

    results_table = ax3.table(
        cellText=results_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25],
        bbox=[0.05, 0.1, 0.9, 0.8]
    )

    #header style
    header_colors = ["white", "#3498DB", "#E74C3C", "#2ECC71"]
    for j, color in enumerate(header_colors):
        cell = results_table[0, j]
        cell.set_facecolor(color)
        cell.set_text_props(color="white", weight="bold")
        cell.set_fontsize(12)

    # cell style
    for table in [params_table, results_table]:
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor(COLOR_PALETTE['text'])
            cell.set_linewidth(0.4)
            cell.set_height(0.08)

    plt.savefig(f"{name}_performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n📈 Visualization saved as '{name}_performance_comparison.png'")
    print(f"⏱ Total execution time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    # Global Settings
    RUN_PART3 = True  # Set to False for normal operation
    GA_PARAMS = { #GA settings for part 3 mode, for normal operation, update these values in the below after else statement
        'num_epochs': 100,
        'pop_size': 200,
        'mutation_probability': 0.33,
        'crossover_probability': 0.6
    }

    # Load TSP Data
    dataframe, name, file_type, comment, dimension, edge_weight_type = parse_tsp_file("berlin11.tsp")
    distance_matrix, city_to_idx = create_distance_matrix(dataframe)

    if RUN_PART3:
        run_part3_comparison(
            dataframe=dataframe,
            distance_matrix=distance_matrix,
            city_to_idx=city_to_idx,
            dimension=dimension,
            ga_epochs=GA_PARAMS['num_epochs'],
            pop_size=GA_PARAMS['pop_size'],
            mutation_prob=GA_PARAMS['mutation_probability'],
            crossover_prob=GA_PARAMS['crossover_probability']
        )
        
    else:
        
        # Run Genetic Algorithm with Normal mode
        
        random.seed(random.randint(50, 100))
        greedy_solution, greedy_fitness = greedy_algorithm(dataframe)
        population = create_population(dataframe, 100)
        population["Fitness"] = population["Solution"].apply(
            lambda sol: calculate_fitness(sol, distance_matrix, city_to_idx)
        )

        # Genetic algorithm parameters for Normal mode
        num_epochs = 100
        crossover_probability = 0.6
        pop_size = 200
        mutation_probability = 0.33
        
        best_fitness_over_time = []
        
        plt.ion()
        fig = plt.figure(figsize=(15, 7))
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[0])

        settings_text = (
            "⚙️ GA Parameters:\n"
            f"• Epochs: {num_epochs}\n"
            f"• Crossover: {crossover_probability}\n"
            f"• Population: {pop_size}\n"
            f"• Mutation: {mutation_probability:.2f}\n"
            f"• Greedy Solution Fitness: {greedy_fitness:.2f}\n"
            f"• Initial Best Fitness: {population['Fitness'].min():.2f}"
        )

        ax1.set_title(f"Real-Time Optimization: {name} ({dimension} Cities)", color=COLOR_PALETTE['main'], pad=20)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Fitness Value", fontsize=12)
        ax1.set_xlim(1, num_epochs)
        ax1.set_facecolor(COLOR_PALETTE['background'])
        initial_fitness = None

        for epoch in range(num_epochs):
            population = create_new_epoch(
                population, distance_matrix, city_to_idx,
                mutation_probability, crossover_probability, pop_size, dataframe
            )
            best_fitness = population["Fitness"].min()
            best_fitness_over_time.append(best_fitness)
            
            if initial_fitness is None:
                initial_fitness = best_fitness

            ax1.clear()
            current_title = f"Epoch: {epoch+1}/{num_epochs} | Best Fitness: {best_fitness:.2f}"
            ax1.set_title(current_title, color=COLOR_PALETTE['main'], pad=20)
            ax1.set_xlabel("Epoch", fontsize=12)
            ax1.set_ylabel("Fitness Value", fontsize=12)
            ax1.set_xlim(1, num_epochs)
            
            settings_text = (
                f"⚙️ GA Parameters:\n"
                f"• Epochs: {num_epochs}\n"
                f"• Crossover: {crossover_probability}\n"
                f"• Population: {pop_size}\n"
                f"• Mutation: {mutation_probability:.2f}\n"
                f"• Greedy Solution Fitness: {greedy_fitness:.2f}\n"
                f"• Initial Best Fitness: {best_fitness:.2f}\n"
                f"• Improvement: {((greedy_fitness - best_fitness)/greedy_fitness)*100:.1f}%"
            )
            
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

        plt.ioff()
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 2], wspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        ax1.set_title(f"Fitness Summary: {name}", color=COLOR_PALETTE['main'], pad=20)
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

        ax2 = fig.add_subplot(gs[1])
        ax2.set_title("Optimal Route", color=COLOR_PALETTE['main'], pad=20)
        
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
        plot_filename = f"{name}_fitness_{final_fitness}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()

        print("\n" + "="*50)
        print(" Final Results ".center(50, '='))
        print("="*50)
        print(f"\n🏆 GA Best Fitness: {best_solution['Fitness']:.2f}")
        print(f"📉 Greedy Fitness: {greedy_fitness:.2f}")
        print(f"🚀 Improvement: {((greedy_fitness - best_solution['Fitness'])/greedy_fitness)*100:.1f}%")
        
        

