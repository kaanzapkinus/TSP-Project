import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from your_module import (  # Kendi fonksiyonlarınızı import edin
    parse_tsp_file,
    create_distance_matrix,
    create_population,
    calculate_fitness,
    create_new_epoch,
    greedy_algorithm
)

# Görsel Stil Sabitleri
COLOR_PALETTE = {
    'main': "#3498DB",
    'secondary': "#E74C3C",
    'background': "#F8F9F9",
    'text': "#2C3E50",
    'accent': "#1ABC9C"
}

def configure_plots():
    """Görsel ayarları merkezi olarak yönet"""
    plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'lines.linewidth': 2.5,
        'grid.alpha': 0.3,
        'figure.figsize': (16, 8),
        'font.family': 'DejaVu Sans'
    })

if __name__ == "__main__":
    configure_plots()
    
    random.seed(random.randint(50, 100))
    dataframe, name, file_type, comment, dimension, edge_weight_type = parse_tsp_file("berlin52.tsp")
    distance_matrix, city_to_idx = create_distance_matrix(dataframe)

    population = create_population(dataframe, num_individuals=100)
    population["Fitness"] = population["Solution"].apply(
        lambda sol: calculate_fitness(sol, distance_matrix, city_to_idx)
    )

    # Genetic algorithm settings
    num_epochs = 50
    crossover_probability = 0.6
    pop_size = 200
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
        f"• Mutation: {mutation_probability:.2f}"
    )

    # Başlangıç grafik konfigürasyonu
    ax1.set_title(f"Real-Time Optimization: {name} ({dimension} Cities)", color=COLOR_PALETTE['main'], pad=20)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Fitness Value", fontsize=12)
    ax1.set_xlim(1, num_epochs)
    ax1.set_ylim(0, population["Fitness"].min() * 1.2)
    ax1.set_facecolor(COLOR_PALETTE['background'])

    # Parametre kutusu
    param_box = ax1.text(
        0.03, 0.25, settings_text,
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
    
        # Grafik güncelleme
        ax1.clear()
        current_title = f"Epoch: {epoch+1}/{num_epochs} | Best Fitness: {best_fitness:.2f}"
        ax1.set_title(current_title, color=COLOR_PALETTE['main'], pad=20)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Fitness Value", fontsize=12)
        ax1.set_xlim(1, num_epochs)
        ax1.set_ylim(0, population["Fitness"].min() * 1.2)
        ax1.set_facecolor(COLOR_PALETTE['background'])
        
        # Güncellenmiş parametre kutusu
        ax1.text(
            0.03, 0.25, settings_text,
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
    ax1.set_title(f"Optimization Summary: {name}", color=COLOR_PALETTE['main'], pad=20)
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
    plot_filename = f"GA_Optimized_{name}_Result_{final_fitness}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    # Greedy sonuçları
    print("\n" + "="*50)
    print(" Post-Optimization Analysis ".center(50, '='))
    print("="*50)
    greedy_solution, greedy_fitness = greedy_algorithm(dataframe, start_city_id=1)
    print(f"\n🔍 Greedy Solution Fitness: {greedy_fitness:.2f}")
    print(f"🏆 GA Best Fitness: {best_solution['Fitness']:.2f}")
    print(f"💹 Improvement: {((greedy_fitness - best_solution['Fitness'])/greedy_fitness)*100:.1f}%")
    print("\nOptimal Route City IDs:")
    print(' → '.join(map(str, best_route)))