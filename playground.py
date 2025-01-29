# tsp_hyperoptimized.py
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import cycle

COLOR_PALETTE = {
    'main': "#3498DB",
    'secondary': "#E74C3C",
    'background': "#F8F9F9",
    'text': "#2C3E50",
    'accent': "#1ABC9C"
}

# -------------------- HYPER-OPTIMIZED CORE --------------------
class TurboTSPSolver:
    def __init__(self, df):
        self.cities = df[["X", "Y"]].values
        self.n = len(self.cities)
        self.dist_matrix = self._precompute_distances()
        self.city_ids = df["City_ID"].tolist()
        
        # Optimizasyon için önbellekler
        self._id_to_idx = {cid: i for i, cid in enumerate(self.city_ids)}
        self._best_ever = float('inf')
        
        if self.n < 3:
            raise ValueError("En az 3 şehir gereklidir")

    def _precompute_distances(self):
        """Vektörleştirilmiş mesafe hesaplama"""
        diff = self.cities[:, np.newaxis, :] - self.cities[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))

    def _tour_length(self, tour_indices):
        """Numpy ile optimize edilmiş tur uzunluğu"""
        return np.sum(self.dist_matrix[tour_indices, np.roll(tour_indices, -1)])

    def greedy_solve(self):
        """Hızlandırılmış açgözlü algoritma"""
        current = 0  # İndex bazlı çalış
        unvisited = set(range(1, self.n))
        tour = [current]
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.dist_matrix[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        return [self.city_ids[i] for i in tour], self._tour_length(tour)

    def genetic_solve(self, time_limit=5):
        """Zaman sınırlı genetik algoritma"""
        start_time = time.time()
        pop_size = min(50, self.n*10)
        population = self._init_population(pop_size)
        best = min(population, key=lambda x: x[1])
        self._best_ever = best[1]
        
        # Adaptif parametreler
        mutation_rate = 0.3
        no_improvement = 0
        
        while time.time() - start_time < time_limit:
            # Hızlı seçim
            parents = random.choices(population, k=pop_size, 
                                   weights=[1/(s+1e-6) for s in [x[1] for x in population]])
            
            # Vektörleştirilmiş çaprazlama
            new_pop = []
            for i in range(0, pop_size, 2):
                child1 = self._fast_crossover(parents[i][0], parents[i+1][0])
                child2 = self._fast_crossover(parents[i+1][0], parents[i][0])
                new_pop.extend([
                    self._mutate(child1, mutation_rate),
                    self._mutate(child2, mutation_rate)
                ])
            
            # Elitizm
            combined = population + new_pop
            combined.sort(key=lambda x: x[1])
            population = combined[:pop_size]
            
            # Adaptif parametre ayarı
            if population[0][1] < self._best_ever:
                self._best_ever = population[0][1]
                no_improvement = 0
                mutation_rate = max(0.1, mutation_rate*0.95)
            else:
                no_improvement += 1
                mutation_rate = min(0.6, mutation_rate*1.05)
            
            if no_improvement > 10:
                population = self._diversify(population)
                no_improvement = 0
                
        best_tour = [self.city_ids[i] for i in population[0][0]]
        return best_tour, population[0][1]

    def _init_population(self, size):
        """Önbelleklenmiş popülasyon"""
        return [self._create_individual() for _ in range(size)]

    def _create_individual(self):
        """Hızlı birey oluşturma"""
        tour = list(range(self.n))
        random.shuffle(tour)
        return (tour, self._tour_length(tour))

    def _fast_crossover(self, p1, p2):
        """Sıralı çaprazlama (OX) ile hızlandırılmış"""
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        segment = p1[a:b]
        child = [-1]*size
        
        # Segmenti kopyala
        child[a:b] = segment
        
        # Kalanları p2'den doldur
        ptr = 0
        for i in range(size):
            if child[i] == -1:
                while p2[ptr] in segment:
                    ptr += 1
                child[i] = p2[ptr]
                ptr += 1
                
        return (child, self._tour_length(child))

    def _mutate(self, individual, rate):
        """Swap + 2-opt combo mutasyon"""
        tour, score = individual
        if random.random() < rate:
            # 2-opt
            a, b = sorted(random.sample(range(self.n), 2))
            new_tour = tour[:a] + tour[a:b+1][::-1] + tour[b+1:]
            new_score = self._tour_length(new_tour)
            return (new_tour, new_score)
        return individual

    def _diversify(self, population):
        """Popülasyon çeşitlendirme"""
        elite = population[:len(population)//2]
        new = [self._create_individual() for _ in range(len(population)//2)]
        return sorted(elite + new, key=lambda x: x[1])[:len(population)]

# -------------------- LIGHTNING REPORT --------------------
def generate_report(tsp_file):
    # Veri yükleme
    df, _ = parse_tsp_file(tsp_file)
    
    # Çözücüyü başlat
    print("⚡ Turbo Çözücü Başlatılıyor...")
    solver = TurboTSPSolver(df)
    
    # Algoritmaları çalıştır
    print("\n🚀 Açgözlü Algoritma Çalışıyor...")
    start = time.time()
    greedy_tour, greedy_len = solver.greedy_solve()
    print(f"✅ Tamamlandı! Süre: {time.time()-start:.3f}s")
    
    print("\n🧬 Genetik Algoritma Çalışıyor (5s limit)...")
    start = time.time()
    ga_tour, ga_len = solver.genetic_solve(time_limit=5)
    print(f"✅ Tamamlandı! Süre: {time.time()-start:.3f}s")
    
    # Rastgele 1000 deneme
    print("\n🎲 Rastgele Tarama Yapılıyor...")
    random_len = []
    for _ in range(1000):
        tour = random.sample(solver.city_ids, solver.n)
        indices = [solver._id_to_idx[cid] for cid in tour]
        random_len.append(solver._tour_length(indices))
    
    # Sonuçları göster
    print("\n⭐ Nihai Sonuçlar:")
    print(f"{'Metrik':<15} | {'Açgözlü':<15} | {'Genetik':<15} | {'Rastgele':<15}")
    print("-"*65)
    print(f"{'En İyi':<15} | {greedy_len:<15.2f} | {ga_len:<15.2f} | {min(random_len):<15.2f}")
    print(f"{'Ortalama':<15} | {'-':<15} | {'-':<15} | {np.mean(random_len):<15.2f}")
    
    # Hızlı görselleştirme
    plt.figure(figsize=(12,6))
    plt.plot(greedy_len, 'o', color=COLOR_PALETTE['main'], label='Açgözlü')
    plt.plot(ga_len, 's', color=COLOR_PALETTE['secondary'], label='Genetik')
    plt.hlines(min(random_len), 0, 2, color=COLOR_PALETTE['accent'], label='Rastgele En İyi')
    plt.legend()
    plt.title("Performans Karşılaştırması")
    plt.savefig("tsp_quick_report.png", dpi=120)
    print("\n📊 Görsel rapor kaydedildi: tsp_quick_report.png")

# -------------------- UTILITIES --------------------
def parse_tsp_file(file_path):
    """Hızlı TSP dosya okuyucu"""
    with open(file_path, 'r') as f:
        cities = []
        read_coords = False
        for line in f:
            line = line.strip()
            if line.startswith("DIMENSION"):
                dim = int(line.split(":")[1])
            if line == "NODE_COORD_SECTION":
                read_coords = True
                continue
            if read_coords and line != "EOF":
                parts = line.split()
                cities.append([int(parts[0]), float(parts[1]), float(parts[2])])
        df = pd.DataFrame(cities, columns=["City_ID", "X", "Y"])
        return df, dim

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Kullanım: python tsp_hyperoptimized.py <tsp_dosyası>")
        sys.exit(1)
        
    try:
        generate_report(sys.argv[1])
    except Exception as e:
        print(f"Hata: {str(e)}")