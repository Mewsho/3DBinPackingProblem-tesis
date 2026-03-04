import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time
import os

@dataclass
class Box:
    dims: Tuple[float, float, float]
    weight: float
    id: str

def parse_data() -> Tuple[Tuple[float, float, float], float, List[Box]]:
    cont_dims = (260, 280, 200)
    max_w = 1000
    
    boxes = []
    data = [
        ("1", 2, 34, 16, 27, 40.07), ("2", 4, 75, 83, 80, 15.97), ("3", 5, 79, 85, 83, 19.36),
        ("4", 4, 46, 97, 95, 41.98), ("5", 2, 77, 42, 87, 25.92), ("6", 4, 68, 82, 23, 3.54),
        ("7", 3, 66, 87, 62, 2.86), ("8", 5, 14, 92, 92, 15.37), ("9", 1, 54, 43, 30, 43.76),
        ("10", 5, 22, 55, 12, 33.65)
    ]
    
    for id_, cant, l, w, h, pw in data:
        for _ in range(int(cant)):
            boxes.append(Box((float(l), float(w), float(h)), float(pw), id_))
    
    print(f"Datos cargados: {len(boxes)} cajas")
    return cont_dims, max_w, boxes

def fast_volume_fitness(individual, box_volumes, box_weights, cont_volume, max_weight, n_boxes):
    """Fitness ultrarrápido SIN numpy para evitar problemas"""
    selected_vol = sum(box_volumes[i] for i in range(n_boxes) if individual[i] == 1)
    selected_weight = sum(box_weights[i] for i in range(n_boxes) if individual[i] == 1)
    n_selected = sum(1 for x in individual if x == 1)
    
    n_containers_vol = max(1, int(np.ceil(selected_vol / cont_volume)))
    n_containers_weight = max(1, int(np.ceil(selected_weight / max_weight)))
    n_containers = max(n_containers_vol, n_containers_weight)
    
    density = selected_vol / (n_containers * cont_volume)
    utilization = n_selected / n_boxes
    
    fitness = density * 0.7 + utilization * 0.3
    return (fitness, selected_vol, n_containers, n_selected)

class GeneticAlgorithm3DBinPacking:
    def __init__(self, boxes: List[Box], cont_dims: Tuple[float, float, float], max_w: float):
        self.boxes = boxes
        self.cont_dims = cont_dims
        self.max_w = max_w
        self.n_boxes = len(boxes)
        self.cont_volume = np.prod(cont_dims)
        self.box_volumes = [np.prod(b.dims) for b in boxes]
        self.box_weights = [b.weight for b in boxes]
        
    def create_individual(self):
        return [random.randint(0, 1) for _ in range(self.n_boxes)]
    
    def mutate(self, individual, rate=0.1):
        mutated = individual[:]
        for i in range(self.n_boxes):
            if random.random() < rate:
                mutated[i] = 1 - mutated[i]
        return mutated
    
    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def log_progress(self, gen, best_fitness, best_n_boxes, best_vol, population_size, elapsed):
        print(f"Gen {gen:3d} | Fitness: {best_fitness:.4f} | Cajas: {best_n_boxes}/35 | "
              f"Vol: {best_vol/1e6:6.2f}m3 | Tiempo: {elapsed:.1f}s")
    
    def run(self, population_size=200, generations=50, mutation_rate=0.12):
        print(f"Algoritmo Genetico 3D Bin Packing")
        print(f"Poblacion: {population_size}, Generaciones: {generations}")
        print(f"Cajas totales: {self.n_boxes}")
        print("-" * 60)
        
        population = [self.create_individual() for _ in range(population_size)]
        best_solution = None
        best_fitness = -float('inf')
        
        start_time = time.time()
        
        for gen in range(generations):
            gen_start = time.time()
            
            # EVALUACIÓN SECUENCIAL (SIN MULTIPROCESSING para evitar errores)
            fitness_results = []
            for i, ind in enumerate(population):
                result = fast_volume_fitness(ind, self.box_volumes, self.box_weights, 
                                          self.cont_volume, self.max_w, self.n_boxes)
                fitness_results.append((result[0], result[1], result[2], result[3], i))
            
            # ORDENAR POR FITNESS
            fitness_results.sort(reverse=True, key=lambda x: x[0])
            
            # MEJOR SOLUCIÓN ACTUAL
            current_fitness, current_vol, current_containers, current_n_boxes, current_idx = fitness_results[0]
            
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = {
                    'individual': population[current_idx][:],
                    'fitness': current_fitness,
                    'volume': current_vol,
                    'containers': current_containers,
                    'n_boxes': current_n_boxes
                }
            
            elapsed_gen = time.time() - gen_start
            self.log_progress(gen, current_fitness, current_n_boxes, current_vol, 
                            population_size, elapsed_gen)
            
            # ✅ NUEVA GENERACIÓN - SIN ERRORES DE UNPACKING
            elite_size = max(1, population_size // 10)
            elite_indices = [item[4] for item in fitness_results[:elite_size]]  # ✅ SOLO EL ÍNDICE
            next_population = [population[idx] for idx in elite_indices]
            
            # GENERAR RESTO DE POBLACIÓN
            while len(next_population) < population_size:
                if random.random() < 0.7 and len(fitness_results) >= 2:
                    # Crossover entre top performers
                    p1_idx = random.randint(0, min(elite_size*2-1, len(fitness_results)-1))
                    p2_idx = random.randint(0, min(elite_size*2-1, len(fitness_results)-1))
                    p1 = population[fitness_results[p1_idx][4]]
                    p2 = population[fitness_results[p2_idx][4]]
                    c1, c2 = self.crossover(p1, p2)
                    next_population.extend([c1, c2])
                else:
                    # Mutación del mejor
                    parent_idx = random.randint(0, min(elite_size*2-1, len(fitness_results)-1))
                    parent = population[fitness_results[parent_idx][4]]
                    child = self.mutate(parent, mutation_rate)
                    next_population.append(child)
            
            population = next_population[:population_size]
        
        total_time = time.time() - start_time
        print(f"\nTiempo total: {total_time:.1f} segundos")
        return best_solution

def print_final_report(best_solution, boxes, cont_dims):
    cont_volume = np.prod(cont_dims)
    n_containers = best_solution['containers']
    selected_indices = [i for i, x in enumerate(best_solution['individual']) if x == 1]
    
    total_volume_used = sum(np.prod(boxes[i].dims) for i in selected_indices)
    total_weight_used = sum(boxes[i].weight for i in selected_indices)
    
    vol_used_m3 = total_volume_used / 1_000_000
    vol_remaining_m3 = (n_containers * cont_volume / 1_000_000) - vol_used_m3
    
    print("\n" + "="*80)
    print("RESULTADO FINAL - OPTIMIZACION 3D BIN PACKING")
    print("="*80)
    print(f"Paquetes utilizados: {best_solution['n_boxes']:2d} / 35")
    print(f"Contenedores requeridos: {n_containers}")
    print()
    print(f"Volumen usado:     {vol_used_m3:8.3f} m3 ({total_volume_used:>10,.0f} cm3)")
    print(f"Volumen restante:  {vol_remaining_m3:8.3f} m3")
    print(f"Densidad volumen:  {total_volume_used/(n_containers*cont_volume)*100:6.1f}%")
    print()
    print(f"Peso usado:        {total_weight_used:8.1f} kg")
    print(f"Peso restante:     {n_containers*1000 - total_weight_used:8.1f} kg")
    print(f"Densidad peso:     {total_weight_used/(n_containers*1000)*100:6.1f}%")
    print(f"Fitness final:     {best_solution['fitness']:6.4f}")
    print("-"*80)

if __name__ == "__main__":
    cont_dims, max_w, boxes = parse_data()
    
    population_size = 200
    generations = 50
    mutation_rate = 0.12
    
    ga = GeneticAlgorithm3DBinPacking(boxes, cont_dims, max_w)
    best_solution = ga.run(population_size, generations, mutation_rate)
    
    print_final_report(best_solution, boxes, cont_dims)
