import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import copy

@dataclass
class Box:
    dims: Tuple[float, float, float]
    weight: float
    id: str

class Container:
    def __init__(self, dims: Tuple[float, float, float], max_weight: float):
        self.dims = np.array(dims)
        self.max_weight = max_weight
        self.placed: List[Tuple[Tuple[float, float, float], float, str]] = []

    def get_used_vol(self) -> float:
        return sum(np.prod(np.array(r[0])) for r in self.placed)

    def get_used_weight(self) -> float:
        return sum(r[1] for r in self.placed)

    def can_add(self, rot_dims: Tuple[float, float, float], w: float) -> bool:
        total_vol = self.get_used_vol() + np.prod(np.array(rot_dims))
        total_w = self.get_used_weight() + w
        return total_vol <= np.prod(self.dims) * 0.98 and total_w <= self.max_weight

def get_orientations(dims: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
    l, w, h = dims
    return [(l,w,h), (l,h,w), (w,l,h), (w,h,l), (h,l,w), (h,w,l)]

def simple_packing(boxes: List[Box], bps_order: List[int], cont_dims: Tuple[float,float,float], max_w: float):
    """Packing simplificado pero FUNCIONAL - siempre coloca cajas"""
    oc: List[Container] = []
    
    # Ordenar cajas según BPS
    sorted_boxes = [boxes[i] for i in sorted(range(len(boxes)), key=lambda x: bps_order[x])]
    
    for box in sorted_boxes:
        placed = False
        
        # Intentar contenedores existentes (menor volumen usado primero)
        oc.sort(key=lambda c: c.get_used_vol())
        for c in oc:
            for rot in get_orientations(box.dims):
                if c.can_add(rot, box.weight):
                    c.placed.append((rot, box.weight, box.id))
                    placed = True
                    break
            if placed: break
        
        # Nuevo contenedor si no cabe
        if not placed:
            new_c = Container(cont_dims, max_w)
            # Siempre cabe la primera caja
            best_rot = min(get_orientations(box.dims), key=lambda r: np.prod(r))
            new_c.placed.append((best_rot, box.weight, box.id))
            oc.append(new_c)
            placed = True
    
    return len(oc), oc

def parse_manual_input() -> Tuple[Tuple[float, float, float], float, List[Box]]:
    print("=== 3D BIN PACKING - GENETIC ALGORITHM (FIXED) ===")
    
    # Datos de ejemplo de tu instancia para test rápido
    print("Usando tu instancia de ejemplo:")
    cont_dims = (240, 180, 200)
    max_w = 1000
    
    boxes = []
    data = [
        ("1", 2, 34, 16, 27, 40.07),
        ("2", 4, 75, 83, 80, 15.97),
        ("3", 5, 79, 85, 83, 19.36),
        ("4", 4, 46, 97, 95, 41.98),
        ("5", 2, 77, 42, 87, 25.92),
        ("6", 4, 68, 82, 23, 3.54),
        ("7", 3, 66, 87, 62, 2.86),
        ("8", 5, 14, 92, 92, 15.37),
        ("9", 1, 54, 43, 30, 43.76),
        ("10", 5, 22, 55, 12, 33.65)
    ]
    
    for id_, cant, l, w, h, pw in data:
        for _ in range(cant):
            boxes.append(Box((l, w, h), pw, id_))
    
    print(f"📦 Contenedor: {cont_dims} cm, {max_w} kg")
    print(f"📦 {len(boxes)} cajas cargadas")
    return cont_dims, max_w, boxes

class GeneticAlgorithm:
    def __init__(self, boxes: List[Box], cont_dims: Tuple[float,float,float], max_w: float):
        self.boxes = boxes
        self.cont_dims = cont_dims
        self.max_w = max_w
        self.n_boxes = len(boxes)
    
    def create_individual(self) -> List[int]:
        """Solo BPS order - simplificado"""
        order = list(range(self.n_boxes))
        random.shuffle(order)
        return order
    
    def fitness(self, individual: List[int]) -> float:
        n_bins, _ = simple_packing(self.boxes, individual, self.cont_dims, self.max_w)
        # Fitness = 1/n_bins (menor bins = mejor fitness)
        return 1.0 / n_bins
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        # Partially Mapped Crossover (PMX)
        size = len(parent1)
        p1, p2 = parent1[:], parent2[:]
        
        # Seleccionar dos puntos de corte
        cx1, cx2 = sorted(random.sample(range(size), 2))
        
        # Copiar segmento central
        child1 = [-1] * size
        child2 = [-1] * size
        child1[cx1:cx2+1] = p1[cx1:cx2+1]
        child2[cx1:cx2+1] = p2[cx1:cx2+1]
        
        # Llenar resto con mapping
        for c1, c2 in zip(p1, p2):
            if c1 not in child2[cx1:cx2+1]:
                for i in range(size):
                    if child2[i] == -1:
                        child2[i] = c1
                        break
            if c2 not in child1[cx1:cx2+1]:
                for i in range(size):
                    if child1[i] == -1:
                        child1[i] = c2
                        break
        
        return child1, child2
    
    def mutate(self, individual: List[int], mutation_rate: float) -> List[int]:
        mutated = individual[:]
        if random.random() < mutation_rate:
            # Swap mutation
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def run(self, population_size=50, generations=30, mutation_rate=0.2, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"\n🚀 GA Corregido: Pop={population_size}, Gen={generations}, Mut={mutation_rate:.2f}, Seed={seed}")
        print("⏳ Evaluando población inicial...")
        
        population = [self.create_individual() for _ in range(population_size)]
        
        for gen in range(generations):
            # Evaluar fitness
            fitnesses = []
            for i, ind in enumerate(population):
                f = self.fitness(ind)
                fitnesses.append((f, ind))
                if i % 10 == 0:
                    print(f"  Gen {gen}, Ind {i}: fitness={f:.4f}", end='\r')
            
            fitnesses.sort(reverse=True, key=lambda x: x[0])
            population = [ind for _, ind in fitnesses]
            
            best_fitness = fitnesses[0][0]
            best_n_bins = int(1/best_fitness)
            
            if gen % 5 == 0 or gen == generations-1:
                print(f"\nGen {gen}: Best={best_n_bins} bins (fitness={best_fitness:.4f})")
            
            # Nueva generación
            next_gen = population[:5]  # Elite
            while len(next_gen) < population_size:
                p1, p2 = random.choices(population[:20], k=2)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1, mutation_rate)
                c2 = self.mutate(c2, mutation_rate)
                next_gen.extend([c1[:], c2[:]])
            
            population = next_gen[:population_size]
        
        # Mejor solución final
        best_ind = population[0]
        best_n_bins, best_containers = simple_packing(self.boxes, best_ind, self.cont_dims, self.max_w)
        return [(best_n_bins, best_containers)]

def print_report(solutions: List, boxes: List[Box], cont_dims: Tuple[float,float,float]):
    total_vol = sum(np.prod(b.dims) for b in boxes)
    total_w = sum(b.weight for b in boxes)
    
    print("\n" + "="*70)
    print("🏆 MEJOR SOLUCIÓN ENCONTRADA")
    print("="*70)
    
    n_bins, containers = solutions[0]
    print(f"📦 {n_bins} contenedores necesarios")
    print(f"🎯 Vol total cajas: {total_vol:,.0f} cm³")
    
    for i, c in enumerate(containers, 1):
        c_vol = c.get_used_vol()
        c_w = c.get_used_weight()
        print(f"\n  Contenedor {i}:")
        print(f"    Vol: {c_vol:,.0f}/{np.prod(cont_dims):,.0f} cm³ ({c_vol/np.prod(cont_dims)*100:.1f}%)")
        print(f"    Peso: {c_w:.1f}/{c.max_weight} kg ({c_w/c.max_weight*100:.1f}%)")
        print(f"    Cajas ({len(c.placed)}): " + ", ".join(f"{id_}" for _,_,id_ in c.placed))

# EJECUTAR
if __name__ == "__main__":
    cont_dims, max_w, boxes = parse_manual_input()
    
    while True:
        seed = input("\nSemilla (1234=default, Enter): ").strip() or "1234"
        pop = input("Población (50=default): ").strip() or "50"
        gens = input("Generaciones (30=default): ").strip() or "30"
        
        try:
            ga = GeneticAlgorithm(boxes, cont_dims, max_w)
            solutions = ga.run(
                population_size=int(pop),
                generations=int(gens),
                mutation_rate=0.2,
                seed=int(seed) if seed.isdigit() else None
            )
            print_report(solutions, boxes, cont_dims)
        except KeyboardInterrupt:
            print("\nSaliendo...")
            break
        except Exception as e:
            print(f"Error: {e}")
