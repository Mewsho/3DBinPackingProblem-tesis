import re
import sys
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time

@dataclass
class Box:
    dims: Tuple[float, float, float]  # (largo, ancho, alto) en cm
    weight: float                      # kg
    id: str

class Space:
    __slots__ = ['x', 'y', 'z', 'w', 'd', 'h']

    def __init__(self, x, y, z, w, d, h):
        self.x, self.y, self.z = x, y, z   # esquina de origen
        self.w, self.d, self.h = w, d, h   # ancho, profundidad, alto

    def volume(self):
        return self.w * self.d * self.h

    def can_fit(self, bw, bd, bh) -> bool:
        return bw <= self.w + 1e-9 and bd <= self.d + 1e-9 and bh <= self.h + 1e-9


class Container3D:

    def __init__(self, dims: Tuple[float, float, float], max_weight: float):
        self.W, self.D, self.H = dims
        self.max_weight = max_weight
        self.total_volume = self.W * self.D * self.H
        self.reset()

    def reset(self):
        self.placed: List[dict] = []
        self.current_weight = 0.0
        self.volume_used = 0.0
        # Espacio inicial = todo el contenedor
        self.spaces: List[Space] = [Space(0, 0, 0, self.W, self.D, self.H)]

    @staticmethod
    def get_rotations(l, w, h) -> List[Tuple[float, float, float]]:
        return list({(l, w, h), (l, h, w), (w, l, h),
                     (w, h, l), (h, l, w), (h, w, l)})

    def try_place(self, box: Box) -> bool:
        # Verificar restricción de peso
        if self.current_weight + box.weight > self.max_weight + 1e-9:
            return False

        l, w, h = box.dims
        best_space_idx = None
        best_rotation = None
        best_waste = float('inf')

        for rot in self.get_rotations(l, w, h):
            bw, bd, bh = rot
            for i, space in enumerate(self.spaces):
                if space.can_fit(bw, bd, bh):
                    waste = space.volume() - bw * bd * bh
                    if waste < best_waste:
                        best_waste = waste
                        best_space_idx = i
                        best_rotation = rot

        if best_space_idx is None:
            return False  

        space = self.spaces[best_space_idx]
        bw, bd, bh = best_rotation

        self.placed.append({
            'box': box,
            'pos': (space.x, space.y, space.z),
            'dims': (bw, bd, bh)
        })
        self.current_weight += box.weight
        self.volume_used += bw * bd * bh

        new_spaces = []

        if space.w - bw > 1e-6:
            new_spaces.append(Space(
                space.x + bw, space.y, space.z,
                space.w - bw, space.d, space.h
            ))
        if space.d - bd > 1e-6:
            new_spaces.append(Space(
                space.x, space.y + bd, space.z,
                bw, space.d - bd, space.h
            ))
        if space.h - bh > 1e-6:
            new_spaces.append(Space(
                space.x, space.y, space.z + bh,
                bw, bd, space.h - bh
            ))

        self.spaces.pop(best_space_idx)
        self.spaces.extend(new_spaces)
        return True

    def vol_utilization(self) -> float:
        return self.volume_used / self.total_volume

    def weight_utilization(self) -> float:
        return self.current_weight / self.max_weight


def simulate_packing(order: List[int], boxes: List[Box],
                     cont_dims: Tuple, max_weight: float) -> dict:
    container = Container3D(cont_dims, max_weight)
    for idx in order:
        container.try_place(boxes[idx])

    return {
        'packed_count': len(container.placed),
        'total':        len(boxes),
        'vol_util':     container.vol_utilization(),
        'weight_util':  container.weight_utilization(),
        'vol_used':     container.volume_used,
        'weight_used':  container.current_weight,
        'placed':       container.placed,
    }

class GeneticAlgorithm3DBinPacking:
    def __init__(self, boxes: List[Box], cont_dims: Tuple[float, float, float], max_w: float):
        self.boxes = boxes
        self.cont_dims = cont_dims
        self.max_w = max_w
        self.n = len(boxes)

    def create_individual(self) -> List[int]:
        ind = list(range(self.n))
        random.shuffle(ind)
        return ind

    def fitness(self, individual: List[int]) -> Tuple[float, dict]:
        result = simulate_packing(individual, self.boxes, self.cont_dims, self.max_w)
        pack_ratio = result['packed_count'] / result['total']
        score = pack_ratio * 0.65 + result['vol_util'] * 0.35
        return score, result

    # CORRECCIÓN BUG 4: Order Crossover (OX) — válido para permutaciones
    def order_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = self.n
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[a:b + 1] = p1[a:b + 1]
        in_segment = set(p1[a:b + 1])
        fill = [x for x in p2 if x not in in_segment]
        j = 0
        for i in list(range(b + 1, n)) + list(range(0, a)):
            child[i] = fill[j]
            j += 1
        return child

    def mutate(self, individual: List[int], rate: float = 0.1) -> List[int]:
        """Mutación por intercambio de dos posiciones (preserva la permutación)."""
        ind = individual[:]
        for i in range(self.n):
            if random.random() < rate:
                j = random.randint(0, self.n - 1)
                ind[i], ind[j] = ind[j], ind[i]
        return ind

    def run(self, population_size=100, generations=50, mutation_rate=0.1):
        print("=" * 65)
        print("  ALGORITMO GENÉTICO — 3D BIN PACKING CON PESO")
        print("=" * 65)
        print(f"  Población:     {population_size}")
        print(f"  Generaciones:  {generations}")
        print(f"  Mutación:      {mutation_rate}")
        print(f"  Cajas totales: {self.n}")
        print(f"  Contenedor:    {self.cont_dims[0]}×{self.cont_dims[1]}×{self.cont_dims[2]} cm")
        print(f"  Peso máximo:   {self.max_w} kg")
        print("=" * 65)

        population = [self.create_individual() for _ in range(population_size)]
        best_individual = None
        best_fitness = -float('inf')
        best_result = None
        start = time.time()

        for gen in range(generations):

            evaluated = []
            for ind in population:
                f, result = self.fitness(ind)
                evaluated.append((f, result, ind))
            evaluated.sort(key=lambda x: x[0], reverse=True)

            if evaluated[0][0] > best_fitness:
                best_fitness = evaluated[0][0]
                best_result  = evaluated[0][1]
                best_individual = evaluated[0][2][:]

            r = evaluated[0][1]
            elapsed = time.time() - start
            print(f"  Gen {gen + 1:3d}/{generations} | "
                  f"Fitness: {evaluated[0][0]:.4f} | "
                  f"Cajas: {r['packed_count']:2d}/{r['total']} | "
                  f"Vol: {r['vol_util'] * 100:5.1f}% | "
                  f"Peso: {r['weight_util'] * 100:5.1f}% | "
                  f"t: {elapsed:.1f}s")

            elite_size    = max(2, population_size // 10)
            max_parent_idx = min(elite_size * 3, len(evaluated) - 1)
            next_pop = [ind for _, _, ind in evaluated[:elite_size]]

            while len(next_pop) < population_size:
                if random.random() < 0.7:
                    p1 = evaluated[random.randint(0, max_parent_idx)][2]
                    p2 = evaluated[random.randint(0, max_parent_idx)][2]
                    child = self.order_crossover(p1, p2)
                else:
                    parent = evaluated[random.randint(0, max_parent_idx)][2]
                    child = self.mutate(parent, mutation_rate)
                next_pop.append(child)

            population = next_pop[:population_size]

        print(f"\n  Tiempo total: {time.time() - start:.1f}s")
        return best_individual, best_result


# Datos y reporte final
def parse_data(filepath: str = None) -> Tuple[Tuple[float, float, float], float, List[Box]]:
   
    if filepath is None:
        if len(sys.argv) < 2:
            raise ValueError("Uso: python 3DBPPGA.py <archivo_entrada.txt>")
        filepath = sys.argv[1]

    cont_dims = None
    max_w     = float('inf')   
    boxes     = []
    in_table  = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()

            m = re.match(r'#\s*Bin dimensions.*?:\s*\(([^)]+)\)', stripped)
            if m:
                parts = [float(x) for x in m.group(1).split(',')]
                cont_dims = tuple(parts[:3])
                continue

            m = re.match(r'#\s*Max weight\s*:\s*(.*)', stripped)
            if m:
                val = m.group(1).strip()
                if val:
                    max_w = float(val)
                continue

            if stripped.startswith('#'):
                continue

            if re.match(r'^-+', stripped):
                in_table = True
                continue

            if re.match(r'\s*id\s+quantity', stripped):
                continue

            if in_table and stripped:
                cols = stripped.split()
                if len(cols) >= 5:
                    id_     = cols[0]
                    qty     = int(cols[1])
                    length  = float(cols[2])
                    width   = float(cols[3])
                    height  = float(cols[4])
                    weight  = float(cols[5]) if len(cols) >= 6 else 0.0
                    for _ in range(qty):
                        boxes.append(Box((length, width, height), weight, id_))

    if cont_dims is None:
        raise ValueError("No se encontraron dimensiones del contenedor en el archivo.")
    if not boxes:
        raise ValueError("No se encontraron cajas en el archivo.")

    total_vol = sum(b.dims[0] * b.dims[1] * b.dims[2] for b in boxes)
    total_w   = sum(b.weight for b in boxes)
    cont_vol  = cont_dims[0] * cont_dims[1] * cont_dims[2]
    peso_str  = f"{max_w} kg" if max_w != float('inf') else "sin restricción"

    print(f"Archivo cargado:     {filepath}")
    print(f"Contenedor:          {cont_dims[0]}×{cont_dims[1]}×{cont_dims[2]}")
    print(f"Cajas cargadas:      {len(boxes)}")
    print(f"Volumen total cajas: {total_vol / 1e6:.3f} m³  "
          f"(contenedor: {cont_vol / 1e6:.3f} m³ → teórico {total_vol / cont_vol * 100:.1f}%)")
    print(f"Peso total cajas:    {total_w:.1f} kg / {peso_str}")
    return cont_dims, max_w, boxes


def print_final_report(best_result: dict, cont_dims: Tuple, max_w: float):
    print("\n" + "=" * 80)
    print("  RESULTADO FINAL — 3D BIN PACKING CON RESTRICCIÓN DE PESO")
    print("=" * 80)
    r = best_result

    print(f"  Cajas empacadas:   {r['packed_count']:2d} / {r['total']}")
    print(f"  Volumen utilizado: {r['vol_used'] / 1e6:.3f} cm³  ({r['vol_util'] * 100:.1f}%)")
    print(f"  Peso utilizado:    {r['weight_used']:.1f} kg / {max_w} kg "
          f"({r['weight_util'] * 100:.1f}%)")
    print()
    print(f"  {'ID':>4}  {'Dims orig (cm)':>16}  {'Dims colocada':>16}  "
          f"{'Posición (x,y,z)':>20}  {'Peso':>8}")
    print("  " + "-" * 72)
    for p in r['placed']:
        box  = p['box']
        orig = tuple(int(d) for d in box.dims)
        dims = tuple(int(d) for d in p['dims'])
        pos  = tuple(int(v) for v in p['pos'])
        rotated = " R" if dims != orig else "  "
        print(f"  {box.id:>4}  {str(orig):>16}  {str(dims):>16}{rotated}"
              f"  {str(pos):>20}  {box.weight:>7.2f}kg")
    print("=" * 80)


if __name__ == "__main__":
    random.seed(42)

    # Uso: python 3DBPPGA.py <archivo.txt>
    # Ejemplo: python 3DBPPGA.py 3dBPP_1.txt
    cont_dims, max_w, boxes = parse_data()
    print()

    ga = GeneticAlgorithm3DBinPacking(boxes, cont_dims, max_w)
    best_ind, best_result = ga.run(
        population_size=100,
        generations=50,
        mutation_rate=0.1
    )

    print_final_report(best_result, cont_dims, max_w)