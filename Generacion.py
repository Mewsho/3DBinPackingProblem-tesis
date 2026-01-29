import random
import math
import argparse

def generar_instancia(
    n_tipos=10,                    # Número de tipos únicos de paquetes (IDs)
    max_cantidad=5,                # Máximo número de unidades por tipo
    dim_contenedor=(240, 180, 200),  # cm: Largo x Ancho x Alto
    peso_max_contenedor=1000,      # kg
    rango_dim_paquetes=(10, 100),  # cm mín y máx por eje
    rango_peso_paquetes=(0.5, 50), # kg mín y máx
    semilla=None,
    archivo_salida=None
):
    if semilla is not None:
        random.seed(semilla)
    
    L, W, H = dim_contenedor
    volumen_contenedor = L * W * H
    peso_total = 0
    volumen_total = 0
    paquetes = []  # Lista de (ID, cantidad, l, w, h, peso_unitario)

    for i in range(1, n_tipos + 1):
        l = random.randint(*rango_dim_paquetes)
        w = random.randint(*rango_dim_paquetes)
        h = random.randint(*rango_dim_paquetes)
        peso = round(random.uniform(*rango_peso_paquetes), 2)
        cantidad = random.randint(1, max_cantidad)  # Aleatorio entre 1 y max_cantidad
        
        volumen = l * w * h
        volumen_total += volumen * cantidad
        peso_total += peso * cantidad
        paquetes.append((i, cantidad, l, w, h, peso))

    # Cálculo teórico mínimo de contenedores
    contenedores_por_volumen = math.ceil(volumen_total / volumen_contenedor)
    contenedores_por_peso = math.ceil(peso_total / peso_max_contenedor)
    contenedores_teoricos = max(contenedores_por_volumen, contenedores_por_peso)

    # Construir salida
    lineas = []
    lineas.append(f"Dimensiones del contenedor (cm): {L} x {W} x {H}")
    lineas.append(f"Peso máximo del contenedor (kg): {peso_max_contenedor}")
    lineas.append(f"Número teórico mínimo de contenedores: {contenedores_teoricos}")
    lineas.append(f"Volumen total de paquetes (cm³): {volumen_total}")
    lineas.append(f"Peso total de paquetes (kg): {round(peso_total, 2)}")
    
    # Línea amigable
    total_paquetes = sum(cant for _, cant, _, _, _, _ in paquetes)
    lineas.append(f"\n📌 Se generaron {n_tipos} tipos de paquetes, con un total de {total_paquetes} unidades.")
    
    lineas.append("")
    lineas.append("ID\tCantidad\tLargo\tAncho\tAlto\tPeso(kg)")
    for pid, cant, l, w, h, peso in paquetes:
        lineas.append(f"{pid}\t{cant}\t{l}\t{w}\t{h}\t{peso}")

    salida_completa = "\n".join(lineas)

    # Imprimir en consola
    print(salida_completa)

    # Guardar en archivo si se especifica
    if archivo_salida:
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            f.write(salida_completa)
        print(f"\n✅ Instancia guardada en: {archivo_salida}")

    return {
        "contenedor": {"L": L, "W": W, "H": H, "peso_max": peso_max_contenedor},
        "paquetes": paquetes,
        "teorico_min_contenedores": contenedores_teoricos
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de instancias para 3D Bin Packing con peso y múltiples unidades por tipo.")
    parser.add_argument("--tipos", type=int, default=10, help="Número de tipos únicos de paquetes (IDs)")
    parser.add_argument("--max_cantidad", type=int, default=5, help="Máximo número de unidades por tipo")
    parser.add_argument("--L", type=int, default=240, help="Largo del contenedor (cm)")
    parser.add_argument("--W", type=int, default=180, help="Ancho del contenedor (cm)")
    parser.add_argument("--H", type=int, default=200, help="Alto del contenedor (cm)")
    parser.add_argument("--peso_max", type=int, default=1000, help="Peso máximo del contenedor (kg)")
    parser.add_argument("--min_dim", type=int, default=10, help="Dimensión mínima de paquete (cm)")
    parser.add_argument("--max_dim", type=int, default=100, help="Dimensión máxima de paquete (cm)")
    parser.add_argument("--min_peso", type=float, default=0.5, help="Peso mínimo unitario (kg)")
    parser.add_argument("--max_peso", type=float, default=50.0, help="Peso máximo unitario (kg)")
    parser.add_argument("--semilla", type=int, default=None, help="Semilla para replicabilidad")
    parser.add_argument("--salida", type=str, default=None, help="Archivo .txt de salida (ej: instancia_1.txt)")

    args = parser.parse_args()

    generar_instancia(
        n_tipos=args.tipos,
        max_cantidad=args.max_cantidad,
        dim_contenedor=(args.L, args.W, args.H),
        peso_max_contenedor=args.peso_max,
        rango_dim_paquetes=(args.min_dim, args.max_dim),
        rango_peso_paquetes=(args.min_peso, args.max_peso),
        semilla=args.semilla,
        archivo_salida=args.salida
    )