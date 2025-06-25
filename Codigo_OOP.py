"""
PVNS (Variable Neighborhood Search) con POPMUSIC para PCTSP 
(Precedence-Constrained Colored Traveling Salesman Problem)
Implementación Orientada a Objetos

CARACTERÍSTICAS IMPLEMENTADAS:
Lectura de archivos .pctsp con matriz de distancias y restricciones de accesibilidad
Construcción greedy de solución inicial respetando restricciones de accesibilidad y precedencia
Operadores de perturbación: CEM, CMI, C2-EX, P3-EX (con verificación de precedencia)
Búsqueda local: 2-opt intra-ruta, relocate y exchange inter-ruta (con verificación de precedencia)
Soporte completo para restricciones de precedencia entre ciudades
POPMUSIC (Partial Optimization Metaheuristic Under Special Intensification Conditions)
- Descomposición del problema en subproblemas más pequeños
- Optimización local intensiva en subconjuntos de ciudades
- Recombinación de soluciones parciales óptimas
Optimización para instancias grandes (muestreo aleatorio, límites de evaluación)
Reportes detallados con gap respecto al óptimo conocido
Verificación final de todas las restricciones de precedencia
Visualización de soluciones con gráficos automáticos
Medición de tiempo de ejecución

AUTORES: Implementación para el problema PCTSP
FECHA: 2025
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import time
import os
import time
import os


class PCTSPInstance:
    """Clase para representar una instancia del problema PCTSP"""
    
    def __init__(self, archivo_path):
        """
        Inicializa una instancia PCTSP desde un archivo.
        
        Args:
            archivo_path: Ruta al archivo .pctsp
        """
        self.datos = self._leer_archivo_pctsp(archivo_path)
        self.nombre = self.datos['nombre']
        self.num_vendedores = self.datos['num_vendedores']
        self.dimension = self.datos['dimension']
        self.matriz_distancias = self.datos['matriz_distancias']
        self.accesibilidad = self.datos['accesibilidad']
        self.depot = self.datos['depot']
        
        # Variables derivadas
        self.ciudades = list(range(self.dimension))
        self.vendedores = list(range(self.num_vendedores))
        self.precedence = []  # Se configurará externamente si es necesario
        
    def _leer_archivo_pctsp(self, archivo_path):
        """
        Lee un archivo .pctsp y extrae toda la información.
        
        Args:
            archivo_path: Ruta al archivo
            
        Returns:
            dict: Diccionario con todos los datos de la instancia
        """
        with open(archivo_path, 'r') as f:
            lineas = f.readlines()
        
        nombre = ""
        num_vendedores = 0
        dimension = 0
        matriz_distancias = None
        accesibilidad = {}
        depot = 1
        
        i = 0
        leyendo_matriz = False
        leyendo_accesibilidad = False
        fila_matriz = 0
        
        while i < len(lineas):
            linea = lineas[i].strip()
            
            if linea.startswith("NAME:"):
                nombre = linea.split(":")[1].strip()
            elif linea.startswith("SALESMEN:") or linea.startswith("SALESME:"):
                num_vendedores = int(linea.split(":")[1].strip())
            elif linea.startswith("DIMENSION:"):
                dimension = int(linea.split(":")[1].strip())
            elif linea.startswith("EDGE_WEIGHT_SECTION"):
                leyendo_matriz = True
                matriz_distancias = np.zeros((dimension, dimension), dtype=int)
                fila_matriz = 0
            elif linea.startswith("GCTSP_SET_SECTION"):
                leyendo_matriz = False
                leyendo_accesibilidad = True
            elif linea.startswith("DEPOT_SECTION"):
                leyendo_accesibilidad = False
            elif linea.startswith("EOF"):
                break
            elif leyendo_matriz and linea and linea[0].isdigit():
                if linea == str(dimension):
                    pass
                else:
                    elementos = linea.split()
                    for col, valor in enumerate(elementos):
                        if col < dimension and fila_matriz < dimension:
                            matriz_distancias[fila_matriz][col] = int(valor)
                    fila_matriz += 1
            elif leyendo_accesibilidad and linea and linea[0].isdigit():
                partes = linea.split()
                if len(partes) > 2 and partes[-1] == "-1":
                    ciudad = int(partes[0]) - 1
                    vendedores = [int(x) - 1 for x in partes[1:-1]]
                    accesibilidad[ciudad] = vendedores
            elif linea.startswith("DEPOT_SECTION"):
                i += 1
                if i < len(lineas):
                    depot_line = lineas[i].strip()
            
            i += 1
            
        return {
            'nombre': nombre,
            'num_vendedores': num_vendedores,
            'dimension': dimension,
            'matriz_distancias': matriz_distancias,
            'accesibilidad': accesibilidad,
            'depot': depot - 1
        }
    
    def distancia(self, i, j):
        """
        Calcula la distancia entre dos ciudades.
        
        Args:
            i, j: Índices de las ciudades
            
        Returns:
            float: Distancia entre las ciudades (inf si no hay conexión)
        """
        return float('inf') if self.matriz_distancias[i][j] == -1 else self.matriz_distancias[i][j]
    
    def configurar_precedencias(self, precedencias):
        """
        Configura las restricciones de precedencia.
        
        Args:
            precedencias: Lista de tuplas (i, j) donde i debe ir antes que j
        """
        self.precedence = precedencias
    
    def get_optimo_conocido(self):
        """
        Retorna el óptimo conocido para esta instancia si está disponible.
        
        Returns:
            int or None: Valor óptimo conocido
        """
        optimos_conocidos = {
            "eil101": 598,
            "lin318": 40445,
            "rat575": 6924,
            "pcb3038.3": 142121,
            "pcb3038.4": 141738,
            "pcb3038.6": 141778,
            "pr1002.3": 267883,
            "pr1002.6": 266994,
            "u1432.3": 155029,
            "u1432.6": 154782,
            "vm1748.3": 347014,
            "vm1748.6": 347432
        }
        
        for clave, valor in optimos_conocidos.items():
            if clave in self.nombre:
                return valor
        return None


class PCTSPSolution:
    """Clase para representar una solución del problema PCTSP"""
    
    def __init__(self, instancia):
        """
        Inicializa una solución vacía.
        
        Args:
            instancia: Instancia PCTSPInstance
        """
        self.instancia = instancia
        self.rutas = {k: [] for k in instancia.vendedores}
        self._costo_cache = None
    
    def copy(self):
        """
        Crea una copia profunda de la solución.
        
        Returns:
            PCTSPSolution: Nueva instancia con los mismos datos
        """
        nueva_solucion = PCTSPSolution(self.instancia)
        nueva_solucion.rutas = {k: r.copy() for k, r in self.rutas.items()}
        return nueva_solucion
    
    def calcular_costo_ruta(self, ruta):
        """
        Calcula el costo de una ruta individual.
        
        Args:
            ruta: Lista de ciudades en la ruta
            
        Returns:
            float: Costo total de la ruta
        """
        if len(ruta) == 0:
            return 0
            
        costo = self.instancia.distancia(self.instancia.depot, ruta[0])
        for i in range(len(ruta) - 1):
            costo += self.instancia.distancia(ruta[i], ruta[i+1])
        costo += self.instancia.distancia(ruta[-1], self.instancia.depot)
        return costo
    
    def calcular_costo_total(self):
        """
        Calcula el costo total de la solución.
        
        Returns:
            float: Costo total de todas las rutas
        """
        total = 0
        for vendedor, ruta in self.rutas.items():
            if len(ruta) == 0:
                continue
            costo_ruta = self.calcular_costo_ruta(ruta)
            if costo_ruta == float('inf'):
                return float('inf')
            total += costo_ruta
        return total
    
    def invalidar_cache(self):
        """Invalida el cache del costo."""
        self._costo_cache = None
    
    def get_costo(self):
        """
        Obtiene el costo total con cache.
        
        Returns:
            float: Costo total
        """
        if self._costo_cache is None:
            self._costo_cache = self.calcular_costo_total()
        return self._costo_cache
    
    def verificar_precedencias(self, ruta, ciudad_insertada, posicion):
        """
        Verifica si insertar una ciudad en una posición específica viola precedencias.
        
        Args:
            ruta: Lista de ciudades en la ruta actual
            ciudad_insertada: Ciudad que se quiere insertar
            posicion: Posición donde se quiere insertar
            
        Returns:
            bool: True si la inserción es válida
        """
        for (i, j) in self.instancia.precedence:
            if ciudad_insertada == i and j in ruta:
                pos_j = ruta.index(j)
                if pos_j < posicion:
                    return False
            
            if ciudad_insertada == j and i in ruta:
                pos_i = ruta.index(i)
                if pos_i >= posicion:
                    return False
        
        return True
    
    def verificar_precedencias_ruta_completa(self, ruta):
        """
        Verifica si una ruta completa respeta todas las restricciones de precedencia.
        
        Args:
            ruta: Lista de ciudades en orden
            
        Returns:
            bool: True si la ruta es válida
        """
        for (i, j) in self.instancia.precedence:
            if i in ruta and j in ruta:
                pos_i = ruta.index(i)
                pos_j = ruta.index(j)
                if pos_i >= pos_j:
                    return False
        return True
    
    def verificar_precedencias_solucion(self):
        """
        Verifica que todas las restricciones de precedencia se cumplan en la solución.
        
        Returns:
            tuple: (bool, list) - (True si todas se cumplen, lista de violaciones)
        """
        violaciones = []
        
        for vendedor, ruta in self.rutas.items():
            if len(ruta) > 1:
                for (i, j) in self.instancia.precedence:
                    if i in ruta and j in ruta:
                        pos_i = ruta.index(i)
                        pos_j = ruta.index(j)
                        if pos_i >= pos_j:
                            violaciones.append(
                                f"Vendedor {vendedor+1}: Ciudad {i+1} debería ir antes que {j+1}, "
                                f"pero están en posiciones {pos_i+1} y {pos_j+1}"
                            )
        
        return len(violaciones) == 0, violaciones
    
    def get_vendedor_ciudad(self, ciudad):
        """
        Encuentra el vendedor asignado a una ciudad específica.
        
        Args:
            ciudad: ID de la ciudad a buscar
            
        Returns:
            int or None: ID del vendedor que visita la ciudad, None si no está asignada
        """
        for vendedor, ruta in self.rutas.items():
            if ciudad in ruta:
                return vendedor
        return None
    
    def get_estadisticas(self):
        """
        Obtiene estadísticas de la solución.
        
        Returns:
            dict: Diccionario con estadísticas
        """
        rutas_activas = sum(1 for ruta in self.rutas.values() if ruta)
        ciudades_visitadas = sum(len(ruta) for ruta in self.rutas.values())
        total_ciudades = self.instancia.dimension - 1  # Excluyendo depot
        cobertura = (ciudades_visitadas / total_ciudades * 100) if total_ciudades > 0 else 0
        
        costos_por_vendedor = []
        for ruta in self.rutas.values():
            if ruta:
                costos_por_vendedor.append(self.calcular_costo_ruta(ruta))
        
        return {
            'rutas_activas': rutas_activas,
            'ciudades_visitadas': ciudades_visitadas,
            'total_ciudades': total_ciudades,
            'cobertura': cobertura,
            'costos_por_vendedor': costos_por_vendedor,
            'costo_total': self.get_costo()
        }


class ConstructorSolucionGreedy:
    """Clase para construir soluciones iniciales usando algoritmo greedy"""
    
    def __init__(self, instancia):
        """
        Inicializa el constructor.
        
        Args:
            instancia: Instancia PCTSPInstance
        """
        self.instancia = instancia
    
    def construir_solucion_topologica(self):
        """
        Construye una solución inicial usando estrategia greedy topológica.
        
        Returns:
            PCTSPSolution: Solución inicial construida
        """
        solucion = PCTSPSolution(self.instancia)
        depot = self.instancia.depot
        ciudades_a_asignar = [c for c in self.instancia.ciudades if c != depot]
        
        # Ordenar ciudades por número de vendedores que pueden visitarlas (ascendente)
        ciudades_por_restriccion = sorted(
            ciudades_a_asignar, 
            key=lambda c: len(self.instancia.accesibilidad.get(c, []))
        )
        
        for ciudad in ciudades_por_restriccion:
            vendedores_permitidos = self.instancia.accesibilidad.get(ciudad, [])
            if not vendedores_permitidos:
                continue
            
            mejor_costo = float('inf')
            mejor_asignacion = None
            
            # Evaluar inserción en cada vendedor permitido
            for vendedor in vendedores_permitidos:
                ruta_actual = solucion.rutas[vendedor]
                
                # Evaluar todas las posiciones posibles
                for pos in range(len(ruta_actual) + 1):
                    if solucion.verificar_precedencias(ruta_actual, ciudad, pos):
                        costo_insercion = self._calcular_costo_insercion(
                            solucion, vendedor, ciudad, pos
                        )
                        if costo_insercion < mejor_costo:
                            mejor_costo = costo_insercion
                            mejor_asignacion = (vendedor, pos)
            
            # Realizar la mejor asignación encontrada
            if mejor_asignacion:
                vendedor, posicion = mejor_asignacion
                solucion.rutas[vendedor].insert(posicion, ciudad)
                solucion.invalidar_cache()
        
        return solucion
    
    def _calcular_costo_insercion(self, solucion, vendedor, ciudad, posicion):
        """
        Calcula el costo de insertar una ciudad en una posición específica.
        
        Args:
            solucion: Solución actual
            vendedor: Vendedor donde insertar
            ciudad: Ciudad a insertar
            posicion: Posición de inserción
            
        Returns:
            float: Costo de la inserción
        """
        ruta = solucion.rutas[vendedor]
        depot = self.instancia.depot
        
        if len(ruta) == 0:
            # Ruta vacía: depot -> ciudad -> depot
            d1 = self.instancia.distancia(depot, ciudad)
            d2 = self.instancia.distancia(ciudad, depot)
            if d1 != float('inf') and d2 != float('inf'):
                return d1 + d2
            return float('inf')
        
        if posicion == 0:
            # Insertar al principio
            d_depot_ciudad = self.instancia.distancia(depot, ciudad)
            d_ciudad_primera = self.instancia.distancia(ciudad, ruta[0])
            d_depot_primera = self.instancia.distancia(depot, ruta[0])
            
            if (d_depot_ciudad != float('inf') and 
                d_ciudad_primera != float('inf') and 
                d_depot_primera != float('inf')):
                return d_depot_ciudad + d_ciudad_primera - d_depot_primera
        
        elif posicion == len(ruta):
            # Insertar al final
            ultimo = ruta[-1]
            d1 = self.instancia.distancia(ultimo, ciudad)
            d2 = self.instancia.distancia(ciudad, depot)
            d_original = self.instancia.distancia(ultimo, depot)
            
            if (d1 != float('inf') and d2 != float('inf') and d_original != float('inf')):
                return d1 + d2 - d_original
        
        else:
            # Insertar entre dos ciudades
            anterior = ruta[posicion-1]
            siguiente = ruta[posicion]
            d_ant_ciudad = self.instancia.distancia(anterior, ciudad)
            d_ciudad_sig = self.instancia.distancia(ciudad, siguiente)
            d_ant_sig = self.instancia.distancia(anterior, siguiente)
            
            if (d_ant_ciudad != float('inf') and 
                d_ciudad_sig != float('inf') and 
                d_ant_sig != float('inf')):
                return d_ant_ciudad + d_ciudad_sig - d_ant_sig
        
        return float('inf')


class OperadoresBusquedaLocal:
    """Clase con operadores de búsqueda local"""
    
    def __init__(self, instancia):
        """
        Inicializa los operadores.
        
        Args:
            instancia: Instancia PCTSPInstance
        """
        self.instancia = instancia
    
    def two_opt(self, solucion, vendedor):
        """
        Aplica 2-opt a una ruta individual.
        
        Args:
            solucion: Solución actual
            vendedor: Vendedor cuya ruta optimizar
            
        Returns:
            list: Nueva ruta optimizada
        """
        ruta = solucion.rutas[vendedor]
        if len(ruta) < 4:
            return ruta[:]
        
        mejor_ruta = ruta[:]
        mejor_costo = solucion.calcular_costo_ruta(ruta)
        
        # Limitar iteraciones para instancias grandes
        max_iteraciones = min(50, len(ruta) * 2)
        iteraciones = 0
        
        mejoro = True
        while mejoro and iteraciones < max_iteraciones:
            mejoro = False
            iteraciones += 1
            
            # Muestreo para instancias grandes
            if len(ruta) > 20:
                indices_i = random.sample(range(len(ruta) - 1), min(10, len(ruta) - 1))
            else:
                indices_i = range(len(ruta) - 1)
            
            for i in indices_i:
                if len(ruta) > 20:
                    posibles_j = list(range(i + 2, len(ruta)))
                    if posibles_j:
                        indices_j = random.sample(posibles_j, min(5, len(posibles_j)))
                    else:
                        continue
                else:
                    indices_j = range(i + 2, len(ruta))
                
                for j in indices_j:
                    nueva_ruta = ruta[:i+1] + ruta[i+1:j+1][::-1] + ruta[j+1:]
                    
                    if solucion.verificar_precedencias_ruta_completa(nueva_ruta):
                        nuevo_costo = solucion.calcular_costo_ruta(nueva_ruta)
                        
                        if nuevo_costo < mejor_costo:
                            mejor_ruta = nueva_ruta
                            mejor_costo = nuevo_costo
                            ruta = nueva_ruta
                            mejoro = True
                            break
                if mejoro:
                    break
        
        return mejor_ruta
    
    def relocate_inter_route(self, solucion):
        """
        Mueve una ciudad de una ruta a otra.
        
        Args:
            solucion: Solución actual
            
        Returns:
            PCTSPSolution: Nueva solución con el mejor movimiento aplicado
        """
        nueva_solucion = solucion.copy()
        costo_original = nueva_solucion.get_costo()
        mejor_costo = costo_original
        mejor_movimiento = None
        
        max_evaluaciones = 1000
        evaluaciones = 0
        
        for vendedor_origen in self.instancia.vendedores:
            if len(nueva_solucion.rutas[vendedor_origen]) > 0:
                ciudades_a_evaluar = min(10, len(nueva_solucion.rutas[vendedor_origen]))
                indices = random.sample(
                    range(len(nueva_solucion.rutas[vendedor_origen])), 
                    ciudades_a_evaluar
                )
                
                for pos_ciudad in indices:
                    ciudad = nueva_solucion.rutas[vendedor_origen][pos_ciudad]
                    vendedores_permitidos = self.instancia.accesibilidad.get(ciudad, [])
                    
                    for vendedor_destino in vendedores_permitidos:
                        if vendedor_destino != vendedor_origen:
                            max_posiciones = min(5, len(nueva_solucion.rutas[vendedor_destino]) + 1)
                            posiciones = random.sample(
                                range(len(nueva_solucion.rutas[vendedor_destino]) + 1), 
                                max_posiciones
                            )
                            
                            for pos_destino in posiciones:
                                evaluaciones += 1
                                if evaluaciones > max_evaluaciones:
                                    break
                                
                                # Evaluar movimiento
                                solucion_temp = nueva_solucion.copy()
                                ciudad_movida = solucion_temp.rutas[vendedor_origen].pop(pos_ciudad)
                                
                                if solucion_temp.verificar_precedencias(
                                    solucion_temp.rutas[vendedor_destino], ciudad_movida, pos_destino
                                ):
                                    solucion_temp.rutas[vendedor_destino].insert(pos_destino, ciudad_movida)
                                    
                                    if (solucion_temp.verificar_precedencias_ruta_completa(
                                        solucion_temp.rutas[vendedor_origen]) and
                                        solucion_temp.verificar_precedencias_ruta_completa(
                                        solucion_temp.rutas[vendedor_destino])):
                                        
                                        nuevo_costo = solucion_temp.calcular_costo_total()
                                        if nuevo_costo < mejor_costo:
                                            mejor_costo = nuevo_costo
                                            mejor_movimiento = (vendedor_origen, pos_ciudad, vendedor_destino, pos_destino)
                            
                            if evaluaciones > max_evaluaciones:
                                break
                    if evaluaciones > max_evaluaciones:
                        break
            if evaluaciones > max_evaluaciones:
                break
        
        # Aplicar el mejor movimiento
        if mejor_movimiento:
            vendedor_origen, pos_ciudad, vendedor_destino, pos_destino = mejor_movimiento
            ciudad_movida = nueva_solucion.rutas[vendedor_origen].pop(pos_ciudad)
            nueva_solucion.rutas[vendedor_destino].insert(pos_destino, ciudad_movida)
            nueva_solucion.invalidar_cache()
        
        return nueva_solucion
    
    def exchange_inter_route(self, solucion):
        """
        Intercambia ciudades entre dos rutas diferentes.
        
        Args:
            solucion: Solución actual
            
        Returns:
            PCTSPSolution: Nueva solución con el mejor intercambio aplicado
        """
        nueva_solucion = solucion.copy()
        costo_original = nueva_solucion.get_costo()
        mejor_costo = costo_original
        mejor_intercambio = None
        
        max_evaluaciones = 500
        evaluaciones = 0
        
        for vendedor1 in self.instancia.vendedores:
            for vendedor2 in self.instancia.vendedores:
                if (vendedor1 < vendedor2 and 
                    len(nueva_solucion.rutas[vendedor1]) > 0 and 
                    len(nueva_solucion.rutas[vendedor2]) > 0):
                    
                    max_ciudades1 = min(8, len(nueva_solucion.rutas[vendedor1]))
                    max_ciudades2 = min(8, len(nueva_solucion.rutas[vendedor2]))
                    
                    posiciones1 = random.sample(range(len(nueva_solucion.rutas[vendedor1])), max_ciudades1)
                    posiciones2 = random.sample(range(len(nueva_solucion.rutas[vendedor2])), max_ciudades2)
                    
                    for pos1 in posiciones1:
                        for pos2 in posiciones2:
                            evaluaciones += 1
                            if evaluaciones > max_evaluaciones:
                                break
                            
                            ciudad1 = nueva_solucion.rutas[vendedor1][pos1]
                            ciudad2 = nueva_solucion.rutas[vendedor2][pos2]
                            
                            vendedores_ciudad1 = self.instancia.accesibilidad.get(ciudad1, [])
                            vendedores_ciudad2 = self.instancia.accesibilidad.get(ciudad2, [])
                            
                            if vendedor2 in vendedores_ciudad1 and vendedor1 in vendedores_ciudad2:
                                solucion_temp = nueva_solucion.copy()
                                solucion_temp.rutas[vendedor1][pos1] = ciudad2
                                solucion_temp.rutas[vendedor2][pos2] = ciudad1
                                
                                if (solucion_temp.verificar_precedencias_ruta_completa(
                                    solucion_temp.rutas[vendedor1]) and
                                    solucion_temp.verificar_precedencias_ruta_completa(
                                    solucion_temp.rutas[vendedor2])):
                                    
                                    nuevo_costo = solucion_temp.calcular_costo_total()
                                    if nuevo_costo < mejor_costo:
                                        mejor_costo = nuevo_costo
                                        mejor_intercambio = (vendedor1, pos1, vendedor2, pos2)
                        
                        if evaluaciones > max_evaluaciones:
                            break
                    if evaluaciones > max_evaluaciones:
                        break
            if evaluaciones > max_evaluaciones:
                break
        
        # Aplicar el mejor intercambio
        if mejor_intercambio:
            vendedor1, pos1, vendedor2, pos2 = mejor_intercambio
            nueva_solucion.rutas[vendedor1][pos1], nueva_solucion.rutas[vendedor2][pos2] = \
                nueva_solucion.rutas[vendedor2][pos2], nueva_solucion.rutas[vendedor1][pos1]
            nueva_solucion.invalidar_cache()
        
        return nueva_solucion


class OperadoresPerturbacion:
    """Clase con operadores de perturbación para diversificación"""
    
    def __init__(self, instancia):
        """
        Inicializa los operadores.
        
        Args:
            instancia: Instancia PCTSPInstance
        """
        self.instancia = instancia
    
    def cem(self, solucion):
        """
        City Exchange Move - Intercambia ciudades dentro de cada ruta.
        
        Args:
            solucion: Solución actual
            
        Returns:
            PCTSPSolution: Solución perturbada
        """
        nueva_solucion = solucion.copy()
        
        for vendedor in self.instancia.vendedores:
            if len(nueva_solucion.rutas[vendedor]) >= 2:
                max_intentos = 10
                for _ in range(max_intentos):
                    i, j = sorted(random.sample(range(len(nueva_solucion.rutas[vendedor])), 2))
                    ruta_temp = nueva_solucion.rutas[vendedor].copy()
                    ruta_temp[i], ruta_temp[j] = ruta_temp[j], ruta_temp[i]
                    
                    if nueva_solucion.verificar_precedencias_ruta_completa(ruta_temp):
                        nueva_solucion.rutas[vendedor] = ruta_temp
                        nueva_solucion.invalidar_cache()
                        break
        
        return nueva_solucion
    
    def cmi(self, solucion):
        """
        City Movement Intra-route - Mueve subsecuencias dentro de cada ruta.
        
        Args:
            solucion: Solución actual
            
        Returns:
            PCTSPSolution: Solución perturbada
        """
        nueva_solucion = solucion.copy()
        
        for vendedor in self.instancia.vendedores:
            if len(nueva_solucion.rutas[vendedor]) >= 3:
                max_intentos = 10
                for _ in range(max_intentos):
                    i = random.randint(0, len(nueva_solucion.rutas[vendedor]) - 3)
                    subseq = nueva_solucion.rutas[vendedor][i:i+2]
                    pos = random.randint(0, len(nueva_solucion.rutas[vendedor]) - 2)
                    
                    ruta_temp = nueva_solucion.rutas[vendedor].copy()
                    del ruta_temp[i:i+2]
                    ruta_temp[pos:pos] = subseq
                    
                    if nueva_solucion.verificar_precedencias_ruta_completa(ruta_temp):
                        nueva_solucion.rutas[vendedor] = ruta_temp
                        nueva_solucion.invalidar_cache()
                        break
        
        return nueva_solucion
    
    def c2_ex(self, solucion):
        """
        2-City Exchange - Invierte subsecuencias de 2 ciudades.
        
        Args:
            solucion: Solución actual
            
        Returns:
            PCTSPSolution: Solución perturbada
        """
        nueva_solucion = solucion.copy()
        
        for vendedor in self.instancia.vendedores:
            if len(nueva_solucion.rutas[vendedor]) >= 4:
                max_intentos = 10
                for _ in range(max_intentos):
                    i = random.randint(0, len(nueva_solucion.rutas[vendedor]) - 3)
                    
                    ruta_temp = nueva_solucion.rutas[vendedor].copy()
                    ruta_temp[i:i+2] = reversed(ruta_temp[i:i+2])
                    
                    if nueva_solucion.verificar_precedencias_ruta_completa(ruta_temp):
                        nueva_solucion.rutas[vendedor] = ruta_temp
                        nueva_solucion.invalidar_cache()
                        break
        
        return nueva_solucion
    
    def p3_ex(self, solucion):
        """
        3-Point Exchange - Reordena subsecuencias de 3 segmentos.
        
        Args:
            solucion: Solución actual
            
        Returns:
            PCTSPSolution: Solución perturbada
        """
        nueva_solucion = solucion.copy()
        
        for vendedor in self.instancia.vendedores:
            if len(nueva_solucion.rutas[vendedor]) >= 6:
                max_intentos = 10
                for _ in range(max_intentos):
                    i = random.randint(0, len(nueva_solucion.rutas[vendedor]) - 5)
                    b1 = nueva_solucion.rutas[vendedor][i:i+1]
                    b2 = nueva_solucion.rutas[vendedor][i+1:i+3]
                    b3 = nueva_solucion.rutas[vendedor][i+3:i+4]
                    
                    ruta_temp = nueva_solucion.rutas[vendedor].copy()
                    ruta_temp[i:i+4] = b2 + b1 + b3
                    
                    if nueva_solucion.verificar_precedencias_ruta_completa(ruta_temp):
                        nueva_solucion.rutas[vendedor] = ruta_temp
                        nueva_solucion.invalidar_cache()
                        break
        
        return nueva_solucion


class VisualizadorSoluciones:
    """Clase para visualizar soluciones del PCTSP"""
    
    def __init__(self, instancia):
        """
        Inicializa el visualizador.
        
        Args:
            instancia: Instancia PCTSPInstance
        """
        self.instancia = instancia
    
    def generar_coordenadas_ciudades(self):
        """
        Genera coordenadas aleatorias para visualización.
        
        Returns:
            list: Lista de coordenadas (x, y) para cada ciudad
        """
        np.random.seed(42)  # Para reproducibilidad
        dimension = self.instancia.dimension
        
        if dimension > 500:
            # Distribución en círculos concéntricos para instancias grandes
            coordenadas = [(0, 0)]  # Depot en el centro
            ciudades_restantes = dimension - 1
            num_circulos = max(3, int(np.sqrt(ciudades_restantes / 10)))
            
            for circulo in range(num_circulos):
                radio = (circulo + 1) * (1000 / num_circulos)
                ciudades_en_circulo = max(6, ciudades_restantes // (num_circulos - circulo))
                
                for i in range(min(ciudades_en_circulo, ciudades_restantes)):
                    angulo = (2 * np.pi * i) / ciudades_en_circulo + np.random.uniform(-0.2, 0.2)
                    x = radio * np.cos(angulo) + np.random.uniform(-50, 50)
                    y = radio * np.sin(angulo) + np.random.uniform(-50, 50)
                    coordenadas.append((x, y))
                    ciudades_restantes -= 1
                    
                    if ciudades_restantes <= 0:
                        break
                if ciudades_restantes <= 0:
                    break
            
            # Agregar ciudades restantes
            while len(coordenadas) < dimension:
                x = np.random.uniform(-1200, 1200)
                y = np.random.uniform(-1200, 1200)
                coordenadas.append((x, y))
        else:
            # Cuadrícula con variación para instancias pequeñas
            grid_size = int(np.sqrt(dimension)) + 1
            coordenadas = []
            
            for i in range(dimension):
                x_base = (i % grid_size) * 120
                y_base = (i // grid_size) * 120
                x = x_base + np.random.uniform(-40, 40)
                y = y_base + np.random.uniform(-40, 40)
                coordenadas.append((x, y))
        
        return coordenadas
    
    def generar_grafico_solucion(self, solucion, titulo="Solución PCTSP"):
        """
        Genera un gráfico de la solución.
        
        Args:
            solucion: PCTSPSolution a visualizar
            titulo: Título del gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura generada
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
            import matplotlib.pyplot as plt
        
        coordenadas = self.generar_coordenadas_ciudades()
        depot = self.instancia.depot
        
        # Configurar figura
        plt.figure(figsize=(16, 12))
        
        # Colores para cada vendedor
        colores = ['#FF4444', '#4444FF', '#44FF44', '#FF8800', '#8844FF', 
                  '#FF4488', '#44FFFF', '#888888', '#88FF44', '#FF8844']
        
        # Dibujar depot
        depot_x, depot_y = coordenadas[depot]
        depot_size = 100 if len(self.instancia.ciudades) < 100 else 60
        plt.scatter(depot_x, depot_y, c='black', s=depot_size, marker='s', 
                   label='Depot', zorder=5, edgecolors='white', linewidth=1)
        
        if len(self.instancia.ciudades) < 200:
            plt.annotate('DEPOT', (depot_x, depot_y), xytext=(0, -20), 
                        textcoords='offset points', fontsize=9, fontweight='bold', 
                        color='black', ha='center')
        
        # Dibujar rutas
        for vendedor, ruta in solucion.rutas.items():
            if ruta:
                color = colores[vendedor % len(colores)]
                
                # Coordenadas de la ruta completa
                ruta_completa = [depot] + ruta + [depot]
                x_coords = [coordenadas[ciudad][0] for ciudad in ruta_completa]
                y_coords = [coordenadas[ciudad][1] for ciudad in ruta_completa]
                
                # Líneas
                line_width = 0.8 if len(self.instancia.ciudades) > 500 else 1.5
                plt.plot(x_coords, y_coords, color=color, linewidth=line_width, 
                        alpha=0.7, label=f'Vendedor {vendedor + 1} ({len(ruta)} ciudades)')
                
                # Puntos
                point_size = (8 if len(self.instancia.ciudades) > 500 else 
                            20 if len(self.instancia.ciudades) > 100 else 35)
                for ciudad in ruta:
                    x, y = coordenadas[ciudad]
                    plt.scatter(x, y, c=color, s=point_size, alpha=0.8, zorder=3, 
                              edgecolors='white', linewidth=0.3)
        
        # Configurar gráfico
        plt.title(f'{titulo}\nCosto Total: {solucion.get_costo():,.0f}', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Coordenada X', fontsize=14)
        plt.ylabel('Coordenada Y', fontsize=14)
        
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, 
                  frameon=True, fancybox=True, shadow=True)
        
        plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Información adicional
        estadisticas = solucion.get_estadisticas()
        info_text = f"Instancia: {self.instancia.nombre}\n"
        info_text += f"Ciudades: {estadisticas['total_ciudades']:,}\n"
        info_text += f"Vendedores: {len(self.instancia.vendedores)}\n"
        info_text += f"Rutas activas: {estadisticas['rutas_activas']}"
        
        plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes, 
                verticalalignment='bottom', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                         alpha=0.8, edgecolor='navy'))
        
        plt.tight_layout()
        plt.axis('equal')
        plt.subplots_adjust(left=0.08, right=0.85, top=0.92, bottom=0.08)
        
        # Guardar automáticamente
        if not os.path.exists("Rutas"):
            os.makedirs("Rutas")
        
        nombre_archivo = f"Rutas/solucion_OOP_{self.instancia.nombre}_{estadisticas['rutas_activas']}vendedores_costo{solucion.get_costo():.0f}.png"
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {nombre_archivo}")
        
        plt.show()
        return plt.gcf()


class POPMUSIC:
    """
    Implementación de POPMUSIC (Partial Optimization Metaheuristic Under Special Intensification Conditions)
    para el problema PCTSP con parámetros optimizados
    """
    
    def __init__(self, instancia, tamano_subproblema=None, solapamiento=None, 
                 max_iteraciones=None, umbral_mejora=0.01):
        """
        Inicializa POPMUSIC con parámetros adaptativos optimizados.
        
        Args:
            instancia: Instancia PCTSPInstance
            tamano_subproblema: Tamaño máximo de cada subproblema (None = automático)
            solapamiento: Número de ciudades que se solapan (None = automático)
            max_iteraciones: Número máximo de iteraciones (None = automático)
            umbral_mejora: Umbral mínimo de mejora para continuar iteraciones
        """
        self.instancia = instancia
        self.umbral_mejora = umbral_mejora
        
        # Configuración adaptativa basada en el tamaño de la instancia
        if tamano_subproblema is None:
            if instancia.dimension > 500:
                self.tamano_subproblema = min(30, max(25, instancia.dimension // 20))
            elif instancia.dimension > 100:
                self.tamano_subproblema = min(25, max(18, instancia.dimension // 15))
            else:
                self.tamano_subproblema = min(18, max(15, instancia.dimension // 8))
        else:
            self.tamano_subproblema = tamano_subproblema
        
        # Solapamiento como 35% del tamaño del subproblema
        if solapamiento is None:
            self.solapamiento = max(3, min(10, int(self.tamano_subproblema * 0.35)))
        else:
            self.solapamiento = solapamiento
        
        # Iteraciones máximas adaptativas
        if max_iteraciones is None:
            if instancia.dimension > 500:
                self.max_iteraciones = 7
            elif instancia.dimension > 100:
                self.max_iteraciones = 5
            else:
                self.max_iteraciones = 4
        else:
            self.max_iteraciones = max_iteraciones
        
        self.busqueda_local = OperadoresBusquedaLocal(instancia)
        self.constructor = ConstructorSolucionGreedy(instancia)
        
        print(f"POPMUSIC configurado: subproblema={self.tamano_subproblema}, "
              f"solapamiento={self.solapamiento}, max_iter={self.max_iteraciones}")
    
    def generar_subproblemas_mejorado(self, solucion):
        """
        Genera subproblemas con estrategias mejoradas:
        1. Subproblemas basados en densidad de precedencias
        2. Subproblemas de ciudades críticas (alto costo)
        3. Subproblemas geográficos tradicionales
        
        Args:
            solucion: Solución actual
            
        Returns:
            list: Lista de subproblemas prioritizados
        """
        subproblemas = []
        
        # 1. Subproblemas basados en precedencias críticas
        if self.instancia.precedence:
            subprob_precedencia = self._generar_subproblema_precedencia_critica(solucion)
            if len(subprob_precedencia) >= 4:
                subproblemas.append({
                    'ciudades': subprob_precedencia,
                    'tipo': 'precedencia_critica',
                    'prioridad': 1
                })
        
        # 2. Subproblemas de aristas costosas
        subprob_costoso = self._generar_subproblema_aristas_costosas(solucion)
        if len(subprob_costoso) >= 4:
            subproblemas.append({
                'ciudades': subprob_costoso,
                'tipo': 'aristas_costosas',
                'prioridad': 2
            })
        
        # 3. Subproblemas geográficos tradicionales (mejorados)
        subprob_geograficos = self._generar_subproblemas_geograficos_mejorados(solucion)
        subproblemas.extend(subprob_geograficos)
        
        # Ordenar por prioridad
        subproblemas.sort(key=lambda x: x.get('prioridad', 3))
        
        return subproblemas
    
    def _generar_subproblema_precedencia_critica(self, solucion):
        """
        Genera subproblema enfocado en precedencias que causan mayor restricción.
        """
        # Identificar precedencias que más limitan la solución
        precedencias_criticas = []
        
        for i, j in self.instancia.precedence:
            # Calcular si esta precedencia está causando rutas largas
            vendedor_i = solucion.get_vendedor_ciudad(i)
            vendedor_j = solucion.get_vendedor_ciudad(j)
            
            if vendedor_i != vendedor_j or vendedor_i is None or vendedor_j is None:
                # Precedencia violada o en rutas diferentes - crítica
                precedencias_criticas.extend([i, j])
        
        # Agregar ciudades cercanas para completar el subproblema
        ciudades_subproblema = set(precedencias_criticas)
        
        for ciudad_base in list(ciudades_subproblema):
            if len(ciudades_subproblema) >= self.tamano_subproblema:
                break
            
            # Buscar ciudades cercanas no asignadas o mal posicionadas
            candidatos = []
            for vendedor, ruta in solucion.rutas.items():
                for ciudad in ruta:
                    if ciudad not in ciudades_subproblema:
                        dist = self.instancia.distancia(ciudad_base, ciudad)
                        if dist != float('inf'):
                            candidatos.append((ciudad, dist))
            
            candidatos.sort(key=lambda x: x[1])
            for ciudad, _ in candidatos[:3]:
                ciudades_subproblema.add(ciudad)
                if len(ciudades_subproblema) >= self.tamano_subproblema:
                    break
        
        return list(ciudades_subproblema)
    
    def _generar_subproblema_aristas_costosas(self, solucion):
        """
        Genera subproblema con las aristas más costosas de la solución actual.
        """
        aristas_costos = []
        
        # Identificar aristas más costosas en todas las rutas
        for vendedor, ruta in solucion.rutas.items():
            for i in range(len(ruta) - 1):
                ciudad_actual = ruta[i]
                ciudad_siguiente = ruta[i + 1]
                costo = self.instancia.distancia(ciudad_actual, ciudad_siguiente)
                if costo != float('inf'):
                    aristas_costos.append((ciudad_actual, ciudad_siguiente, costo))
        
        # Ordenar por costo descendente y tomar las más costosas
        aristas_costos.sort(key=lambda x: x[2], reverse=True)
        
        ciudades_subproblema = set()
        for ciudad1, ciudad2, _ in aristas_costos[:self.tamano_subproblema // 2]:
            ciudades_subproblema.add(ciudad1)
            ciudades_subproblema.add(ciudad2)
            
            if len(ciudades_subproblema) >= self.tamano_subproblema:
                break
        
        return list(ciudades_subproblema)
    
    def _generar_subproblemas_geograficos_mejorados(self, solucion):
        """
        Versión mejorada de generación de subproblemas geográficos.
        """
        subproblemas = []
        
        # Generar subproblemas con solapamiento inteligente
        for vendedor, ruta in solucion.rutas.items():
            if len(ruta) > self.tamano_subproblema:
                # Solapamiento adaptativo basado en la longitud de la ruta
                solapamiento_local = min(self.solapamiento, len(ruta) // 4)
                paso = max(1, self.tamano_subproblema - solapamiento_local)
                
                for i in range(0, len(ruta), paso):
                    inicio = max(0, i - solapamiento_local // 2)
                    fin = min(len(ruta), i + self.tamano_subproblema)
                    subproblema = ruta[inicio:fin]
                    
                    if len(subproblema) >= 4:
                        subproblemas.append({
                            'ciudades': subproblema,
                            'vendedor_principal': vendedor,
                            'tipo': 'geografico_mejorado',
                            'prioridad': 3
                        })
        
        return subproblemas
    
    def resolver_subproblema(self, solucion, subproblema):
        """
        Resuelve un subproblema específico usando optimización local intensiva adaptativa.
        
        Args:
            solucion: Solución actual
            subproblema: Diccionario con información del subproblema
            
        Returns:
            PCTSPSolution: Solución mejorada para el subproblema
        """
        ciudades_subproblema = subproblema['ciudades']
        
        if len(ciudades_subproblema) < 3:
            return solucion
        
        # Crear una solución temporal solo con las ciudades del subproblema
        solucion_temp = self._extraer_subsolucion(solucion, ciudades_subproblema)
        
        # Aplicar optimización intensiva
        mejor_subsolucion = solucion_temp.copy()
        mejor_costo = mejor_subsolucion.get_costo()
        
        # Número de pasadas adaptativo según tamaño del subproblema
        if len(ciudades_subproblema) <= 15:
            max_pasadas = 8
        elif len(ciudades_subproblema) <= 25:
            max_pasadas = 6
        else:
            max_pasadas = 4
        
        sin_mejora = 0
        for pasada in range(max_pasadas):
            costo_antes_pasada = mejor_costo
            
            # 2-opt en cada ruta del subproblema
            for vendedor in self.instancia.vendedores:
                if len(mejor_subsolucion.rutas[vendedor]) > 3:
                    nueva_ruta = self.busqueda_local.two_opt(mejor_subsolucion, vendedor)
                    mejor_subsolucion.rutas[vendedor] = nueva_ruta
                    mejor_subsolucion.invalidar_cache()
            
            # Relocalizaciones entre rutas (más frecuentes para subproblemas críticos)
            probabilidad_reloc = 0.7 if subproblema.get('prioridad', 3) <= 2 else 0.4
            if random.random() < probabilidad_reloc:
                subsolucion_reloc = self.busqueda_local.relocate_inter_route(mejor_subsolucion)
                if subsolucion_reloc.get_costo() < mejor_costo:
                    mejor_subsolucion = subsolucion_reloc
                    mejor_costo = mejor_subsolucion.get_costo()
            
            # Intercambios entre rutas
            probabilidad_exch = 0.5 if subproblema.get('prioridad', 3) <= 2 else 0.3
            if random.random() < probabilidad_exch:
                subsolucion_exch = self.busqueda_local.exchange_inter_route(mejor_subsolucion)
                if subsolucion_exch.get_costo() < mejor_costo:
                    mejor_subsolucion = subsolucion_exch
                    mejor_costo = mejor_subsolucion.get_costo()
            
            # Verificar convergencia
            mejora_relativa = (costo_antes_pasada - mejor_costo) / max(costo_antes_pasada, 1)
            if mejora_relativa < 0.001:  # Mejora menor al 0.1%
                sin_mejora += 1
                if sin_mejora >= 2:
                    break
            else:
                sin_mejora = 0
        
        # Reintegrar la subsolucion optimizada en la solución completa
        solucion_mejorada = self._integrar_subsolucion(solucion, mejor_subsolucion, ciudades_subproblema)
        
        return solucion_mejorada
    
    def _extraer_subsolucion(self, solucion, ciudades_subproblema):
        """
        Extrae una subsolucion que contiene solo las ciudades especificadas.
        
        Args:
            solucion: Solución completa
            ciudades_subproblema: Lista de ciudades a incluir
            
        Returns:
            PCTSPSolution: Subsolucion
        """
        subsolucion = PCTSPSolution(self.instancia)
        
        for vendedor, ruta in solucion.rutas.items():
            nueva_ruta = [c for c in ruta if c in ciudades_subproblema]
            subsolucion.rutas[vendedor] = nueva_ruta
        
        return subsolucion
    
    def _integrar_subsolucion(self, solucion_original, subsolucion_optimizada, ciudades_subproblema):
        """
        Integra una subsolucion optimizada de vuelta en la solución completa.
        
        Args:
            solucion_original: Solución completa original
            subsolucion_optimizada: Subsolucion optimizada
            ciudades_subproblema: Ciudades del subproblema
            
        Returns:
            PCTSPSolution: Solución completa con mejoras integradas
        """
        solucion_integrada = solucion_original.copy()
        
        # Remover las ciudades del subproblema de la solución original
        for vendedor in self.instancia.vendedores:
            solucion_integrada.rutas[vendedor] = [
                c for c in solucion_integrada.rutas[vendedor] 
                if c not in ciudades_subproblema
            ]
        
        # Insertar las ciudades optimizadas en sus nuevas posiciones
        for vendedor, ruta_optimizada in subsolucion_optimizada.rutas.items():
            if ruta_optimizada:
                # Encontrar la mejor posición para insertar cada ciudad
                for ciudad in ruta_optimizada:
                    mejor_posicion = 0
                    mejor_costo = float('inf')
                    
                    for pos in range(len(solucion_integrada.rutas[vendedor]) + 1):
                        if solucion_integrada.verificar_precedencias(
                            solucion_integrada.rutas[vendedor], ciudad, pos
                        ):
                            # Calcular costo de inserción
                            solucion_temp = solucion_integrada.copy()
                            solucion_temp.rutas[vendedor].insert(pos, ciudad)
                            costo = solucion_temp.calcular_costo_ruta(solucion_temp.rutas[vendedor])
                            
                            if costo < mejor_costo:
                                mejor_costo = costo
                                mejor_posicion = pos
                    
                    # Insertar en la mejor posición
                    solucion_integrada.rutas[vendedor].insert(mejor_posicion, ciudad)
        
        solucion_integrada.invalidar_cache()
        return solucion_integrada
    
    def aplicar_popmusic(self, solucion, max_iteraciones=None):
        """
        Aplica el algoritmo POPMUSIC completo con criterios de parada mejorados.
        
        Args:
            solucion: Solución inicial
            max_iteraciones: Número máximo de iteraciones (None = usar configurado)
            
        Returns:
            PCTSPSolution: Solución mejorada
        """
        if max_iteraciones is None:
            max_iteraciones = self.max_iteraciones
            
        mejor_solucion = solucion.copy()
        mejor_costo = mejor_solucion.get_costo()
        costo_inicial = mejor_costo
        
        print(f"Iniciando POPMUSIC - Costo inicial: {mejor_costo:,.0f}")
        print(f"Parámetros: subproblema={self.tamano_subproblema}, solapamiento={self.solapamiento}")
        
        for iteracion in range(max_iteraciones):
            print(f"POPMUSIC iteración {iteracion + 1}/{max_iteraciones}")
            
            # Generar subproblemas con estrategia mejorada
            subproblemas = self.generar_subproblemas_mejorado(mejor_solucion)
            print(f"  Generados {len(subproblemas)} subproblemas")
            
            costo_antes_iteracion = mejor_costo
            mejoro_iteracion = False
            
            # Resolver cada subproblema
            for i, subproblema in enumerate(subproblemas):
                if len(subproblema['ciudades']) >= 3:
                    tipo = subproblema.get('tipo', 'desconocido')
                    prioridad = subproblema.get('prioridad', 3)
                    print(f"  Subproblema {i+1}/{len(subproblemas)} "
                          f"(tamaño: {len(subproblema['ciudades'])}, tipo: {tipo}, prioridad: {prioridad})")
                    
                    solucion_mejorada = self.resolver_subproblema(mejor_solucion, subproblema)
                    costo_mejorado = solucion_mejorada.get_costo()
                    
                    if costo_mejorado < mejor_costo:
                        mejora = ((mejor_costo - costo_mejorado) / mejor_costo) * 100
                        print(f"    Mejora encontrada: {mejor_costo:,.0f} -> {costo_mejorado:,.0f} ({mejora:.2f}%)")
                        mejor_solucion = solucion_mejorada
                        mejor_costo = costo_mejorado
                        mejoro_iteracion = True
            
            # Evaluar criterio de parada
            mejora_iteracion = ((costo_antes_iteracion - mejor_costo) / max(costo_antes_iteracion, 1)) * 100
            
            if mejora_iteracion < self.umbral_mejora * 100:
                print(f"  Mejora insuficiente ({mejora_iteracion:.3f}% < {self.umbral_mejora*100:.1f}%) - terminando")
                break
            else:
                print(f"  Costo después de iteración {iteracion + 1}: {mejor_costo:,.0f} (mejora: {mejora_iteracion:.2f}%)")
        
        mejora_total = ((costo_inicial - mejor_costo) / max(costo_inicial, 1)) * 100
        print(f"POPMUSIC completado - Costo final: {mejor_costo:,.0f} (mejora total: {mejora_total:.2f}%)")
        return mejor_solucion


class AlgoritmoPVNS:
    """Clase principal que implementa el algoritmo PVNS con POPMUSIC"""
    
    def __init__(self, instancia, usar_popmusic=True):
        """
        Inicializa el algoritmo.
        
        Args:
            instancia: Instancia PCTSPInstance
            usar_popmusic: Si se debe usar POPMUSIC o no
        """
        self.instancia = instancia
        self.constructor = ConstructorSolucionGreedy(instancia)
        self.busqueda_local = OperadoresBusquedaLocal(instancia)
        self.perturbacion = OperadoresPerturbacion(instancia)
        self.visualizador = VisualizadorSoluciones(instancia)
        self.usar_popmusic = usar_popmusic
        
        if self.usar_popmusic:
            # Configuración automática con parámetros optimizados
            # Los parámetros se configuran automáticamente en POPMUSIC
            self.popmusic = POPMUSIC(instancia)
    
    def busqueda_local_intensiva(self, solucion):
        """
        Aplica búsqueda local intensiva combinando múltiples operadores.
        
        Args:
            solucion: Solución actual
            
        Returns:
            PCTSPSolution: Solución mejorada
        """
        solucion_mejorada = solucion.copy()
        
        # 1. Aplicar 2-opt a cada ruta individual
        for vendedor in self.instancia.vendedores:
            if len(solucion_mejorada.rutas[vendedor]) > 3:
                nueva_ruta = self.busqueda_local.two_opt(solucion_mejorada, vendedor)
                solucion_mejorada.rutas[vendedor] = nueva_ruta
                solucion_mejorada.invalidar_cache()
        
        # 2. Relocalizaciones entre rutas (ocasionalmente)
        if random.random() < 0.3:
            solucion_mejorada = self.busqueda_local.relocate_inter_route(solucion_mejorada)        
        # 3. Intercambios entre rutas (ocasionalmente)
        if random.random() < 0.2:
            solucion_mejorada = self.busqueda_local.exchange_inter_route(solucion_mejorada)
        
        return solucion_mejorada
    
    def ejecutar(self, iteraciones=500):
        """
        Ejecuta el algoritmo PVNS siguiendo la secuencia exacta del paper:
        1. POPMUSIC inicial (genera aristas candidatas)
        2. Construcción greedy topológica
        3. PVNS iterativo (Shaking: CEM+CMI, Búsqueda local: C2-EX+P3-EX)
        
        Args:
            iteraciones: Número de iteraciones a ejecutar
            
        Returns:
            tuple: (mejor_solucion, tiempo_transcurrido)
        """
        tiempo_inicio = time.time()
        
        # Obtener óptimo conocido
        optimo_conocido = self.instancia.get_optimo_conocido()
        
        print(f"Iniciando PVNS para {self.instancia.nombre}")
        
        # PASO 1: POPMUSIC INICIAL (solo una vez según el paper)
        if self.usar_popmusic:
            print("PASO 1: Aplicando POPMUSIC para generar aristas candidatas...")
            solucion_inicial = self.constructor.construir_solucion_topologica()
            solucion_con_aristas = self.popmusic.aplicar_popmusic(solucion_inicial, max_iteraciones=2)
            print(f"POPMUSIC completado. Aristas candidatas generadas.")
        else:
            solucion_con_aristas = None
        
        # PASO 2: CONSTRUCCIÓN GREEDY TOPOLÓGICA
        print("\nPASO 2: Construcción greedy topológica...")
        solucion_actual = self.constructor.construir_solucion_topologica()
        if self.usar_popmusic and solucion_con_aristas:
            # Usar información de POPMUSIC si está disponible
            if solucion_con_aristas.get_costo() < solucion_actual.get_costo():
                solucion_actual = solucion_con_aristas
        
        mejor_solucion = solucion_actual.copy()
        mejor_costo = mejor_solucion.get_costo()
        sin_mejora = 0
        
        print(f"Solución inicial: {mejor_costo:,.0f}")
        
        # PASO 3: PVNS ITERATIVO
        print(f"\nPASO 3: Iniciando PVNS iterativo ({iteraciones} iteraciones)...")
        
        for i in range(iteraciones):
            # Mostrar progreso cada 10 iteraciones
            if (i + 1) % 10 == 0:
                if optimo_conocido:
                    gap_actual = ((mejor_costo - optimo_conocido) / optimo_conocido * 100)
                    print(f"Iteración {i+1}/{iteraciones} - Gap: {gap_actual:.2f}%")
                else:
                    print(f"Iteración {i+1}/{iteraciones} - Costo: {mejor_costo:,.0f}")
            
            # SHAKING: CEM + CMI (según el paper)
            solucion_perturbada = self.perturbacion.cem(solucion_actual)
            solucion_perturbada = self.perturbacion.cmi(solucion_perturbada)
            
            # BÚSQUEDA LOCAL: C2-EX + P3-EX (según el paper)
            solucion_local = self.perturbacion.c2_ex(solucion_perturbada)
            solucion_local = self.perturbacion.p3_ex(solucion_local)
            
            # Búsqueda local adicional con 2-opt
            solucion_mejorada = self.busqueda_local_intensiva(solucion_local)
            costo_mejorado = solucion_mejorada.get_costo()
            
            # Aceptación de solución
            if costo_mejorado < mejor_costo:
                mejor_solucion = solucion_mejorada.copy()
                mejor_costo = costo_mejorado
                solucion_actual = solucion_mejorada
                sin_mejora = 0
                
                # Búsqueda intensiva si estamos cerca del óptimo
                if optimo_conocido:
                    gap_actual = ((mejor_costo - optimo_conocido) / optimo_conocido * 100)
                    if gap_actual <= 3.0:
                        for _ in range(3):
                            solucion_intensiva = self.busqueda_local_intensiva(mejor_solucion)
                            if solucion_intensiva.get_costo() < mejor_costo:
                                mejor_solucion = solucion_intensiva
                                mejor_costo = solucion_intensiva.get_costo()
            else:
                sin_mejora += 1
                # Aceptar solución perturbada para mantener diversidad
                solucion_actual = solucion_mejorada
            
            # Diversificación: reconstruir si no hay mejora por mucho tiempo
            if sin_mejora >= 50:
                solucion_actual = self.constructor.construir_solucion_topologica()
                sin_mejora = 0
        
        tiempo_fin = time.time()
        tiempo_transcurrido = tiempo_fin - tiempo_inicio
        
        print(f"\nPVNS completado.")
        print(f"Costo final: {mejor_costo:,.0f}")
        
        # Mostrar resumen
        self._mostrar_resumen_final(mejor_solucion, optimo_conocido, tiempo_transcurrido)
        
        # Generar gráfico
        print("\nGenerando gráfico de la solución...")
        titulo_grafico = f"Solución PVNS {'con POPMUSIC' if self.usar_popmusic else ''} - {self.instancia.nombre}"
        self.visualizador.generar_grafico_solucion(mejor_solucion, titulo_grafico)
        
        return mejor_solucion, tiempo_transcurrido
    
    def _mostrar_resumen_final(self, solucion, optimo_conocido, tiempo_transcurrido):
        """
        Muestra un resumen completo de la solución.
        
        Args:            solucion: Mejor solución encontrada
            optimo_conocido: Óptimo conocido (si existe)
            tiempo_transcurrido: Tiempo de ejecución
        """
        print("\n" + "=" * 80)
        print("RESUMEN FINAL DE LA EJECUCIÓN (PVNS con POPMUSIC según paper)")
        print("=" * 80)
        
        # Información del archivo
        print(f"Archivo ejecutado: {self.instancia.nombre}")
        print(f"Número de ciudades: {len(self.instancia.ciudades) - 1}")
        print(f"Número de vendedores: {len(self.instancia.vendedores)}")
        print(f"Depot: Ciudad {self.instancia.depot + 1}")
        
        # Información del costo
        costo_final = solucion.get_costo()
        print(f"\nCOSTO TOTAL: {costo_final:,.0f}")
        if optimo_conocido:
            gap = ((costo_final - optimo_conocido) / optimo_conocido * 100)
            print(f"Óptimo conocido: {optimo_conocido:,.0f}")
            print(f"Gap final: {gap:.2f}%")
        else:
            print(f"Óptimo conocido: No disponible")
        
        # Tiempo de ejecución
        if tiempo_transcurrido < 60:
            print(f"Tiempo de ejecución: {tiempo_transcurrido:.2f} segundos")
        elif tiempo_transcurrido < 3600:
            minutos = int(tiempo_transcurrido // 60)
            segundos = tiempo_transcurrido % 60
            print(f"Tiempo de ejecución: {minutos} minutos y {segundos:.2f} segundos")
        else:
            horas = int(tiempo_transcurrido // 3600)
            minutos = int((tiempo_transcurrido % 3600) // 60)
            segundos = tiempo_transcurrido % 60
            print(f"Tiempo de ejecución: {horas} horas, {minutos} minutos y {segundos:.2f} segundos")
        
        # Detalles de rutas
        print(f"\nRUTAS DE LOS VENDEDORES:")
        print("-" * 80)
        
        estadisticas = solucion.get_estadisticas()
        
        for vendedor in self.instancia.vendedores:
            ruta = solucion.rutas[vendedor]
            if ruta:
                costo_ruta = solucion.calcular_costo_ruta(ruta)
                ruta_completa = [self.instancia.depot + 1] + [c + 1 for c in ruta] + [self.instancia.depot + 1]
                ruta_str = " -> ".join(map(str, ruta_completa))
                
                print(f"Vendedor {vendedor + 1}:")
                print(f"  Ruta: {ruta_str}")
                print(f"  Ciudades visitadas: {len(ruta)}")
                print(f"  Costo: {costo_ruta:,.0f}")
                print()
            else:
                print(f"Vendedor {vendedor + 1}: Sin ciudades asignadas")
                print()
        
        # Estadísticas finales
        print("ESTADÍSTICAS:")
        print(f"  Vendedores activos: {estadisticas['rutas_activas']}/{len(self.instancia.vendedores)}")
        print(f"  Ciudades visitadas: {estadisticas['ciudades_visitadas']}/{estadisticas['total_ciudades']}")
        print(f"  Cobertura: {estadisticas['cobertura']:.1f}%")
        
        # Verificar precedencias
        precedencias_ok, violaciones = solucion.verificar_precedencias_solucion()
        if precedencias_ok:
            print(f"  Restricciones de precedencia: TODAS CUMPLIDAS")
        else:
            print(f"  Restricciones de precedencia: {len(violaciones)} VIOLACIONES")
            for violacion in violaciones[:3]:
                print(f"    - {violacion}")
            if len(violaciones) > 3:
                print(f"    ... y {len(violaciones) - 3} más")
        
        print("=" * 80)


def main():
    """Función principal para ejecutar el algoritmo"""
    import os
    
    # Verificar si existe la carpeta PCTSP
    if not os.path.exists("PCTSP"):
        print("ERROR: La carpeta PCTSP no fue encontrada.")
        print("Por favor descarga las instancias desde: http://webhotel4.ruc.dk/~keld/research/LKH-3/")
        print("y extrae la carpeta PCTSP en el directorio raíz del proyecto.")
        print("\nCreando instancia de ejemplo para demostración...")
        
        # Crear una instancia de ejemplo pequeña para demostración
        instancia = crear_instancia_ejemplo()
    else:
        # Verificar si existe el archivo específico
        archivo_instancia = "PCTSP/INSTANCES/Regular/eil101.4.pctsp"
        if os.path.exists(archivo_instancia):
            print(f"Cargando instancia: {archivo_instancia}")
            instancia = PCTSPInstance(archivo_instancia)
        else:
            print(f"El archivo {archivo_instancia} no fue encontrado.")
            print("Buscando otras instancias disponibles...")
            
            # Buscar cualquier archivo .pctsp disponible
            instancia_encontrada = None
            for root, dirs, files in os.walk("PCTSP"):
                for file in files:
                    if file.endswith(".pctsp"):
                        instancia_encontrada = os.path.join(root, file)
                        break
                if instancia_encontrada:
                    break
            
            if instancia_encontrada:
                print(f"Usando instancia encontrada: {instancia_encontrada}")
                instancia = PCTSPInstance(instancia_encontrada)
            else:
                print("No se encontraron archivos .pctsp. Creando instancia de ejemplo...")
                instancia = crear_instancia_ejemplo()
    
    # Configurar precedencias (ejemplo)
    instancia.configurar_precedencias([(1, 3), (2, 4), (0, 5)])
    
    # Crear y ejecutar algoritmo con POPMUSIC habilitado
    print(f"\nEjecutando algoritmo para instancia: {instancia.nombre}")
    print(f"Ciudades: {instancia.dimension}, Vendedores: {instancia.num_vendedores}")
    
    algoritmo = AlgoritmoPVNS(instancia, usar_popmusic=True)
    mejor_solucion, tiempo = algoritmo.ejecutar(iteraciones=1000)
    
    return mejor_solucion, tiempo

def crear_instancia_ejemplo():
    """Crea una instancia de ejemplo pequeña para demostración"""
    import numpy as np
    import tempfile
    import os
    
    # Crear datos de ejemplo
    dimension = 10
    num_vendedores = 3
    
    # Crear archivo temporal con formato PCTSP
    contenido_pctsp = f"""NAME: ejemplo_demo
SALESMEN: {num_vendedores}
DIMENSION: {dimension}
EDGE_WEIGHT_SECTION
{dimension}
"""
    
    # Matriz de distancias aleatoria pero simétrica
    np.random.seed(42)
    matriz = np.random.randint(10, 100, (dimension, dimension))
    # Hacer simétrica
    for i in range(dimension):
        matriz[i][i] = 0
        for j in range(i+1, dimension):
            matriz[j][i] = matriz[i][j]
    
    # Agregar matriz al contenido
    for i in range(dimension):
        fila = " ".join(str(matriz[i][j]) for j in range(dimension))
        contenido_pctsp += fila + "\n"
    
    # Agregar restricciones de accesibilidad
    contenido_pctsp += "GCTSP_SET_SECTION\n"
    for i in range(1, dimension):  # Excluir depot (ciudad 0)
        # Cada ciudad puede ser visitada por 1-2 vendedores aleatorios
        vendedores_permitidos = np.random.choice(range(1, num_vendedores + 1), 
                                                size=np.random.randint(1, num_vendedores + 1), 
                                                replace=False)
        linea = f"{i+1} " + " ".join(str(v) for v in vendedores_permitidos) + " -1\n"
        contenido_pctsp += linea
    
    contenido_pctsp += "DEPOT_SECTION\n1\nEOF\n"
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pctsp', delete=False) as f:
        f.write(contenido_pctsp)
        archivo_temp = f.name
    
    # Crear instancia desde el archivo temporal
    instancia = PCTSPInstance(archivo_temp)
    instancia.nombre = "ejemplo_demo"
    
    # Limpiar archivo temporal
    os.unlink(archivo_temp)
    
    return instancia


if __name__ == "__main__":
    main()


