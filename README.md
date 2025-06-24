# Solver PCTSP - Problema del Vendedor Viajero Coloreado con Restricciones de Precedencia

## 📋 Descripción General

Este proyecto implementa un algoritmo de **Búsqueda en Vecindario Variable (VNS)** para resolver el **Problema del Vendedor Viajero Coloreado con Restricciones de Precedencia (PCTSP)**. El PCTSP es una extensión del clásico Problema del Vendedor Viajero que incorpora restricciones de color (restricciones de accesibilidad) y restricciones de precedencia entre ciudades.

## 🎯 Descripción del Problema

El Problema del Vendedor Viajero Coloreado con Restricciones de Precedencia (PCTSP) es un problema complejo de optimización donde:

- **Múltiples vendedores** deben visitar un conjunto de ciudades
- Cada vendedor solo puede visitar ciudades de **colores específicos** (restricciones de accesibilidad)
- Algunas ciudades deben ser visitadas **antes que otras** (restricciones de precedencia)
- El objetivo es **minimizar la distancia total de viaje**
- Todos los vendedores inician y terminan en un depósito central

### Características Clave:

- **Restricciones de color**: Cada ciudad tiene restricciones de accesibilidad (solo ciertos vendedores pueden visitarla)
- **Restricciones de precedencia**: Algunas ciudades deben ser visitadas en un orden específico
- **Optimización multi-ruta**: Múltiples vendedores con diferentes capacidades
- **Basado en depósito**: Todas las rutas inician y terminan en la misma ubicación

## 🔬 Fundamento de Investigación

Esta implementación está basada en los siguientes artículos de investigación:

1. **Zhang, J., Dai, B., Wang, Z., & Li, Y. (2022)**  
   _"Precedence-Constrained Colored Traveling Salesman Problem: An Augmented Variable Neighborhood Search Approach"_  
   IEEE Transactions on Cybernetics  
   [Enlace DOI](https://ieeexplore.ieee.org/abstract/document/9440778)

2. **Dong, Y., & Cai, C. (2019)**  
   _"A novel genetic algorithm for large scale colored balanced traveling salesman problem"_  
   Expert Systems with Applications, 119, 113–124  
   [Enlace DOI](https://www.sciencedirect.com/science/article/abs/pii/S0167739X1831728X)

## 🚀 Características

### ✅ Algoritmos Implementados

- **Búsqueda en Vecindario Variable (VNS)** con múltiples estructuras de vecindario
- **Construcción greedy** para generación de solución inicial
- **Operadores de búsqueda local**: 2-opt, reubicación, intercambio
- **Operadores de perturbación**: CEM, CMI, C2-EX, P3-EX
- **Verificación de restricciones de precedencia** durante todo el proceso de optimización

### ✅ Soporte del Problema

- **Formato de archivo PCTSP** (análisis de .pctsp)
- **Manejo de matriz de distancias**
- **Restricciones de accesibilidad** (ciudades coloreadas)
- **Restricciones de precedencia** entre ciudades
- **Enrutamiento multi-vendedor**

### ✅ Características de Optimización

- **Operaciones conscientes de restricciones** (todos los movimientos respetan precedencia y accesibilidad)
- **Búsqueda adaptativa** con intensificación y diversificación
- **Optimización de instancias grandes** (estrategias de muestreo para escalabilidad)
- **Análisis de brecha** con soluciones óptimas conocidas

### ✅ Visualización y Reportes

- **Visualización automática de rutas** con matplotlib
- **Reportes detallados de soluciones** con estadísticas
- **Métricas de rendimiento** (análisis de brecha, tiempo de ejecución)
- **Detección de violaciones de restricciones**

## 📁 Estructura del Proyecto

```
Proyecto/
├── Codigo_OOP.py           # Implementación principal (Orientado a Objetos)
├── Codigo.py               # Implementación alternativa
├── README.md               # Documentación del proyecto
├── PCTSP/                  # Instancias del problema y resultados (DESCARGAR SEPARADAMENTE)
│   ├── INSTANCES/          # Instancias de prueba
│   │   ├── Random/         # Instancias generadas aleatoriamente
│   │   └── Regular/        # Instancias benchmark estándar
│   ├── SOLUTIONS/          # Soluciones óptimas conocidas
│   ├── RESULTS/            # Resultados experimentales
│   └── TOURS/              # Tours de soluciones
└── Rutas/                  # Salidas de visualización generadas
```

> **⚠️ Importante**: La carpeta `PCTSP` no está incluida en este repositorio debido a su tamaño. Debe descargarse desde http://webhotel4.ruc.dk/~keld/research/LKH-3/ y colocarse en el directorio raíz del proyecto.

## 🛠️ Instalación y Requisitos

### Prerrequisitos

- Python 3.7+
- Paquetes requeridos:
  ```bash
  pip install numpy matplotlib
  ```

### Inicio Rápido

1. Clona o descarga el proyecto
2. **Descarga las instancias de prueba PCTSP**:
   - Las instancias de prueba deben descargarse desde: http://webhotel4.ruc.dk/~keld/research/LKH-3/
   - Estas instancias no están incluidas en el repositorio debido a su tamaño
   - Descarga y extrae la carpeta `PCTSP` en el directorio raíz del proyecto
   - Esta carpeta contiene los archivos necesarios para realizar pruebas del algoritmo
3. Instala las dependencias:
   ```bash
   pip install numpy matplotlib
   ```
4. Ejecuta el algoritmo principal:
   ```bash
   python Codigo_OOP.py
   ```

## 📊 Ejemplos de Uso

### Uso Básico

```python
from Codigo_OOP import PCTSPInstance, AlgoritmoPVNS

# Cargar una instancia del problema
instancia = PCTSPInstance("PCTSP/INSTANCES/Regular/pcb3038.6.pctsp")

# Configurar restricciones de precedencia (opcional)
instancia.configurar_precedencias([(10, 50), (25, 75), (5, 15)])

# Crear y ejecutar el algoritmo
algoritmo = AlgoritmoPVNS(instancia)
mejor_solucion, tiempo = algoritmo.ejecutar(iteraciones=1000)
```

### Configuración Avanzada

```python
# Crear algoritmo con parámetros personalizados
algoritmo = AlgoritmoPVNS(instancia)

# Ejecutar con diferentes conteos de iteraciones
solucion_rapida, _ = algoritmo.ejecutar(iteraciones=500)    # Ejecución rápida
solucion_intensiva, _ = algoritmo.ejecutar(iteraciones=2000) # Ejecución intensiva
```

## 🏗️ Arquitectura

La implementación sigue un diseño orientado a objetos con las siguientes clases principales:

### Clases Principales

- **`PCTSPInstance`**: Maneja la carga de instancias del problema y gestión de datos
- **`PCTSPSolution`**: Representa y valida soluciones
- **`AlgoritmoPVNS`**: Implementación del algoritmo principal

### Componentes del Algoritmo

- **`ConstructorSolucionGreedy`**: Genera soluciones iniciales
- **`OperadoresBusquedaLocal`**: Operadores de búsqueda local (2-opt, reubicación, intercambio)
- **`OperadoresPerturbacion`**: Operadores de perturbación para diversificación
- **`VisualizadorSoluciones`**: Visualización de soluciones y reportes

## 🔍 Detalles del Algoritmo

### 1. **Construcción de Solución Inicial**

- Enfoque greedy topológico
- Respeta restricciones de accesibilidad y precedencia
- Prioriza ciudades con restricciones primero

### 2. **Búsqueda Local**

- **2-opt**: Optimización intra-ruta
- **Reubicación**: Movimiento de ciudades inter-ruta
- **Intercambio**: Intercambio de ciudades inter-ruta
- Todas las operaciones verifican la satisfacción de restricciones

### 3. **Operadores de Perturbación**

- **CEM** (City Exchange Move): Intercambio de ciudades intra-ruta
- **CMI** (City Movement Intra-route): Reubicación de subsecuencias
- **C2-EX** (2-City Exchange): Reversión de segmentos pequeños
- **P3-EX** (3-Point Exchange): Reordenamiento de tres segmentos

### 4. **Manejo de Restricciones**

- **Verificación de precedencia**: Asegura el orden correcto de ciudades
- **Verificación de accesibilidad**: Valida asignaciones vendedor-ciudad
- **Validación en tiempo real**: Restricciones verificadas durante todas las operaciones

## 📈 Rendimiento y Resultados

### Instancias Benchmark

El algoritmo ha sido probado en instancias PCTSP estándar:

- **Instancias regulares**: eil101, lin318, pcb3038, pr1002, rat575, u1432, vm1748
- **Instancias aleatorias**: Varias series R.700.100
- **Rango de tamaño**: De 101 a 3038+ ciudades

### Métricas de Rendimiento

- **Análisis de brecha** contra soluciones óptimas conocidas
- **Seguimiento de tiempo de ejecución**
- **Análisis de convergencia** con historial de iteraciones
- **Verificación de satisfacción de restricciones**

### Resultados de Ejemplo

```
Instancia: pcb3038.6.pctsp
Ciudades: 3038
Vendedores: 6
Mejor Conocido: 141,778
Resultado del Algoritmo: ~142,500
Brecha: ~0.5%
Tiempo de Ejecución: ~15 minutos
```

## 🎨 Visualización

El algoritmo genera automáticamente visualizaciones detalladas:

- **Mapas de rutas** con diferentes colores para cada vendedor
- **Indicadores de accesibilidad** de ciudades
- **Resaltado de ubicación del depósito**
- **Superposición de estadísticas** de rendimiento
- **Guardado automático de archivos** en formato PNG

### Archivos Generados

- Guardados en el directorio `Rutas/`
- Formato: `solucion_OOP_{instancia}_{vendedores}vendedores_costo{costo}.png`
- Alta resolución (300 DPI) para calidad de publicación

## 🔧 Opciones de Configuración

### Carga de Instancias

```python
# Cargar diferentes tipos de instancias
instancia = PCTSPInstance("ruta/a/instancia.pctsp")

# Configurar restricciones de precedencia personalizadas
instancia.configurar_precedencias([(ciudad1, ciudad2), (ciudad3, ciudad4)])
```

### Parámetros del Algoritmo

```python
# Ajustar intensidad del algoritmo
algoritmo.ejecutar(iteraciones=1000)  # Ejecución estándar
algoritmo.ejecutar(iteraciones=500)   # Ejecución rápida
algoritmo.ejecutar(iteraciones=2000)  # Ejecución intensiva
```

## 📚 Formatos de Archivo

### Formato de Instancia PCTSP

```
NAME: nombre_instancia
SALESMEN: numero_de_vendedores
DIMENSION: numero_de_ciudades
EDGE_WEIGHT_SECTION
[matriz_de_distancias]
GCTSP_SET_SECTION
[restricciones_de_accesibilidad]
DEPOT_SECTION
ciudad_deposito
EOF
```

## 📖 Referencias

1. Zhang, J., Dai, B., Wang, Z., & Li, Y. (2022). Precedence-Constrained Colored Traveling Salesman Problem: An Augmented Variable Neighborhood Search Approach. _IEEE Transactions on Cybernetics_.

2. Dong, Y., & Cai, C. (2019). A novel genetic algorithm for large scale colored balanced traveling salesman problem. _Expert Systems with Applications_, 119, 113–124.

3. Hansen, P., & Mladenović, N. (2001). Variable neighborhood search: Principles and applications. _European Journal of Operational Research_, 130(3), 449-467.

---

**Nota**: Esta implementación se enfoca en la variante con restricciones de precedencia del TSP coloreado, proporcionando una base robusta para investigación y aplicaciones prácticas en logística, programación y optimización de rutas.
