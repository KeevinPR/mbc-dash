# 📊 Network Visualization Feature - MBC Dashboard

## 🎯 Overview
Se ha agregado visualización interactiva de redes bayesianas aprendidas al dashboard MBC usando **Dash Cytoscape**.

## ✨ Características

### Visualización Interactiva
- **Grafo interactivo** con nodos y aristas
- **Layout automático** tipo force-directed (CoSE)
- **Colores diferenciados**:
  - 🔴 **Nodos rojos (círculos)**: Variables de clase
  - 🔷 **Nodos azules (rectángulos)**: Variables features
  - 🟠 **Aristas naranjas gruesas**: Conexiones clase → feature
  - 🔴 **Aristas rojas**: Conexiones entre clases
  - 🔷 **Aristas azules**: Conexiones entre features

### Interactividad
- **Zoom**: Scroll del ratón
- **Pan**: Arrastrar el fondo
- **Mover nodos**: Arrastrar nodos individuales
- **Hover**: Información sobre nodos y aristas

## 🔧 Implementación Técnica

### Archivos Modificados
- ✅ `dash_mbc.py`: Dashboard principal con visualización

### Componentes Agregados

#### 1. **Import de Cytoscape**
```python
import dash_cytoscape as cyto
```

#### 2. **Estilos de Visualización**
Define los estilos visuales para:
- Nodos (clase vs feature)
- Aristas (clase, feature, bridge)
- Colores, formas, tamaños

#### 3. **Store de Datos de Red**
```python
dcc.Store(id='mbc-network-store')
```
Almacena la estructura de la red:
- Arcos (from → to)
- Lista de clases
- Lista de features

#### 4. **Callback de Visualización**
```python
@app.callback(
    Output('mbc-network-container', 'children'),
    Input('mbc-network-store', 'data')
)
def visualize_network(network_data):
    # Construye elementos de Cytoscape
    # Retorna componente visual
```

## 📦 Dependencias
```txt
dash-cytoscape==0.3.0  # Ya incluido en requirements.txt
```

## 🚀 Uso

### 1. Cargar Dataset
- Subir archivo CSV con datos

### 2. Seleccionar Variables
- **Classes**: Variables a predecir
- **Features**: Variables predictoras

### 3. Configurar Opciones
- Approach (Filter/Wrapper)
- Discretización
- Train/Val split

### 4. Run MBC
- Click en "Run MBC"
- **Resultado**: Se muestra automáticamente:
  1. 🔗 **Network Structure** (visualización interactiva)
  2. 📊 **Performance metrics** (accuracy)

## 🎨 Personalización del Layout

### Parámetros de CoSE Layout
```python
layout={
    'name': 'cose',
    'animate': True,
    'nodeRepulsion': 8000,      # Separación entre nodos
    'idealEdgeLength': 100,     # Longitud ideal de aristas
    'edgeElasticity': 100,      # Elasticidad de aristas
    'gravity': 80,              # Fuerza de gravedad central
    'numIter': 1000            # Iteraciones de optimización
}
```

### Otros Layouts Disponibles
- `'circle'`: Circular
- `'grid'`: Rejilla
- `'breadthfirst'`: Jerárquico
- `'concentric'`: Concéntrico

## 🔍 Ejemplo de Datos de Red

```json
{
  "arcs": [
    {"from": "Age", "to": "Income"},
    {"from": "Income", "to": "Class1"},
    {"from": "Education", "to": "Class2"}
  ],
  "classes": ["Class1", "Class2"],
  "features": ["Age", "Income", "Education"]
}
```

## 🐛 Troubleshooting

### Red No Se Muestra
1. ✅ Verificar que hay arcos aprendidos
2. ✅ Revisar consola del navegador (F12)
3. ✅ Verificar que `dash-cytoscape` está instalado

### Layout Desordenado
- Refrescar la página
- Ajustar parámetros de `nodeRepulsion` o `idealEdgeLength`

### Nodos Superpuestos
- Aumentar `nodeRepulsion`
- Cambiar a layout `'breadthfirst'` para grafos jerárquicos

## 📚 Referencias

- **Dash Cytoscape**: https://dash.plotly.com/cytoscape
- **Cytoscape.js**: https://js.cytoscape.org/
- **CoSE Layout**: https://github.com/cytoscape/cytoscape.js-cose-bilkent

## 🎯 Próximas Mejoras Posibles

- [ ] Exportar grafo a imagen (PNG/SVG)
- [ ] Mostrar probabilidades en aristas
- [ ] Filtrar tipos de aristas (toggle)
- [ ] Vista de tabla de CPTs al hacer click en nodos
- [ ] Comparación de múltiples modelos
- [ ] Layout jerárquico automático según subgrafos

---

**Autor**: Implementado para MBC Dashboard  
**Fecha**: 2025  
**Versión**: 1.0

