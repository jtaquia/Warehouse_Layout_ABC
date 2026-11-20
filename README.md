# Almacén 3D ABC + SimPy (versión ampliada)

## Instalar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Mejoras

- Media de tiempo entre arribos en minutos, con distribución exponencial (`numpy.random.Generator.exponential`). Ambos escenarios comparten los mismos pedidos.
- Imagen PNG/JPG opcional sobre el piso 3D, con ancho, fondo y centro ajustables en metros. Es una textura, no un mapa de obstáculos.
- Se crean exactamente tantos operarios 3D como los indicados en la barra lateral. Los operarios sin tarea esperan en la zona de inicio.
- SimPy registra el operario asignado a cada pedido; la animación reproduce los primeros N pedidos elegidos, mientras los KPIs consideran todos.

## Nota

El recorrido es Manhattan simplificado. La asignación ABC prioriza ubicaciones cercanas pero no obliga exclusividad por zona. La imagen se carga en el navegador y no se envía a servicios externos por el código de la app; Three.js se obtiene de CDN.
