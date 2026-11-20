import json
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import simpy
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Gemelo digital de almacén ABC", page_icon="🏭", layout="wide")
st.title("🏭 Almacén 3D: ubicación aleatoria vs clasificación ABC")
st.caption("SimPy + Streamlit + Three.js · mismos pedidos y recursos, distinta política de ubicación")

with st.sidebar:
    st.header("Parámetros de simulación")
    n_pedidos = st.slider("Pedidos (una línea por pedido)", 50, 800, 250, 50)
    n_operarios = st.slider("Operarios de picking", 1, 8, 2)
    velocidad = st.slider("Velocidad de desplazamiento (m/min)", 20, 100, 55, 5)
    media_arribos = st.number_input("Media entre arribos (min) · exponencial", min_value=0.1, max_value=120.0, value=2.0, step=0.5, format="%.1f")
    st.caption(f"Tasa media equivalente: {60 / media_arribos:.1f} pedidos/hora")
    semilla = st.number_input("Semilla aleatoria", min_value=0, max_value=999999, value=42)
    st.divider()
    st.write("**Animación 3D**")
    max_anim = st.slider("Pedidos a animar", 5, 50, 20, 5)
    st.caption("La animación es una reproducción acelerada de eventos SimPy; los KPIs usan todos los pedidos.")
    st.divider()
    st.write("**Plano de fondo del almacén (opcional)**")
    plano = st.file_uploader("Subir plano PNG/JPG", type=["png", "jpg", "jpeg"])
    mostrar_plano = st.checkbox("Mostrar imagen sobre el piso", value=True)
    ancho_plano = st.number_input("Ancho del plano (m)", min_value=1.0, value=70.0, step=1.0)
    largo_plano = st.number_input("Fondo del plano (m)", min_value=1.0, value=42.0, step=1.0)
    plano_x = st.number_input("Centro X del plano (m)", value=29.0, step=1.0)
    plano_z = st.number_input("Centro Z del plano (m)", value=13.0, step=1.0)
    st.caption("El plano es una textura visual; no modifica distancias ni obstáculos.")

# Ubicaciones: 3 bloques físicos, cada uno 3 columnas x 4 filas.
# La posición es el punto de acceso del rack, con despacho en (0,0).
def crear_ubicaciones():
    rows = []
    for zona, x0 in [("A", 8), ("B", 28), ("C", 48)]:
        for i in range(12):
            x = x0 + (i % 3) * 4
            y = 5 + (i // 3) * 5
            rows.append({
                "ubicacion": f"{zona}-{i + 1:02d}", "zona_fisica": zona,
                "x": float(x), "y": float(y), "distancia_ida_vuelta": float(2 * (x + y))
            })
    return pd.DataFrame(rows)


def crear_productos():
    n = 30
    pesos = 1 / np.arange(1, n + 1, dtype=float) ** 1.25
    pesos /= pesos.sum()
    df = pd.DataFrame({"sku": [f"SKU-{i + 1:03d}" for i in range(n)], "probabilidad": pesos})
    acumulado = df.probabilidad.cumsum()
    anterior = acumulado - df.probabilidad
    df["clase_abc"] = np.where(anterior < 0.80, "A", np.where(anterior < 0.95, "B", "C"))
    return df


def asignar(ubicaciones, productos, politica, seed):
    rng = np.random.default_rng(seed + 1000)
    if politica == "Aleatoria":
        elegidas = ubicaciones.iloc[rng.permutation(len(ubicaciones))[:len(productos)]].reset_index(drop=True)
    else:
        # Distancias más cortas primero; en este layout están en A, después B, después C.
        elegidas = ubicaciones.sort_values(["distancia_ida_vuelta", "ubicacion"]).head(len(productos)).reset_index(drop=True)
    return pd.concat([productos.reset_index(drop=True), elegidas], axis=1)


def crear_pedidos(productos, cantidad, media_min, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "pedido": np.arange(1, cantidad + 1),
        "llegada": np.cumsum(rng.exponential(media_min, size=cantidad)),
        "sku": rng.choice(productos.sku.to_numpy(), size=cantidad, p=productos.probabilidad.to_numpy()),
        "manipulacion": rng.triangular(0.6, 1.1, 2.0, size=cantidad)
    })


def simular(pedidos, asignaciones, cantidad_operarios, velocidad_m_min):
    env = simpy.Environment()
    recurso = simpy.Resource(env, capacity=cantidad_operarios)
    mapa = asignaciones.set_index("sku").to_dict("index")
    resultados = []
    eventos = []
    # Asignación estable de operarios al terminar solicitudes: el primero disponible.
    disponibles = list(range(cantidad_operarios))

    def procesar(p):
        yield env.timeout(max(0, float(p.llegada) - env.now))
        with recurso.request() as solicitud:
            yield solicitud
            op = disponibles.pop(0)
            inicio = env.now
            loc = mapa[p.sku]
            distancia = float(loc["distancia_ida_vuelta"])
            ida = distancia / (2 * velocidad_m_min)
            manip = float(p.manipulacion)
            # Ida, manipulación y vuelta son eventos independientes de SimPy.
            eventos.append({"pedido": int(p.pedido), "operario": op, "tipo": "salida", "t": float(env.now), "x": 0, "y": 0})
            yield env.timeout(ida)
            eventos.append({"pedido": int(p.pedido), "operario": op, "tipo": "llegada_rack", "t": float(env.now), "x": float(loc["x"]), "y": float(loc["y"])})
            yield env.timeout(manip)
            eventos.append({"pedido": int(p.pedido), "operario": op, "tipo": "regreso", "t": float(env.now), "x": float(loc["x"]), "y": float(loc["y"])})
            yield env.timeout(ida)
            eventos.append({"pedido": int(p.pedido), "operario": op, "tipo": "finalizado", "t": float(env.now), "x": 0, "y": 0})
            resultados.append({
                "pedido": int(p.pedido), "sku": p.sku, "clase_abc": loc["clase_abc"],
                "ubicacion": loc["ubicacion"], "zona_fisica": loc["zona_fisica"],
                "llegada": float(p.llegada), "inicio": float(inicio), "fin": float(env.now),
                "espera": float(inicio - p.llegada), "servicio": float(env.now - inicio),
                "ciclo": float(env.now - p.llegada), "distancia": distancia,
                "operario": op, "x": float(loc["x"]), "y": float(loc["y"])
            })
            disponibles.append(op)
            disponibles.sort()

    for p in pedidos.itertuples(index=False):
        env.process(procesar(p))
    env.run()
    df = pd.DataFrame(resultados).sort_values("pedido").reset_index(drop=True)
    horizonte = float(df.fin.max())
    kpi = {
        "Distancia total (m)": float(df.distancia.sum()),
        "Tiempo de ciclo promedio (min)": float(df.ciclo.mean()),
        "Espera promedio (min)": float(df.espera.mean()),
        "Productividad observada (pedidos/h)": float(len(df) * 60 / horizonte),
        "Utilización de operarios (%)": float(df.servicio.sum() / (cantidad_operarios * horizonte) * 100),
        "Horizonte para completar (min)": horizonte,
    }
    return df, kpi, eventos

ubicaciones = crear_ubicaciones()
productos = crear_productos()
pedidos = crear_pedidos(productos, n_pedidos, media_arribos, int(semilla))
asig_random = asignar(ubicaciones, productos, "Aleatoria", int(semilla))
asig_abc = asignar(ubicaciones, productos, "ABC", int(semilla))
res_random, kpi_random, eventos_random = simular(pedidos, asig_random, n_operarios, velocidad)
res_abc, kpi_abc, eventos_abc = simular(pedidos, asig_abc, n_operarios, velocidad)

comparacion = pd.DataFrame({
    "Indicador": list(kpi_random),
    "Sin ABC": list(kpi_random.values()),
    "Con ABC": list(kpi_abc.values()),
})
comparacion["Variación (%)"] = np.where(
    comparacion["Sin ABC"] != 0,
    (comparacion["Con ABC"] / comparacion["Sin ABC"] - 1) * 100, np.nan
)

reduccion_dist = (1 - kpi_abc["Distancia total (m)"] / kpi_random["Distancia total (m)"]) * 100
reduccion_ciclo = (1 - kpi_abc["Tiempo de ciclo promedio (min)"] / kpi_random["Tiempo de ciclo promedio (min)"]) * 100
c1, c2, c3 = st.columns(3)
c1.metric("Ahorro de distancia con ABC", f"{reduccion_dist:.1f}%")
c2.metric("Reducción del ciclo con ABC", f"{reduccion_ciclo:.1f}%")
c3.metric("Pedidos simulados por escenario", str(n_pedidos))

st.subheader("Visualización tridimensional")
scenario = st.radio("Escenario a visualizar", ["Sin ABC", "Con ABC"], horizontal=True, key="scenario_3d")
sel = asig_random if scenario == "Sin ABC" else asig_abc
res = res_random if scenario == "Sin ABC" else res_abc

# El JSON es embebido en la página; no hay servidor WebSocket.
# Se animan las primeras N órdenes POR ID; cada orden se reproduce a partir de sus eventos SimPy.
ids = set(range(1, max_anim + 1))
rows_anim = res[res.pedido.isin(ids)].sort_values("inicio")
plano_data = ""
if plano is not None and mostrar_plano:
    tipo = "image/png" if plano.type == "image/png" else "image/jpeg"
    plano_data = f"data:{tipo};base64,{base64.b64encode(plano.getvalue()).decode('ascii')}"

trayectorias = [
    {"pedido": int(r.pedido), "sku": str(r.sku), "clase": str(r.clase_abc),
     "operario": int(r.operario), "inicio": float(r.inicio), "fin": float(r.fin),
     "llegadaRack": float(r.inicio + r.distancia / (2 * velocidad)),
     "salidaRack": float(r.fin - r.distancia / (2 * velocidad)),
     "x": float(r.x), "z": float(r.y)}
    for r in rows_anim.itertuples(index=False)
]
layout_js = [{"id": str(r.ubicacion), "zona": str(r.zona_fisica), "x": float(r.x), "z": float(r.y),
              "sku": str(r.sku), "clase": str(r.clase_abc)} for r in sel.itertuples(index=False)]

html = r'''
<div style="font-family:system-ui;color:#172033">
 <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
  <button id="play">▶ Reproducir</button><button id="reset">↺ Reiniciar</button>
  <label>Velocidad <select id="speed"><option value="1">1×</option><option value="5" selected>5×</option><option value="15">15×</option><option value="40">40×</option></select></label>
  <span id="clock">t = 0.0 min</span><span id="status">Pausa</span>
 </div>
 <div id="scene" style="width:100%;height:530px;border-radius:12px;overflow:hidden;background:#e8edf5"></div>
 <div style="font-size:12px;margin-top:5px">Ratón: arrastrar para girar · rueda para zoom · clic derecho para desplazar. Rojo = SKU A, naranja = SKU B, verde = SKU C.</div>
</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const layout = __LAYOUT__;
const tasks = __TASKS__;
const totalWorkers = __WORKERS__;
const floorImage = __FLOOR_IMAGE__;
const floorConfig = __FLOOR_CONFIG__;
const root = document.getElementById('scene');
const scene = new THREE.Scene(); scene.background = new THREE.Color(0xe8edf5);
const camera = new THREE.PerspectiveCamera(48, root.clientWidth/530, .1, 500);
camera.position.set(42,58,90);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(root.clientWidth,530);renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
root.appendChild(renderer.domElement);
const controls = new OrbitControls(camera,renderer.domElement);
controls.target.set(28,0,13);controls.enableDamping=true;controls.update();
scene.add(new THREE.HemisphereLight(0xffffff,0x8899bb,2.1));
const light = new THREE.DirectionalLight(0xffffff,1.8);light.position.set(20,55,15);scene.add(light);
function block(x,y,z,w,h,d,color){
 const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),new THREE.MeshStandardMaterial({color,roughness:.78}));
 m.position.set(x,y,z);scene.add(m);return m;
}
block(29,-.3,13,70,.4,42,0xcbd5e1);
if(floorImage){
 const texture = new THREE.TextureLoader().load(floorImage);
 texture.colorSpace = THREE.SRGBColorSpace;
 const mat = new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,transparent:true,opacity:0.90,depthWrite:false});
 const floorPlan = new THREE.Mesh(new THREE.PlaneGeometry(floorConfig.width,floorConfig.depth),mat);
 floorPlan.rotation.x = -Math.PI/2;
 floorPlan.position.set(floorConfig.x,-0.055,floorConfig.z);
 scene.add(floorPlan);
}
const grid = new THREE.GridHelper(72,36,0xa1adbd,0xd1d9e3);grid.position.set(29,-.08,13);scene.add(grid);
const zoneColors={A:0x93c5fd,B:0xc4b5fd,C:0xfcd34d};
for(const [name,x] of [['A',12],['B',32],['C',52]]){
 const plate=block(x,.02,13,18,.05,33,zoneColors[name]);plate.material.transparent=true;plate.material.opacity=.20;
}
block(0,.16,0,5,.32,5,0x0f766e);
for(const p of layout){
 const c={A:0xdc2626,B:0xf59e0b,C:0x16a34a}[p.clase];
 block(p.x,1.4,p.z,2.8,2.8,2.7,0x64748b);
 block(p.x,3.0,p.z,2.2,.7,2.1,c);
 block(p.x,0.2,p.z,3.0,.4,3.0,0x475569);
}
// Operarios visibles solo durante el recorrido de un pedido.
const nOps = totalWorkers;
const workers = [];
for(let i=0;i<nOps;i++){
 const g=new THREE.Group();
 const body=new THREE.Mesh(new THREE.CylinderGeometry(.45,.5,1.3,10),new THREE.MeshStandardMaterial({color:0x2563eb}));body.position.y=1.0;g.add(body);
 const head=new THREE.Mesh(new THREE.SphereGeometry(.35,12,12),new THREE.MeshStandardMaterial({color:0xfbbf24}));head.position.y=1.95;g.add(head);
 // Cada operario permanece visible incluso cuando no tiene pedidos.
 g.position.set(-2 + (i%4)*1.4,0,-4-Math.floor(i/4)*1.4);
 scene.add(g);workers.push(g);
}
let t=0,playing=false,last=performance.now();
const end=Math.max(1,...tasks.map(d=>d.fin));
function position(task,now){
 let f=0;
 if(now<task.llegadaRack) f=(now-task.inicio)/Math.max(.001,task.llegadaRack-task.inicio);
 else if(now<task.salidaRack) f=1;
 else f=1-(now-task.salidaRack)/Math.max(.001,task.fin-task.salidaRack);
 f=Math.max(0,Math.min(1,f));
 // Movimiento ortogonal: primero eje X, después eje Z.
 const d1=task.x/(task.x+task.z);
 const x=f<=d1 ? f/d1*task.x : task.x;
 const z=f<=d1 ? 0 : (f-d1)/(1-d1)*task.z;
 return [x,z];
}
function renderWorkers(){
 workers.forEach((w,i)=>{
  w.visible=true;
  w.position.set(-2+(i%4)*1.4,0,-4-Math.floor(i/4)*1.4);
 });
 for(const task of tasks){
  if(t>=task.inicio && t<task.fin){
   const w=workers[task.operario];if(!w)continue;
   const [x,z]=position(task,t);w.position.set(x,0,z);w.visible=true;
  }
 }
 document.getElementById('clock').textContent=`t = ${t.toFixed(1)} min / ${end.toFixed(1)} min`;
 document.getElementById('status').textContent=playing?'En reproducción':'Pausa';
}
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'⏸ Pausar':'▶ Reproducir'};
document.getElementById('reset').onclick=()=>{t=0;playing=false;document.getElementById('play').textContent='▶ Reproducir';renderWorkers()};
function loop(now){
 const dt=Math.min(.1,(now-last)/1000);last=now;
 if(playing){t+=dt*Number(document.getElementById('speed').value);if(t>=end){t=end;playing=false;document.getElementById('play').textContent='▶ Reproducir'}}
 renderWorkers();controls.update();renderer.render(scene,camera);requestAnimationFrame(loop);
}
window.addEventListener('resize',()=>{camera.aspect=root.clientWidth/530;camera.updateProjectionMatrix();renderer.setSize(root.clientWidth,530)});
renderWorkers();requestAnimationFrame(loop);
</script>
'''
html = (html.replace("__LAYOUT__", json.dumps(layout_js, ensure_ascii=False))
        .replace("__TASKS__", json.dumps(trayectorias, ensure_ascii=False))
        .replace("__WORKERS__", str(n_operarios))
        .replace("__FLOOR_IMAGE__", json.dumps(plano_data))
        .replace("__FLOOR_CONFIG__", json.dumps({"width": ancho_plano, "depth": largo_plano, "x": plano_x, "z": plano_z})))
components.html(html, height=610, scrolling=False)
st.caption(f"Se muestran {n_operarios} operarios (en movimiento o esperando). La animación reproduce hasta {max_anim} pedidos; los KPIs incluyen los {n_pedidos} pedidos. El plano de fondo es solo visual.")

st.subheader("Resultados de simulación")
st.dataframe(comparacion.style.format({"Sin ABC":"{:,.2f}", "Con ABC":"{:,.2f}", "Variación (%)":"{:+.1f}%"}), use_container_width=True, hide_index=True)
plot = comparacion[comparacion.Indicador.isin(["Distancia total (m)", "Tiempo de ciclo promedio (min)", "Espera promedio (min)", "Horizonte para completar (min)"])].melt(id_vars="Indicador", var_name="Escenario", value_name="Valor")
fig = px.bar(plot, x="Escenario", y="Valor", color="Escenario", facet_col="Indicador", facet_col_wrap=2)
fig.update_yaxes(matches=None, showticklabels=True)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Mapa 2D de las posiciones")
fig2 = px.scatter(sel, x="x", y="y", color="clase_abc", hover_data=["sku", "ubicacion", "zona_fisica", "distancia_ida_vuelta"],
                  color_discrete_map={"A":"#dc2626", "B":"#f59e0b", "C":"#16a34a"},
                  labels={"clase_abc":"Rotación SKU"})
fig2.update_traces(marker_size=13)
fig2.update_yaxes(scaleanchor="x", scaleratio=1)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Ver datos de pedidos y asignaciones"):
    st.write("**Catálogo ABC**")
    st.dataframe(productos, use_container_width=True, hide_index=True)
    st.write("**Ubicaciones del escenario seleccionado**")
    st.dataframe(sel, use_container_width=True, hide_index=True)
    st.write("**Pedidos procesados**")
    st.dataframe(res, use_container_width=True, hide_index=True)

buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    productos.to_excel(writer, sheet_name="Productos", index=False)
    pedidos.to_excel(writer, sheet_name="Pedidos_comunes", index=False)
    asig_random.to_excel(writer, sheet_name="Ubicaciones_sin_ABC", index=False)
    asig_abc.to_excel(writer, sheet_name="Ubicaciones_con_ABC", index=False)
    res_random.to_excel(writer, sheet_name="Resultados_sin_ABC", index=False)
    res_abc.to_excel(writer, sheet_name="Resultados_con_ABC", index=False)
    comparacion.to_excel(writer, sheet_name="Comparacion", index=False)
st.download_button("📥 Descargar resultados completos en Excel", buffer.getvalue(), file_name="simulacion_almacen_abc.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.info("Modelo didáctico: un SKU por pedido, una ubicación por SKU y recorrido Manhattan. Para estudios operativos reales agregue red de pasillos, capacidades, reposición, pedidos multilínea y réplicas estadísticas.")
