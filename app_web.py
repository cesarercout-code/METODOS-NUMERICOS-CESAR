import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página web
st.set_page_config(page_title="Métodos Numéricos de Cesar", layout="wide")

st.title("📊 Métodos Numéricos de Denisse para todos")
st.write("Herramienta interactiva para matrices, sistemas de ecuaciones, iteraciones y búsqueda de raíces.")

# Creamos pestañas (Tabs) en la página web para organizar los métodos
tab1, tab2, tab3 = st.tabs(["🧮 Sistemas Directos", "🔄 Métodos Iterativos", "📈 Búsqueda de Raíces (Bisección)"])

# =====================================================================
# PESTAÑA 1: SISTEMAS DIRECTOS (Cramer, Gauss, LU)
# =====================================================================
with tab1:
    st.header("Resolución de Sistemas de Ecuaciones")
    n = st.number_input("Selecciona el tamaño de la matriz (n):", min_value=2, max_value=6, value=3, step=1)
    
    st.write("Ingresa los coeficientes de la Matriz A y los resultados del Vector B:")
    
    # Creamos contenedores dinámicos para las entradas en la web
    matriz_A = []
    vector_B = []
    
    for i in range(n):
        cols = st.columns(n + 2)
        fila = []
        for j in range(n):
            val = cols[j].number_input(f"A[{i+1},{j+1}]", value=0.0, key=f"a_{i}_{j}", label_visibility="collapsed")
            fila.append(val)
        matriz_A.append(fila)
        cols[n].text(" | ")
        b_val = cols[n+1].number_input(f"B[{i+1}]", value=0.0, key=f"b_{i}", label_visibility="collapsed")
        vector_B.append(b_val)
        
    A = np.array(matriz_A)
    B = np.array(vector_B)
    
    metodo_sel = st.selectbox("Elige el método:", ["Cramer", "Gauss", "Factorización LU"])
    
    if st.button("Resolver Sistema Directo"):
        st.subheader("Procedimiento y Resultados:")
        if metodo_sel == "Cramer":
            det_A = np.linalg.det(A)
            st.write(f"**Determinante Principal (Det_A):** {det_A:.4f}")
            if abs(det_A) < 1e-9:
                st.error("El determinante es 0. El sistema no tiene solución única.")
            else:
                for i in range(n):
                    A_mod = A.copy()
                    A_mod[:, i] = B
                    det_i = np.linalg.det(A_mod)
                    x_i = det_i / det_A
                    st.write(f"* x{i+1} = Det_x{i+1} / Det_A = {det_i:.4f} / {det_A:.4f} = **{x_i:.4f}**")
                    
        elif metodo_sel == "Gauss":
            Ab = np.column_stack((A, B))
            st.write("Matriz Aumentada inicial [A | B]:")
            st.write(np.round(Ab, 4))
            for i in range(n):
                for j in range(i+1, n):
                    factor = Ab[j, i] / Ab[i, i]
                    Ab[j] = Ab[j] - factor * Ab[i]
                    st.text(f"F{j+1} = F{j+1} - ({factor:.4f}) * F{i+1}")
            st.write("Matriz Triangulada final:")
            st.write(np.round(Ab, 4))
            
        elif metodo_sel == "Factorización LU":
            L = np.eye(n)
            U = A.copy()
            for i in range(n):
                for j in range(i+1, n):
                    factor = U[j, i] / U[i, i]
                    L[j, i] = factor
                    U[j] = U[j] - factor * U[i]
            st.write("**Matriz L (Inferior):**")
            st.write(np.round(L, 4))
            st.write("**Matriz U (Superior):**")
            st.write(np.round(U, 4))
            # Resolución con numpy solve para mostrar la respuesta limpia
            x = np.linalg.solve(A, B)
            st.success("Solución final del sistema:")
            for i in range(n):
                st.write(f"x{i+1} = {x[i]:.4f}")

# =====================================================================
# PESTAÑA 2: MÉTODOS ITERATIVOS (Jacobi y Gauss-Seidel)
# =====================================================================
with tab2:
    st.header("Métodos Iterativos (Sistemas Lineales)")
    n_iter = st.number_input("Tamaño de la matriz (n) para iterar:", min_value=2, max_value=5, value=3, key="n_it")
    
    col_tol, col_max = st.columns(2)
    tol = col_tol.number_input("Tolerancia:", value=0.0001, format="%.5f")
    max_iter = col_max.number_input("Máximo de Iteraciones:", value=30, step=1)
    
    st.write("Ingresa los datos para Jacobi / Gauss-Seidel:")
    mat_ai = []
    vec_bi = []
    for i in range(n_iter):
        cols = st.columns(n_iter + 2)
        fila = []
        for j in range(n_iter):
            val = cols[j].number_input(f"AI[{i+1},{j+1}]", value=0.0, key=f"ai_{i}_{j}", label_visibility="collapsed")
            fila.append(val)
        mat_ai.append(fila)
        cols[n_iter].text(" | ")
        b_val = cols[n_iter+1].number_input(f"BI[{i+1}]", value=0.0, key=f"bi_{i}", label_visibility="collapsed")
        vec_bi.append(b_val)
        
    A_i = np.array(mat_ai)
    B_i = np.array(vec_bi)
    
    op_iter = st.radio("Selecciona el algoritmo:", ["Jacobi", "Gauss-Seidel"])
    
    if st.button("Ejecutar Iteraciones"):
        st.subheader("Desarrollo paso a paso:")
        x = np.zeros(n_iter)
        x_old = np.zeros(n_iter)
        
        for k in range(int(max_iter)):
            st.markdown(f"**--- Iteración {k+1} ---**")
            if op_iter == "Jacobi":
                x_new = np.zeros(n_iter)
                for i in range(n_iter):
                    suma = sum(A_i[i, j] * x_old[j] for j in range(n_iter) if j != i)
                    x_new[i] = (B_i[i] - suma) / A_i[i, i]
                    st.text(f"  x{i+1} = ({B_i[i]} - [{suma:.4f}]) / {A_i[i,i]} = {x_new[i]:.4f}")
                error = np.max(np.abs(x_new - x_old))
                st.text(f"  Error calculado = {error:.6f}")
                if error < tol:
                    st.success(f"¡Convergencia lograda en la iteración {k+1}!")
                    break
                x_old = x_new.copy()
            else: # Gauss-Seidel
                for i in range(n_iter):
                    suma = sum(A_i[i, j] * x[j] for j in range(n_iter) if j != i)
                    x_new_val = (B_i[i] - suma) / A_i[i, i]
                    st.text(f"  x{i+1} = ({B_i[i]} - [{suma:.4f}]) / {A_i[i,i]} = {x_new_val:.4f}")
                    x[i] = x_new_val
                error = np.max(np.abs(x - x_old))
                st.text(f"  Error máximo = {error:.6f}")
                if error < tol:
                    st.success(f"¡Convergencia lograda en la iteración {k+1}!")
                    break
                x_old = x.copy()

# =====================================================================
# PESTAÑA 3: BISECCIÓN CON GRÁFICOS WEB
# =====================================================================
with tab3:
    st.header("Método de Bisección (Búsqueda de Raíces)")
    formula = st.text_input("Ingresa la función f(x):", value="x**3 - x - 2")
    
    col1, col2, col3, col4 = st.columns(4)
    a_val = col1.number_input("Intervalo a:", value=1.0)
    b_val = col2.number_input("Intervalo b:", value=2.0)
    tol_bis = col3.number_input("Tolerancia Bisección:", value=0.001, format="%.4f")
    max_bis = col4.number_input("Máx Iteraciones:", value=20, step=1)
    
    def evaluar(x_num, form):
        entorno = {"x": x_num, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "sqrt": np.sqrt}
        return eval(form, {"__builtins__": {}}, entorno)

    if st.button("Calcular Bisección y Graficar"):
        try:
            a, b = a_val, b_val
            fa = evaluar(a, formula)
            fb = evaluar(b, formula)
            
            if fa * fb > 0:
                st.error("f(a) y f(b) tienen el mismo signo. Elige otro intervalo.")
            else:
                st.subheader("Tabla de Iteraciones:")
                c_final = None
                for k in range(int(max_bis)):
                    c = (a + b) / 2
                    fc = evaluar(c, formula)
                    err = abs(b - a) / 2
                    st.text(f"Iter {k+1} | a={a:.4f}, b={b:.4f} | c={c:.4f} | f(c)={fc:.4f} | Error={err:.4f}")
                    
                    if err < tol_bis or abs(fc) < 1e-9:
                        c_final = c
                        st.success(f"Raíz encontrada en x = {c:.6f}")
                        break
                    if fa * fc < 0:
                        b = c
                    else:
                        a = c
                        fa = fc
                
                # Generar Gráfica integrada directamente en la web
                st.subheader("Visualización Gráfica:")
                fig, ax = plt.subplots(figsize=(8, 4))
                x_pts = np.linspace(a_val - 1, b_val + 1, 300)
                y_pts = [evaluar(x_pt, formula) for x_pt in x_pts]
                
                ax.plot(x_pts, y_pts, label=f"f(x) = {formula}", color="blue")
                ax.axhline(0, color="black", linewidth=1)
                ax.axvspan(a_val, b_val, color="yellow", alpha=0.3, label="Intervalo inicial")
                if c_final is not None:
                    ax.scatter([c_final], [evaluar(c_final, formula)], color="green", s=100, zorder=5, label=f"Raíz: {c_final:.4f}")
                
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.legend()
                ax.grid(True)
                
                # Streamlit dibuja la gráfica automáticamente en la página
                st.pyplot(fig)
                
        except Exception as ex:
            st.error(f"Revisa tu fórmula o valores. Detalle: {ex}")