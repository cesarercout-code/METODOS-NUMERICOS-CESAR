import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página web
st.set_page_config(page_title="Métodos Numéricos de Cesar", layout="wide")

st.title("📊 Métodos Numéricos de Cesar Ramirez")
st.write("Herramienta interactiva para matrices, sistemas de ecuaciones, iteraciones y búsqueda de raíces.")

# Creamos pestañas (Tabs) en la página web para organizar los métodos
tab1, tab2, tab3 = st.tabs(["🧮 Sistemas Directos", "🔄 Métodos Iterativos", "📈 Búsqueda de Raíces (Bisección)"])

# =====================================================================
# PESTAÑA 1: SISTEMAS DIRECTOS (Cramer, Gauss, LU)
# =====================================================================
def matrix_to_latex(mat):
    """Función auxiliar para convertir matrices de numpy a formato LaTeX"""
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    lines = []
    for row in mat:
        lines.append(" & ".join([f"{val:.3f}" for val in row]))
    return "\\begin{bmatrix} " + " \\\\ ".join(lines) + " \\end{bmatrix}"

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
        if metodo_sel == "Cramer":
            st.subheader("👨‍🏫 Fundamento Matemático: Regla de Cramer")
            st.info("La Regla de Cramer establece que la solución de un sistema se encuentra dividiendo el determinante de matrices modificadas (donde se reemplaza la columna $i$ por el vector de resultados $B$) entre el determinante de la matriz original $A$.")
            st.latex(r"x_i = \frac{\det(A_i)}{\det(A)}")
            
            st.subheader("Desarrollo paso a paso:")
            det_A = np.linalg.det(A)
            st.markdown("**1. Matriz original y Determinante Principal ($\det(A)$):**")
            st.latex(f"A = {matrix_to_latex(A)} \\implies \det(A) = {det_A:.4f}")
            
            if abs(det_A) < 1e-9:
                st.error("El determinante es 0. El sistema no tiene solución única.")
            else:
                st.markdown("**2. Cálculo de incógnitas sustituyendo el vector B:**")
                for i in range(n):
                    A_mod = A.copy()
                    A_mod[:, i] = B
                    det_i = np.linalg.det(A_mod)
                    x_i = det_i / det_A
                    
                    st.markdown(f"$\circ \quad$ **Cálculo de $x_{i+1}$**: Sustituyendo la columna {i+1} por el vector de resultados $B$:")
                    st.latex(f"A_{{{i+1}}} = {matrix_to_latex(A_mod)} \\implies \det(A_{{{i+1}}}) = {det_i:.4f}")
                    st.latex(f"x_{{{i+1}}} = \\frac{{\det(A_{{{i+1}}})}}{{\det(A)}} = \\frac{{{det_i:.4f}}}{{{det_A:.4f}}} = \mathbf{{{x_i:.4f}}}")
                    st.divider()
                    
        elif metodo_sel == "Gauss":
            st.subheader("Procedimiento y Resultados:")
            Ab = np.column_stack((A, B))
            st.write("Matriz Aumentada inicial $[A | B]$:")
            st.latex(matrix_to_latex(Ab))
            for i in range(n):
                for j in range(i+1, n):
                    factor = Ab[j, i] / Ab[i, i]
                    Ab[j] = Ab[j] - factor * Ab[i]
                    st.text(f"F{j+1} = F{j+1} - ({factor:.4f}) * F{i+1}")
            st.write("Matriz Triangulada final:")
            st.latex(matrix_to_latex(Ab))
            
        elif metodo_sel == "Factorización LU":
            st.subheader("👨‍🏫 Fundamento Matemático: Factorización LU")
            st.info("Descompone la matriz original $A$ en el producto de una matriz triangular inferior $L$ (con unos en la diagonal) y una triangular superior $U$. Luego, resuelve el sistema en dos etapas: primero $Ly = B$ y después $Ux = y$.")
            st.latex(r"A = L \cdot U \quad \implies \quad Ly = B, \quad Ux = y")
            
            st.subheader("Desarrollo paso a paso:")
            L = np.eye(n)
            U = A.copy()
            
            st.markdown("**1. Proceso de triangulación (Encontrando $L$ y $U$):**")
            for i in range(n):
                for j in range(i+1, n):
                    if U[i, i] == 0:
                        st.error("División por cero detectada. Se requiere pivoteo (no implementado en LU básico).")
                        st.stop()
                    factor = U[j, i] / U[i, i]
                    
                    st.markdown(f"$\circ \quad$ Multiplicador para fila {j+1} usando pivote en fila {i+1}:")
                    st.latex(f"m_{{{j+1},{i+1}}} = \\frac{{{U[j,i]:.3f}}}{{{U[i,i]:.3f}}} = {factor:.4f} \quad \implies \quad F_{j+1} \\leftarrow F_{j+1} - ({factor:.4f})F_{i+1}")
                    
                    L[j, i] = factor
                    U[j] = U[j] - factor * U[i]
                    
            st.markdown("**Matrices resultantes de la factorización:**")
            col_l, col_u = st.columns(2)
            with col_l:
                st.markdown("**Matriz $L$ (Inferior):**")
                st.latex(matrix_to_latex(L))
            with col_u:
                st.markdown("**Matriz $U$ (Superior):**")
                st.latex(matrix_to_latex(U))
                
            st.markdown("**2. Sustitución hacia adelante ($Ly = B$):**")
            y = np.linalg.solve(L, B)
            st.latex(f"y = {matrix_to_latex(y)}")
            
            st.markdown("**3. Sustitución hacia atrás ($Ux = y$):**")
            x = np.linalg.solve(U, y)
            st.latex(f"x = {matrix_to_latex(x)}")
            
            st.success("Solución final del sistema:")
            res_str = ", \ ".join([f"x_{i+1} = {x[i]:.4f}" for i in range(n)])
            st.markdown(f"${res_str}$")
            
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
        st.subheader(f"👨‍🏫 Fundamento Matemático: {op_iter}")
        if op_iter == "Jacobi":
            st.info("En el método de Jacobi, los valores de la nueva iteración se calculan usando únicamente los valores de la iteración anterior.")
            st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)")
        else:
            st.info("En el método de Gauss-Seidel, los valores recién calculados se utilizan inmediatamente en la misma iteración para acelerar la convergencia.")
            st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} a_{ij} x_j^{(k)} \right)")
            
        st.subheader("Desarrollo paso a paso:")
        x = np.zeros(n_iter)
        x_old = np.zeros(n_iter)
        
        for k in range(int(max_iter)):
            st.markdown(f"### Iteración {k+1} ($k = {k}$)")
            
            if op_iter == "Jacobi":
                st.write("Se sustituyen los valores de la iteración anterior. Cada resultado se calcula con los valores previos:")
                x_new = np.zeros(n_iter)
                for i in range(n_iter):
                    usando_list = [f"x_{j+1} = {x_old[j]:.3f}" for j in range(n_iter) if j != i]
                    usando_str = " \text{ y } ".join(usando_list)
                    st.markdown(f"$\circ \quad$ **Cálculo de $x_{i+1}^{({k+1})}$**: Usando ${usando_str}$")
                    
                    suma = sum(A_i[i, j] * x_old[j] for j in range(n_iter) if j != i)
                    x_new[i] = (B_i[i] - suma) / A_i[i, i]
                    
                    terms = [f"{B_i[i]:.3f}"]
                    for j in range(n_iter):
                        if j != i:
                            val_sub = A_i[i, j] * x_old[j]
                            if val_sub < 0: terms.append(f"+ {abs(val_sub):.3f}")
                            else: terms.append(f"- {abs(val_sub):.3f}")
                    num_latex = " ".join(terms)
                    num_val = B_i[i] - suma
                    
                    st.latex(f"x_{{{i+1}}}^{({k+1})} = \\frac{{{num_latex}}}{{{A_i[i,i]:.3f}}} = \\frac{{{num_val:.3f}}}{{{A_i[i,i]:.3f}}} = \mathbf{{{x_new[i]:.3f}}}")
                    
                error = np.max(np.abs(x_new - x_old))
                res_str = ", \ ".join([f"x_{i+1} = {x_new[i]:.3f}" for i in range(n_iter)])
                st.markdown(f"Resultados al final de la Iteración {k+1}: ${res_str}$.")
                st.caption(f"Error calculado: {error:.6f}")
                st.divider()
                
                if error < tol:
                    st.success(f"¡Convergencia lograda en la iteración {k+1}!")
                    break
                x_old = x_new.copy()
                
            else: # Gauss-Seidel
                st.write("Se sustituyen los valores iniciales. Cada resultado nuevo se utiliza inmediatamente en la siguiente fórmula:")
                for i in range(n_iter):
                    usando_list = []
                    for j in range(n_iter):
                        if j != i:
                            val_usado = x[j] if j < i else x_old[j]
                            usando_list.append(f"x_{j+1} = {val_usado:.3f}")
                    usando_str = " \text{ y } ".join(usando_list)
                    st.markdown(f"$\circ \quad$ **Cálculo de $x_{i+1}^{({k+1})}$**: Usando ${usando_str}$")
                    
                    suma = sum(A_i[i, j] * (x[j] if j < i else x_old[j]) for j in range(n_iter) if j != i)
                    x_new_val = (B_i[i] - suma) / A_i[i, i]
                    
                    terms = [f"{B_i[i]:.3f}"]
                    for j in range(n_iter):
                        if j != i:
                            val_sub = A_i[i, j] * (x[j] if j < i else x_old[j])
                            if val_sub < 0: terms.append(f"+ {abs(val_sub):.3f}")
                            else: terms.append(f"- {abs(val_sub):.3f}")
                    num_latex = " ".join(terms)
                    num_val = B_i[i] - suma
                    
                    st.latex(f"x_{{{i+1}}}^{({k+1})} = \\frac{{{num_latex}}}{{{A_i[i,i]:.3f}}} = \\frac{{{num_val:.3f}}}{{{A_i[i,i]:.3f}}} = \mathbf{{{x_new_val:.3f}}}")
                    x[i] = x_new_val
                    
                error = np.max(np.abs(x - x_old))
                res_str = ", \ ".join([f"x_{i+1} = {x[i]:.3f}" for i in range(n_iter)])
                st.markdown(f"Resultados al final de la Iteración {k+1}: ${res_str}$.")
                st.caption(f"Error calculado: {error:.6f}")
                st.divider()
                
                if error < tol:
                    st.success(f"¡Convergencia lograda en la iteración {k+1}!")
                    break
                x_old = x.copy()

# =====================================================================
# PESTAÑA 3: BISECCIÓN CON GRÁFICOS WEB
# =====================================================================
with tab3:
    st.header("Método de Bisección (Búsqueda de Raíces)")
    
    st.info("Basado en el **Teorema del Valor Intermedio (Bolzano)**. Si $f(a) \cdot f(b) < 0$, existe al menos una raíz en el intervalo $[a, b]$.")
    st.latex(r"c = \frac{a + b}{2} \quad \text{Error} = \left| \frac{b - a}{2} \right|")
    
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
                st.subheader("👨‍🏫 Desarrollo de Iteraciones:")
                c_final = None
                for k in range(int(max_bis)):
                    c = (a + b) / 2
                    fc = evaluar(c, formula)
                    err = abs(b - a) / 2
                    
                    # Decisión paso a paso
                    if fa * fc < 0:
                        decision = "f(a)*f(c) < 0 ➔ b cambia a c"
                    else:
                        decision = "f(a)*f(c) > 0 ➔ a cambia a c"
                        
                    st.markdown(f"**Iter {k+1}:** $c = \\frac{{{a:.4f} + {b:.4f}}}{{2}} = {c:.4f}$ | $f(c) = {fc:.4f}$ | Err= {err:.4f} | _{decision}_")
                    
                    if err < tol_bis or abs(fc) < 1e-9:
                        c_final = c
                        st.success(f"Raíz encontrada en x = {c:.6f}")
                        break
                    if fa * fc < 0:
                        b = c
                    else:
                        a = c
                        fa = fc
                
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
                st.pyplot(fig)
                
        except Exception as ex:
            st.error(f"Revisa tu fórmula o valores. Detalle: {ex}")