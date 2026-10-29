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
# PESTAÑA 1: SISTEMAS DIRECTOS (Cramer, Gauss, Gauss-Jordan, LU)
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
    # Creamos contenedores dinámicos con valores iniciales automáticos
    matriz_A = []
    vector_B = []
    
    for i in range(n):
        cols = st.columns(n + 2)
        fila = []
        for j in range(n):
            # Diagonal de 10, resto valores positivos (ej. 2, 3, 4...)
            def_val = 10.0 if i == j else float(i + j + 1)
            val = cols[j].number_input(f"A[{i+1},{j+1}]", value=def_val, key=f"a_{i}_{j}", label_visibility="collapsed")
            fila.append(val)
        matriz_A.append(fila)
        cols[n].text(" | ")
        
        # Vector B inicia en 15, 20, 25...
        b_val = cols[n+1].number_input(f"B[{i+1}]", value=float(15 + i * 5), key=f"b_{i}", label_visibility="collapsed")
        vector_B.append(b_val)
        
    A = np.array(matriz_A, dtype=float)
    B = np.array(vector_B, dtype=float)
    
    metodo_sel = st.selectbox("Elige el método:", ["Cramer", "Gauss", "Gauss-Jordan", "Factorización LU"])
    
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
                soluciones_cramer = []
                for i in range(n):
                    A_mod = A.copy()
                    A_mod[:, i] = B
                    det_i = np.linalg.det(A_mod)
                    x_i = det_i / det_A
                    soluciones_cramer.append(x_i)
                    
                    st.markdown(f"$\circ \quad$ **Cálculo de $x_{i+1}$**: Sustituyendo la columna {i+1} por el vector $B$:")
                    st.latex(f"A_{{{i+1}}} = {matrix_to_latex(A_mod)} \\implies \det(A_{{{i+1}}}) = {det_i:.4f}")
                    st.latex(f"x_{{{i+1}}} = \\frac{{\det(A_{{{i+1}}})}}{{\det(A)}} = \\frac{{{det_i:.4f}}}{{{det_A:.4f}}} = \mathbf{{{x_i:.4f}}}")
                    st.divider()
                
                st.success("🎉 Solución final del sistema:")
                res_str = ", \ ".join([f"x_{i+1} = {soluciones_cramer[i]:.4f}" for i in range(n)])
                st.markdown(f"${res_str}$")
                    
        elif metodo_sel == "Gauss":
            st.subheader("👨‍🏫 Fundamento Matemático: Eliminación Gaussiana")
            st.info("Transforma el sistema en una matriz triangular superior mediante operaciones elementales de fila, y luego encuentra las incógnitas mediante sustitución hacia atrás.")
            
            Ab = np.column_stack((A, B))
            st.write("**1. Matriz Aumentada inicial $[A | B]$:**")
            st.latex(matrix_to_latex(Ab))
            
            st.write("**2. Proceso de Triangulación:**")
            for i in range(n):
                for j in range(i+1, n):
                    if Ab[i, i] == 0:
                        st.error("Se encontró un 0 en el pivote. Se requiere intercambio de filas (no implementado en esta versión básica).")
                        st.stop()
                    factor = Ab[j, i] / Ab[i, i]
                    Ab[j] = Ab[j] - factor * Ab[i]
                    st.text(f"F{j+1} = F{j+1} - ({factor:.4f}) * F{i+1}")
            
            st.write("**Matriz Triangulada final:**")
            st.latex(matrix_to_latex(Ab))
            
            st.write("**3. Sustitución hacia atrás:**")
            x_gauss = np.zeros(n)
            for i in range(n-1, -1, -1):
                suma = sum(Ab[i, j] * x_gauss[j] for j in range(i+1, n))
                x_gauss[i] = (Ab[i, n] - suma) / Ab[i, i]
                
            st.success("🎉 Solución final del sistema:")
            res_str = ", \ ".join([f"x_{i+1} = {x_gauss[i]:.4f}" for i in range(n)])
            st.markdown(f"${res_str}$")

        elif metodo_sel == "Gauss-Jordan":
            st.subheader("👨‍🏫 Fundamento Matemático: Gauss-Jordan")
            st.info("Es una extensión del método de Gauss. Continúa realizando operaciones entre filas hasta que la matriz de coeficientes se convierte en la matriz identidad. La columna de resultados contendrá la respuesta directa sin necesidad de sustitución.")
            
            Ab = np.column_stack((A, B)).astype(float)
            st.write("**1. Matriz Aumentada inicial $[A | B]$:**")
            st.latex(matrix_to_latex(Ab))
            
            st.write("**2. Proceso de Eliminación completa:**")
            for i in range(n):
                pivote = Ab[i, i]
                if pivote == 0:
                    st.error("Se encontró un 0 en el pivote. Se requiere intercambio de filas.")
                    st.stop()
                    
                # Convertir el pivote en 1
                Ab[i] = Ab[i] / pivote
                st.text(f"F{i+1} = F{i+1} / {pivote:.4f} (Normalizar pivote)")
                
                # Hacer ceros en toda la columna (arriba y abajo)
                for j in range(n):
                    if i != j:
                        factor = Ab[j, i]
                        Ab[j] = Ab[j] - factor * Ab[i]
                        st.text(f"F{j+1} = F{j+1} - ({factor:.4f}) * F{i+1}")
            
            st.write("**Matriz Reducida (Identidad) final $[I | X]$:**")
            st.latex(matrix_to_latex(Ab))
            
            st.success("🎉 Solución final del sistema:")
            res_str = ", \ ".join([f"x_{i+1} = {Ab[i, n]:.4f}" for i in range(n)])
            st.markdown(f"${res_str}$")
            
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
                        st.error("División por cero detectada en el pivote.")
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
            
            st.success("🎉 Solución final del sistema:")
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
            # Mismo patrón diagonalmente dominante para iteraciones
            def_val = 10.0 if i == j else float(i + j + 1)
            val = cols[j].number_input(f"AI[{i+1},{j+1}]", value=def_val, key=f"ai_{i}_{j}", label_visibility="collapsed")
            fila.append(val)
        mat_ai.append(fila)
        cols[n_iter].text(" | ")
        
        b_val = cols[n_iter+1].number_input(f"BI[{i+1}]", value=float(15 + i * 5), key=f"bi_{i}", label_visibility="collapsed")
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
# PESTAÑA 3: BÚSQUEDA DE RAÍCES (Bisección, Falsa Posición, Newton-Raphson)
# =====================================================================
with tab3:
    st.header("Métodos Numéricos para Búsqueda de Raíces")
    
    metodo_raiz = st.radio("Selecciona el método a utilizar:", 
                           ["Bisección", "Falsa Posición", "Newton-Raphson"], horizontal=True)
    
    formula = st.text_input("Ingresa la función f(x) (p. ej. e**-x - x):", value="e**-x - x")
    
    # --- Interfaz adaptativa según el método ---
    if metodo_raiz in ["Bisección", "Falsa Posición"]:
        col1, col2, col3, col4 = st.columns(4)
        a_val = col1.number_input("Intervalo a:", value=-1.0, format="%.4f")
        b_val = col2.number_input("Intervalo b:", value=1.0, format="%.4f")
        tol_raiz = col3.number_input("Tolerancia % (E_s):", value=1.0, format="%.4f")
        max_iter = col4.number_input("Máx Iteraciones:", value=5, step=1)
    else: # Newton-Raphson
        formula_deriv = st.text_input("Ingresa la derivada f'(x) (p. ej. -e**-x - 1):", value="-e**-x - 1")
        col1, col2, col3 = st.columns(3)
        x0_val = col1.number_input("Valor inicial x0:", value=0.0, format="%.4f")
        tol_raiz = col2.number_input("Tolerancia % (E_s):", value=1.0, format="%.4f")
        max_iter = col3.number_input("Máx Iteraciones:", value=5, step=1)

    def evaluar(x_num, form):
        entorno = {
            "x": x_num, 
            "np": np, 
            "sin": np.sin, 
            "cos": np.cos, 
            "tan": np.tan, 
            "exp": np.exp, 
            "sqrt": np.sqrt,
            "sen": np.sin,
            "e": np.e
        }
        return eval(form, {"__builtins__": {}}, entorno)

    if st.button("Ejecutar Método"):
        try:
            # ==============================================================
            # LÓGICA BISECCIÓN Y FALSA POSICIÓN
            # ==============================================================
            if metodo_raiz in ["Bisección", "Falsa Posición"]:
                a, b = a_val, b_val
                fa = evaluar(a, formula)
                fb = evaluar(b, formula)
                
                if fa * fb > 0:
                    st.error("⚠️ f(a) y f(b) tienen el mismo signo. No se garantiza una raíz en este intervalo.")
                else:
                    st.subheader(f"👨‍🏫 Desarrollo de Iteraciones: {metodo_raiz}")
                    
                    if metodo_raiz == "Bisección":
                        st.latex(r"c_i = \frac{a_i + b_i}{2}")
                    else:
                        st.latex(r"x_r = \frac{a_i \cdot f(b_i) - b_i \cdot f(a_i)}{f(b_i) - f(a_i)}")
                    
                    prev_xr = None
                    for k in range(int(max_iter)):
                        i = k + 1
                        fa_old, fb_old = evaluar(a, formula), evaluar(b, formula)
                        
                        if metodo_raiz == "Bisección":
                            xr = (a + b) / 2
                        else: # Falsa Posición
                            xr = (a * fb_old - b * fa_old) / (fb_old - fa_old)
                            
                        fxr = evaluar(xr, formula)
                        
                        # Cálculo del error relativo aproximado (Ea)
                        ea = 100.0 if prev_xr is None else abs((xr - prev_xr) / xr) * 100
                        
                        st.markdown(f"### Iteración {i} $\quad \therefore i = {i}$")
                        st.markdown(f"**1:** $[a_{i}, b_{i}] = [{a:.6f}, {b:.6f}]$")
                        st.markdown(f"**2:** $f(a_{i}) = {fa_old:.6f} \quad ; \quad f(b_{i}) = {fb_old:.6f}$")
                        
                        if metodo_raiz == "Bisección":
                            st.latex(f"c = \\frac{{{a:.6f} + {b:.6f}}}{{2}} = {xr:.6f}")
                        else:
                            st.latex(f"x_r = \\frac{{({a:.6f})({fb_old:.6f}) - ({b:.6f})({fa_old:.6f})}}{{{fb_old:.6f} - ({fa_old:.6f})}} = {xr:.6f}")
                        
                        st.markdown(f"**4: Evaluar**")
                        st.markdown(f"$f(x_r) = f({xr:.6f}) = {fxr:.6f}$")
                        
                        # Análisis de signos
                        prod = fa_old * fxr
                        signo_prod = "\\le 0" if prod <= 0 else "> 0"
                        st.markdown(f"Analizamos: $f(a_i) \cdot f(x_r) = ({fa_old:.6f}) \cdot ({fxr:.6f}) {signo_prod}$")
                        
                        if prod <= 0:
                            st.markdown(f"$\therefore$ Nuevo intervalo: $[{a:.6f}, {xr:.6f}]$")
                        else:
                            st.markdown(f"$\therefore$ Nuevo intervalo: $[{xr:.6f}, {b:.6f}]$")
                            
                        if i > 1:
                            st.markdown(f"**5:** $E_a = \\left| \\frac{{{xr:.6f} - {prev_xr:.6f}}}{{{xr:.6f}}} \\right| \\cdot 100 = {ea:.2f}\\%$")
                            
                        # Gráfico visual de la iteración
                        fig, ax = plt.subplots(figsize=(6, 3))
                        x_pts = np.linspace(a - 0.5, b + 0.5, 300)
                        y_pts = [evaluar(x_pt, formula) for x_pt in x_pts]
                        
                        ax.plot(x_pts, y_pts, color="#4A90E2", linewidth=2, label="$f(x)$")
                        ax.axhline(0, color="black", linewidth=1)
                        
                        if metodo_raiz == "Falsa Posición":
                            # Dibuja la línea recta que conecta los puntos
                            ax.plot([a, b], [fa_old, fb_old], color="#c0392b", linestyle="--", alpha=0.7, label="Secante")
                            
                        ax.scatter([a, b, xr], [fa_old, fb_old, fxr], color=["#c0392b", "#27ae60", "#f39c12"], zorder=5)
                        ax.axvline(xr, color="#f39c12", linestyle=":", alpha=0.7)
                        
                        ax.set_title(f"Iteración {i}", fontsize=9)
                        ax.grid(True, linestyle=":", alpha=0.6)
                        if metodo_raiz == "Falsa Posición":
                            ax.legend()
                        st.pyplot(fig)
                        st.divider()
                        
                        # Actualización de valores
                        if ea <= tol_raiz and i > 1:
                            st.success(f"🎉 Condición cumplida ($E_a = {ea:.2f}\\% \le {tol_raiz}\\%$). Raíz en: {xr:.6f}")
                            break
                            
                        if prod <= 0:
                            b = xr
                        else:
                            a = xr
                            
                        prev_xr = xr

            # ==============================================================
            # LÓGICA NEWTON-RAPHSON
            # ==============================================================
            elif metodo_raiz == "Newton-Raphson":
                st.subheader(f"👨‍🏫 Desarrollo de Iteraciones: Newton-Raphson")
                st.latex(r"x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}")
                
                xi = x0_val
                
                for k in range(int(max_iter)):
                    i = k + 1
                    
                    f_xi = evaluar(xi, formula)
                    df_xi = evaluar(xi, formula_deriv)
                    
                    if df_xi == 0:
                        st.error(f"La derivada se hace cero en x={xi:.6f}. El método falla.")
                        break
                        
                    xi_next = xi - (f_xi / df_xi)
                    
                    ea = abs((xi_next - xi) / xi_next) * 100 if xi_next != 0 else 100.0
                    
                    st.markdown(f"### Iteración {i}")
                    st.markdown(f"$x_{k} = {xi:.6f}$")
                    st.markdown(f"$f(x_{k}) = {f_xi:.6f}$")
                    st.markdown(f"$f'(x_{k}) = {df_xi:.6f}$")
                    
                    st.latex(f"x_{k+1} = {xi:.6f} - \\frac{{{f_xi:.6f}}}{{{df_xi:.6f}}} = {xi_next:.6f}")
                    st.markdown(f"$E_a = {ea:.2f}\\%$")
                    
                    # Gráfico
                    fig, ax = plt.subplots(figsize=(6, 3))
                    x_span = abs(xi_next - xi) * 2 if xi_next != xi else 1
                    x_pts = np.linspace(min(xi, xi_next) - x_span, max(xi, xi_next) + x_span, 300)
                    y_pts = [evaluar(x_pt, formula) for x_pt in x_pts]
                    
                    ax.plot(x_pts, y_pts, color="#4A90E2", linewidth=2, label="$f(x)$")
                    ax.axhline(0, color="black", linewidth=1)
                    
                    # Recta tangente
                    # y - y1 = m(x - x1)  =>  y = m(x - x1) + y1
                    y_tangente = [df_xi * (pt - xi) + f_xi for pt in x_pts]
                    ax.plot(x_pts, y_tangente, color="#c0392b", linestyle="--", label="Tangente")
                    
                    ax.scatter([xi, xi_next], [f_xi, 0], color=["#c0392b", "#f39c12"], zorder=5)
                    ax.axvline(xi, color="#c0392b", linestyle=":", alpha=0.3)
                    ax.axvline(xi_next, color="#f39c12", linestyle=":", alpha=0.7)
                    
                    ax.set_title(f"Iteración {i}", fontsize=9)
                    ax.grid(True, linestyle=":", alpha=0.6)
                    ax.legend()
                    st.pyplot(fig)
                    st.divider()
                    
                    if ea <= tol_raiz:
                        st.success(f"🎉 Condición cumplida ($E_a = {ea:.2f}\\% \le {tol_raiz}\\%$). Raíz en: {xi_next:.6f}")
                        break
                        
                    xi = xi_next

        except Exception as ex:
            st.error(f"Revisa tu fórmula o los valores de entrada. Detalle técnico: {ex}")