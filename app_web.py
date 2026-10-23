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
# PESTAÑA 3: BISECCIÓN CON GRÁFICOS WEB (Basado en Apuntes del Pizarrón)
# =====================================================================
with tab3:
    st.header("Método de Bisección (Búsqueda de Raíces)")
    
    st.info("Basado en el **Teorema del Valor Intermedio (Bolzano)**. Si $f(a) \cdot f(b) < 0$, existe al menos una raíz en el intervalo $[a, b]$ para una función continua.")
    st.latex(r"\text{c.i: } [a_i, b_i] \quad \therefore c_i = \frac{a_i + b_i}{2}, \quad f(c_i)=0 \implies \text{Raíz}")
    
    # Campo para la función (usamos np.sin para soportar funciones trigonométricas)
    formula = st.text_input("Ingresa la función f(x) (p. ej. np.sin(x) + 2*x - 1):", value="np.sin(x) + 2*x - 1")
    
    col1, col2, col3, col4 = st.columns(4)
    a_val = col1.number_input("Intervalo a:", value=-2.0)
    b_val = col2.number_input("Intervalo b:", value=2.0)
    tol_bis = col3.number_input("Tolerancia:", value=0.01, format="%.4f")
    max_bis = col4.number_input("Máx Iteraciones:", value=5, step=1)
    
    def evaluar(x_num, form):
        entorno = {
            "x": x_num, 
            "np": np, 
            "sin": np.sin, 
            "cos": np.cos, 
            "tan": np.tan, 
            "exp": np.exp, 
            "sqrt": np.sqrt,
            "sen": np.sin  # Soporte para que acepte escribir 'sen' como en el pizarrón
        }
        return eval(form, {"__builtins__": {}}, entorno)

    if st.button("Ejecutar Bisección"):
        try:
            a, b = a_val, b_val
            fa = evaluar(a, formula)
            fb = evaluar(b, formula)
            
            if fa * fb > 0:
                st.error("⚠️ f(a) y f(b) tienen el mismo signo. No se garantiza una raíz en este intervalo.")
            else:
                st.subheader("👨‍🏫 Desarrollo de Iteraciones (Como en clase):")
                
                c_final = None
                for k in range(int(max_bis)):
                    i = k + 1  # Índice de iteración
                    c = (a + b) / 2
                    fc = evaluar(c, formula)
                    err = abs(b - a) / 2
                    
                    # Guardamos los límites para el texto antes de actualizar
                    a_old, b_old = a, b
                    fa_old, fb_old = fa, fb
                    
                    st.markdown(f"### Iteración {i} $\quad i = {i}$")
                    st.markdown(f"**1.-** $[a_{i}, b_{i}] = [{a_old:.4f}, {b_old:.4f}]$")
                    
                    # Detalle del cálculo de c
                    st.latex(f"c_{i} = \\frac{{{a_old:.4f} + {b_old:.4f}}}{{2}} = {c:.4f}")
                    
                    st.markdown("**3.- Evaluar (Sustituir $a, b, c$ en $f(x)$):**")
                    st.markdown(f"$f(a_{i}) = f({a_old:.4f}) = {fa_old:.4f}$")
                    st.markdown(f"$f(b_{i}) = f({b_old:.4f}) = {fb_old:.4f}$")
                    st.markdown(f"$f(c_{i}) = f({c:.4f}) = {fc:.4f}$")
                    
                    # Decisión del cambio de intervalo
                    if fa_old * fc < 0:
                        decision_intervalo = f"$[a_{i+1}, b_{i+1}] = [{a_old:.4f}, {c:.4f}]$"
                        marcador_decisivo = f"f(a_{i}) \\cdot f(c_{i}) < 0 \\implies b \\text{{ cambia a }} c"
                    else:
                        decision_intervalo = f"$[a_{i+1}, b_{i+1}] = [{c:.4f}, {b_old:.4f}]$"
                        marcador_decisivo = f"f(a_{i}) \\cdot f(c_{i}) > 0 \\implies a \\text{{ cambia a }} c"
                        
                    st.latex(marcador_decisivo)
                    st.markdown(f"**Siguiente Intervalo:** {decision_intervalo} (Error actual: $\\pm{err:.4f}$)")
                    
                    # --- DISEÑO DEL GRÁFICO TIPO PIZARRÓN ---
                    fig, ax = plt.subplots(figsize=(6, 3))
                    x_pts = np.linspace(a_old - 0.5, b_old + 0.5, 300)
                    y_pts = [evaluar(x_pt, formula) for x_pt in x_pts]
                    
                    ax.plot(x_pts, y_pts, color="#4A90E2", linewidth=2, label="$f(x)$")
                    ax.axhline(0, color="#2c3e50", linewidth=1)
                    
                    # Líneas de referencia y puntos
                    ax.axvline(a_old, color="#c0392b", linestyle="--", alpha=0.5)
                    ax.axvline(b_old, color="#27ae60", linestyle="--", alpha=0.5)
                    ax.axvline(c, color="#f39c12", linestyle="--", alpha=0.7)
                    
                    ax.scatter([a_old, b_old, c], [fa_old, fb_old, fc], color=["#c0392b", "#27ae60", "#f39c12"], s=60, zorder=5)
                    
                    # Etiquetas de coordenadas
                    ax.text(a_old, fa_old, f" (a, f(a))\n ({a_old:.1f}, {fa_old:.2f})", color="#c0392b", fontsize=8, ha="right")
                    ax.text(b_old, fb_old, f" (b, f(b))\n ({b_old:.1f}, {fb_old:.2f})", color="#27ae60", fontsize=8, ha="left")
                    ax.text(c, fc, f" (c, f(c))\n ({c:.1f}, {fc:.2f})", color="#f39c12", fontsize=8, ha="center")
                    
                    # Resaltar intervalo y decisión
                    ax.axvspan(a_old, b_old, color="#f1c40f", alpha=0.1)
                    if fa_old * fc < 0:
                        ax.annotate(f"Nuevo: [{a_old:.2f}, {c:.2f}]", xy=((a_old + c)/2, 0), xytext=((a_old + c)/2, max(y_pts)*0.3),
                                    arrowprops=dict(facecolor="#2980b9", shrink=0.05, width=1, headwidth=6), color="#2980b9", ha="center")
                    else:
                        ax.annotate(f"Nuevo: [{c:.2f}, {b_old:.2f}]", xy=((c + b_old)/2, 0), xytext=((c + b_old)/2, max(y_pts)*0.3),
                                    arrowprops=dict(facecolor="#2980b9", shrink=0.05, width=1, headwidth=6), color="#2980b9", ha="center")
                    
                    ax.set_title(f"Visualización Iteración {i}", fontsize=9)
                    ax.grid(True, linestyle=":", alpha=0.6)
                    st.pyplot(fig)
                    st.divider()
                    
                    # Verificar criterio de parada
                    if err < tol_bis or abs(fc) < 1e-9:
                        c_final = c
                        st.success(f"🎉 ¡Convergencia lograda! Raíz estimada en c = {c:.6f} con error de ±{err:.6f}")
                        break
                        
                    # Actualización de intervalos
                    if fa_old * fc < 0:
                        b = c
                        fb = fc
                    else:
                        a = c
                        fa = fc
                        
        except Exception as ex:
            st.error(f"Revisa tu fórmula o los valores de los intervalos. Detalle técnico: {ex}")