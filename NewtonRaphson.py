import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

# ---- App Title & UI ----
st.set_page_config(page_title="Newton-Raphson Optimizer", layout="wide")
st.title("Newton-Raphson Optimization Interactive Tool")
st.markdown("""
Use this tool to find the minimum of any polynomial or mathematical function with the Newton-Raphson optimization method.
Each step is shown in a detailed table, and the function is plotted for visual clarity.
""")

# ---- Sidebar Input ----
with st.sidebar:
    st.header("User Input")
    func_str = st.text_input(
        "Function f(x) (e.g., x**3 - x + 4):", 
        value="x**3 - 6*x**2 + 4*x + 12"
    )
    left = st.number_input(
        "Left Interval Endpoint:",
        value=-2.0, format="%.2f"
    )
    right = st.number_input(
        "Right Interval Endpoint:",
        value=8.0, format="%.2f"
    )
    tolerance_str = st.text_input(
        "Tolerance (any decimal, e.g., 0.01, 0.001, 0.0001):",
        value="0.001"
    )
    try:
        tolerance = float(tolerance_str)
        if tolerance <= 0 or tolerance > 0.1:
            st.sidebar.error("Please enter a decimal between 0.000001 and 0.1.")
            tolerance = None
    except ValueError:
        st.sidebar.error("Please enter a valid decimal for tolerance.")
        tolerance = None

    max_iters = st.number_input(
        "Max Iterations:",
        value=20,
        min_value=3,
        max_value=100
    )
    init_guess = st.number_input(
        "Initial Guess x₀:",
        value=1.00,
        format="%.2f"
    )
    st.caption("Modify any field and click 'Run' to update results.")

# ---- Symbolic Parsing ----
x = sp.Symbol('x')
try:
    f = sp.sympify(func_str)
    f_prime = sp.diff(f, x)
    f_double_prime = sp.diff(f_prime, x)
    st.write(f"**Parsed Function:**<br>"
             f"- f(x): {sp.pretty(f)}<br>"
             f"- f'(x): {sp.pretty(f_prime)}<br>"
             f"- f''(x): {sp.pretty(f_double_prime)}", unsafe_allow_html=True)
except Exception:
    st.error("Function Parse Error: Please check your input syntax.")

# ---- Newton-Raphson Iteration ----
def newton_raphson_full(func, func_prime, func_double_prime, x0, tol, max_iter, tol_display):
    table = []
    x_i = x0
    for i in range(1, max_iter + 1):
        f_p = float(func_prime.evalf(subs={x: x_i}))
        f_pp = float(func_double_prime.evalf(subs={x: x_i}))
        if abs(f_pp) < 1e-10:
            st.warning("Second derivative near zero, method may fail.")
            break
        x_next = x_i - f_p / f_pp
        abs_err = abs(x_next - x_i)
        error_status = "TRUE" if abs_err < tol else "FALSE"
        table.append([
            i,                             # Iteration
            round(x_i, 6),                 # X_i
            round(f_p, 6),                 # f'(x)
            round(f_pp, 6),                # f''(x)
            round(x_next, 6),              # x_{n+1}
            round(abs_err, 6),             # |x_{n+1} - X_i|
            tol_display,                   # e(tolerance): display string
            error_status                   # TRUE/FALSE as string
        ])
        if abs_err < tol:
            break
        x_i = x_next
    return table, x_next

# ---- Run and Display Results ----
if st.button("Run Newton-Raphson Optimization"):
    if tolerance is not None:
        table, minimizer = newton_raphson_full(f, f_prime, f_double_prime, init_guess, tolerance, max_iters, tolerance_str)
        df = pd.DataFrame(table, columns=[
            "Iteration", 
            "X_i", 
            "f'(x)", 
            "f''(x)", 
            "x_n+1", 
            "|x_n+1 - X_i|", 
            "e(tolerance)", 
            "Error < Tol?"
        ])
        st.subheader("Newton-Raphson Iteration Table")
        st.dataframe(df, hide_index=True)

        st.success(f"Estimated minimizer: x = {minimizer:.6f}, f(x) = {float(f.evalf(subs={x: minimizer})): .6f}")

        # ---- Plot Function & Steps ----
        xs = np.linspace(left, right, 400)
        ys = [float(f.evalf(subs={x: val})) for val in xs]
        plt.figure(figsize=(10,5))
        plt.plot(xs, ys, label="f(x)", color="blue", linewidth=2)
        if len(table):
            it_xs = [row[1] for row in table]
            it_ys = [float(f.evalf(subs={x: val})) for val in it_xs]
            plt.scatter(it_xs, it_ys, color="red", s=55, label="Iterations")
            plt.plot(it_xs, it_ys, color="red", linestyle="--", alpha=0.5)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("f(x)", fontsize=12)
        plt.title("Function and Newton-Raphson Steps", fontsize=15)
        plt.legend()
        st.subheader("Function Plot (with Iteration Steps)")
        st.pyplot(plt, use_container_width=True)

        # ---- Table Explanation ----
        st.info("""
        | **Iteration:** Step number
        | **X_i:** Current \(x\) value
        | **f'(x):** First derivative at step
        | **f''(x):** Second derivative for update
        | **x_n+1:** Next approximation of \(x\)
        | **|x_n+1 - X_i|:** Absolute error for iteration
        | **e(tolerance):** Your selected stopping tolerance (as typed, no extra zeros)
        | **Error < Tol?:** Displays TRUE or FALSE as text.
        """)

st.markdown("""
**Instructions:**  
- Type in any polynomial or mathematical function.
- Enter tolerance as any decimal (e.g., 0.01, 0.001, 0.0001)—no padded zeros.
- Adjust interval for the plot.  
- Set guess and iteration controls.
- Review each calculation step in the table and graph!
""")
