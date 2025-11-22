import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

# ---- App Title & UI ----
st.set_page_config(page_title="Newton-Raphson Optimizer", layout="wide")
st.title("Newton-Raphson Optimization Interactive Tool")
st.markdown("""
This app finds the minimum of your own mathematical function using the Newton-Raphson optimization method.
Build your function by typing it or using the calculator-style buttons below. Review each calculation step!
""")

# ---- Calculator-style input ----
st.subheader("Input your function with calculator buttons or by typing:")
if "func_input" not in st.session_state:
    st.session_state['func_input'] = ""

# Button rows for calculator
button_row1 = ['+', '-', '*', '/', '^', '(', ')']
button_row2 = ['x', 'sin(', 'cos(', 'exp(', 'log(', 'sqrt(']
button_row3 = ['Clear', 'Del']

cols1 = st.columns(len(button_row1))
cols2 = st.columns(len(button_row2))
cols3 = st.columns(len(button_row3))

for i, label in enumerate(button_row1):
    if cols1[i].button(label):
        st.session_state['func_input'] += label if label != '^' else '**'

for i, label in enumerate(button_row2):
    if cols2[i].button(label):
        st.session_state['func_input'] += label

for i, label in enumerate(button_row3):
    if cols3[i].button(label):
        if label == 'Clear':
            st.session_state['func_input'] = ""
        elif label == 'Del':
            st.session_state['func_input'] = st.session_state['func_input'][:-1]

func_str = st.text_input(
    "Function f(x):", 
    value=st.session_state['func_input'], 
    key='func_field'
)

# ---- Sidebar Inputs ----
st.sidebar.header("Computation Settings")
left = st.sidebar.number_input(
    "Left Interval Endpoint:",
    value=-2.0, format="%.2f"
)
right = st.sidebar.number_input(
    "Right Interval Endpoint:",
    value=8.0, format="%.2f"
)
tolerance_str = st.sidebar.text_input(
    "Tolerance (decimal, e.g., 0.01, 0.001):",
    value="0.001"
)
try:
    tolerance = float(tolerance_str)
    if tolerance <= 0 or tolerance > 0.1:
        st.sidebar.error("Enter a decimal between 0.000001 and 0.1.")
        tolerance = None
except ValueError:
    st.sidebar.error("Enter a valid decimal.")
    tolerance = None

max_iters = st.sidebar.number_input(
    "Max Iterations:",
    value=20,
    min_value=3,
    max_value=100
)
init_guess = st.sidebar.number_input(
    "Initial Guess x₀:",
    value=1.00,
    format="%.2f"
)
st.sidebar.caption("Change and click 'Run' to update results.")

# ---- Symbolic Parsing ----
x = sp.Symbol('x')
parsed_func = False
try:
    f = sp.sympify(func_str)
    f_prime = sp.diff(f, x)
    f_double_prime = sp.diff(f_prime, x)
    st.write(f"**Parsed Function:**<br>"
             f"- f(x): `{sp.pretty(f)}`<br>"
             f"- f'(x): `{sp.pretty(f_prime)}`<br>"
             f"- f''(x): `{sp.pretty(f_double_prime)}`", unsafe_allow_html=True)
    parsed_func = True
except Exception:
    st.error("Function Parse Error. See calculator help or try typing with proper math notation.")

st.markdown("""
**Calculator Help:**  
- Type or use calculator buttons: `+`, `-`, `*`, `/`, `(`, `)`, `^` for powers (auto-converts to `**` for Python).
- Use `x` for your variable, and functions like `sin(`, `cos(`, `exp(`, `log(`, `sqrt(`.
- Example: `x**3 - 6*x**2 + 4*x + 12`
""")

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
if st.button("Run Newton-Raphson Optimization") and parsed_func and tolerance is not None:
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
    | **e(tolerance):** As typed, no padded zeros
    | **Error < Tol?:** TRUE or FALSE (text)
    """)

st.markdown("""
**Instructions:**  
- Use calculator buttons or type your function directly using `x` and math operators/functions.
- Enter tolerance as a decimal (e.g., 0.01, 0.001, 0.0001).
- Adjust all other settings and click 'Run' for step-by-step results and graph!
""")
