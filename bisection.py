import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import re

# --- Preprocess user input for math syntax ---
def preprocess_math_input(expr):
    # Replace ^ with **
    expr = re.sub(r'(\w|\))\^(\w|\()', r'\1**\2', expr)
    # Add * between number and variable/function/paren: 4x, 2sin(x)
    expr = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', expr)
    # Add * between close paren and variable: )(x) -> )*x
    expr = re.sub(r'(\))(x)', r'\1*\2', expr)
    # Add * between variable/function and (: x(y) -> x*(y)
    expr = re.sub(r'([a-zA-Z])(\()', r'\1*\2', expr)
    return expr

# --- Streamlit config ---
st.set_page_config(page_title="Bisection Optimizer", layout="wide")
st.title("Bisection Optimization Interactive Tool")

# --- Session state for expression ---
if 'func_input' not in st.session_state:
    st.session_state['func_input'] = ""

# --- Insert calculator symbol with implicit multiplication support ---
def insert_symbol(symbol):
    s = st.session_state['func_input']
    needs_mult = False
    START_TOKENS = ['x', '(', 'sin(', 'cos(', 'tan(', 'log(', 'ln(', 'sqrt(', 'pi', 'e']
    last_char = s[-1] if s else ''
    last_is_num = re.match(r'[0-9.]', last_char)
    is_new_num = re.match(r"[0-9.]", symbol)
    is_new_start_token = symbol in START_TOKENS
    if symbol == '^':
        symbol = '^'
    if is_new_start_token and (last_char == 'x' or last_char == ')' or last_is_num):
        needs_mult = True
    elif is_new_num and (last_char == 'x' or last_char == ')'):
        needs_mult = True
    if symbol == 'ln(':
        symbol = 'ln('
    elif symbol == 'log(':
        symbol = 'log('
    elif symbol == 'pi':
        symbol = 'pi'
    elif symbol == 'e':
        symbol = 'e'
    elif symbol == 'sqrt(':
        symbol = 'sqrt('
    elif symbol == 'tan(':
        symbol = 'tan('
    st.session_state['func_input'] += ('*' if needs_mult else '') + symbol

# --- Calculator Button Grid ---
button_rows = [
    ['7', '8', '9', '/', '^', 'Del', 'Clear'],
    ['4', '5', '6', '*', 'sqrt(', 'pi', 'e'],
    ['1', '2', '3', '-', 'sin(', 'cos(', 'tan('],
    ['0', '.', '(', ')', '+', 'ln(', 'log(']
]
for row in button_rows:
    cols = st.columns(7)
    for i, label in enumerate(row):
        if cols[i].button(label, key=f"btn_{label}_{i}"):
            if label == "Clear":
                st.session_state['func_input'] = ""
            elif label == "Del":
                st.session_state['func_input'] = st.session_state['func_input'][:-1]
            else:
                insert_symbol(label)

# --- Editable textbox for both keyboard and calculator ---
func_str = st.text_input(
    "Function f(x):",
    value=st.session_state['func_input'],
    key='func_field',
    help="Type your function as written in mathematics (e.g., 2x^3 + 5x^2 - 4x + 1)."
)
# Sync session with keyboard edits
st.session_state['func_input'] = func_str

# --- Preprocess the input for implicit multiplication and powers ---
prepped_func_str = preprocess_math_input(func_str)

# --- Show LaTeX version for visual confirmation ---
try:
    st.latex(f"f(x) = {func_str}")
except:
    st.write(f"f(x) = {func_str}")

x = sp.Symbol('x')
st.sidebar.header("Computation Settings")
left = st.sidebar.number_input("Left Interval Endpoint (a):", value=0.0, format="%.2f")
right = st.sidebar.number_input("Right Interval Endpoint (b):", value=3.0, format="%.2f")
tolerance = st.sidebar.number_input("Tolerance:", value=0.001, min_value=1e-6, max_value=0.1, format="%.6f")
max_iters = st.sidebar.number_input("Max Iterations:", value=20, min_value=5, max_value=100)

# --- Try to parse function ---
try:
    f = sp.sympify(prepped_func_str)
    f_prime = sp.diff(f, x)
    parsed_func = True
except Exception:
    st.error("Function Parse Error. Please check your syntax. Use math-like syntax (2x^3 + ...), avoid missing operators.")
    parsed_func = False

# --- Bisection method ---
def bisection_method(func_prime, a, b, tol, max_iter):
    table = []
    fa = float(func_prime.evalf(subs={x: a}))
    fb = float(func_prime.evalf(subs={x: b}))
    if fa * fb > 0:
        st.error("Derivative at endpoints must have opposite signs.")
        return table, None
    for i in range(max_iter+1):
        c = (a + b) / 2.0
        fc = float(func_prime.evalf(subs={x: c}))
        interval_width = abs(b - a)
        table.append([
            i, round(a, 6), round(b, 6), round(c, 6),
            round(fc, 6), round(interval_width, 6), tol
        ])
        if abs(fc) < 1e-12 or (interval_width < tol):
            return table, c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return table, c

if st.button("Run Bisection Optimization") and parsed_func:
    table, root = bisection_method(f_prime, left, right, tolerance, max_iters)
    if table:
        df = pd.DataFrame(table, columns=[
            "Iteration", "a", "b", "c", "f(c)", "|b-a|", f"{tolerance}"
        ])
        st.markdown("#### Bisection Iteration Table")
        st.dataframe(df, hide_index=True, use_container_width=True, height=min(len(df)*35+40, 400))
        final_fx = float(f.evalf(subs={x: root}))
        st.success(f"Estimated critical point (root of derivative): x = {root:.6f}, f(x) = {final_fx:.6f}")
        xs = np.linspace(left, right, 400)
        ys = [float(f.evalf(subs={x: val})) for val in xs]
        plt.figure(figsize=(10,5))
        plt.plot(xs, ys, label="$f(x)$", color="blue", linewidth=2)
        cs = [row[3] for row in table]
        ys_cs = [float(f.evalf(subs={x: val})) for val in cs]
        plt.scatter(cs, ys_cs, color="red", s=55, label="Bisection Midpoints")
        plt.plot(cs, ys_cs, color="red", linestyle="--", alpha=0.5)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Function and Bisection Steps")
        plt.legend()
        st.pyplot(plt, use_container_width=True)
    else:
        st.error("No iterations performed. Check interval and derivative sign change.")

st.markdown("""
*You may type math-like polynomials (e.g., `2x^3 + 5x^2 - 4x + 1`) or use the calculator buttons.  
Multiplication and powers are inferred for mathematical syntax; functions like sin, cos, tan, ln, log, sqrt, pi, and e are supported.*
""")
