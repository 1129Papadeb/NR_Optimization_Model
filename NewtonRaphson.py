import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import re

# ---- Page config (must be first Streamlit command) ----
st.set_page_config(page_title="Newton-Raphson Optimizer", layout="wide")

# ---- Compact Button & Visibility Fix CSS ----
st.markdown("""
    <style>
    /* 1. TIGHTEN COLUMN SPACING */
    div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
        /* Reduce padding between columns dramatically */
        padding-right: 0.1rem !important; 
        padding-left: 0.1rem !important;
        /* Ensure columns shrink around content */
        flex: 1 1 auto !important;
        min-width: unset !important;
    }
    
    /* 2. ENSURE BUTTONS ARE VISIBLE AND SPACIOUS - Adjusted min-width for 7-column grid */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        /* Increase padding for better visual spacing and visibility */
        padding: 0.5rem 0.5rem !important; 
        /* Adjusted min-width slightly lower to fit 7 columns cleanly */
        min-width: 40px !important; 
        font-size: 0.92rem !important;
        
        /* THEME STYLES */
        color: black !important;
        background: #f4f4f4 !important;
        border-radius: 6px !important;
        border: 1px solid #ddd !important;
    }
    
    /* DARK THEME FIX */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            color: white !important;
            background: #2b2b2b !important; /* Darker background */
            border: 1px solid #555 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True
)

# ---- App Title & UI ----
st.title("Newton-Raphson Optimization Interactive Tool")
st.markdown("""
Type your function directly or use the calculator buttons below. **Implicit multiplication** is supported (e.g., `2x`, `5(x+1)`, `x sin(x)`).
Example: `x^3 - 6x^2 + 4x + 12`, `e^(-x) cos(x)`
""")

# ---- Improved Calculator Input ----
if 'func_input' not in st.session_state:
    st.session_state['func_input'] = ""

def insert_symbol(symbol):
    s = st.session_state['func_input']
    needs_mult = False
    
    # 1. Handle Power operator conversion
    if symbol == '^':
        symbol = '**'
    
    # Define tokens/characters that START an expression (right side of implicit multiplication)
    START_TOKENS = ['x', '(', 'sin(', 'cos(', 'tan(', 'log(', 'ln(', 'sqrt(', 'pi', 'e']
    
    # Define characters/types that END an expression (left side of implicit multiplication)
    last_char = s[-1] if s else ''
    last_is_num = re.match(r'[0-9.]', last_char)
    
    # Check if the new symbol needs multiplication
    is_new_num = re.match(r"[0-9.]", symbol)
    is_new_start_token = symbol in START_TOKENS

    # Case A: New token STARTS an expression (x, (, func, pi, e) following an ENDING token (x, ), number)
    if is_new_start_token and (last_char == 'x' or last_char == ')' or last_is_num):
        needs_mult = True
    
    # Case B: New token is a number following 'x' or ')'
    elif is_new_num and (last_char == 'x' or last_char == ')'):
        needs_mult = True

    # 2. Handle SymPy syntax conversions for math functions and constants
    if symbol == 'ln(':
        symbol = 'log(' # SymPy's log() is natural log (ln)
    elif symbol == 'log(':
        symbol = 'log10(' # SymPy's log10() is base 10 log
    elif symbol == 'pi':
        symbol = 'pi' # Insert symbolic pi
    elif symbol == 'e':
        symbol = 'E' # Insert symbolic Euler's number
    elif symbol == 'sqrt(':
        symbol = 'sqrt('
    elif symbol == 'tan(':
        symbol = 'tan('

    # 3. Append the symbol (with optional multiplication)
    st.session_state['func_input'] += ('*' if needs_mult else '') + symbol

# --- Button Grid (Full Scientific Layout - 7 columns) ---
button_rows = [
    ['7', '8', '9', '/', '^', 'Del', 'Clear'],
    ['4', '5', '6', '*', 'sqrt(', 'pi', 'e'],
    ['1', '2', '3', '-', 'sin(', 'cos(', 'tan('],
    ['0', '.', '(', ')', '+', 'ln(', 'log(']
]

for row in button_rows:
    # Use 7 columns to match the button row length
    cols = st.columns(7) 
    for i, label in enumerate(row):
        if cols[i].button(label, key=f"btn_{label}_{i}"):
            if label == "Clear":
                st.session_state['func_input'] = ""
            elif label == "Del":
                st.session_state['func_input'] = st.session_state['func_input'][:-1]
            else:
                insert_symbol(label)

func_str = st.text_input(
    "Function f(x):", 
    value=st.session_state['func_input'],
    key='func_field'
)

# ---- Sidebar Inputs ----
st.sidebar.header("Computation Settings")
left = st.sidebar.number_input("Left Interval Endpoint:", value=-2.0, format="%.2f")
right = st.sidebar.number_input("Right Interval Endpoint:", value=8.0, format="%.2f")
tolerance_str = st.sidebar.text_input("Tolerance (decimal, e.g., 0.001):", value="0.001")
try:
    tolerance = float(tolerance_str)
    if tolerance <= 0 or tolerance > 0.1:
        st.sidebar.error("Enter a decimal between 0.000001 and 0.1.")
        tolerance = None
except ValueError:
    st.sidebar.error("Enter a valid decimal.")
    tolerance = None

max_iters = st.sidebar.number_input("Max Iterations:", value=20, min_value=3, max_value=100)
init_guess = st.sidebar.number_input("Initial Guess $x_0$:", value=1.00, format="%.2f")
st.sidebar.caption("Change any input and click 'Run' for updates.")

# ---- Symbolic Parsing ----
x = sp.Symbol('x')
parsed_func = False
try:
    # SymPy correctly interprets 'pi' and 'E' as constants.
    # We must replace 'E' with 'exp(1)' if we want the exponential function, 
    # but since the button inserts the symbolic constant 'E', we keep it as is.
    f = sp.sympify(func_str)
    f_prime = sp.diff(f, x)
    f_double_prime = sp.diff(f_prime, x)
    st.write(f"**Parsed Function:**<br>"
             f"- $f(x)$: `{sp.pretty(f)}`<br>"
             f"- $f'(x)$: `{sp.pretty(f_prime)}`<br>"
             f"- $f''(x)$: `{sp.pretty(f_double_prime)}`", unsafe_allow_html=True)
    parsed_func = True
except Exception:
    st.error("Function Parse Error. Try using supported math syntax or calculator buttons.")

st.markdown("""
**Calculator Help:**
- **Functions:** $\\sin(x)$, $\\cos(x)$, $\\tan(x)$, $\\sqrt{x}$, $\\ln(x)$ (natural log), $\\log(x)$ (base 10 log).
- **Constants:** $\\pi$ (pi), $e$ (Euler's number).
- **Powers:** Use `^` (e.g., `x^2`) which is automatically converted to Python's `**`.
- **Multiplication:** Use implicit multiplication (e.g., `5x`, `2\\pi`, `3(x+1)`).
""")

# ---- Newton-Raphson Iteration ----
def newton_raphson_full(func, func_prime, func_double_prime, x0, tol, max_iter, tol_display):
    table = []
    x_i = x0
    for i in range(1, max_iter + 1):
        try:
            # Evaluate using evalf
            f_p = float(func_prime.evalf(subs={x: x_i}))
            f_pp = float(func_double_prime.evalf(subs={x: x_i}))
        except (TypeError, ValueError):
            st.error(f"Iteration {i}: Error evaluating function derivatives at $x = {x_i:.6f}$. Check for domain errors (e.g., $\\log(0)$, $\\sqrt{-1}$).")
            break

        if abs(f_pp) < 1e-10:
            st.warning(f"Iteration {i}: Second derivative ($f''({x_i:.6f}) = {f_pp:.6e}$) is near zero, method may fail.")
            break
            
        x_next = x_i - f_p / f_pp
        abs_err = abs(x_next - x_i)
        error_status = "TRUE" if abs_err < tol else "FALSE"
        
        # Calculate f(x) for display purposes in the final success message
        f_i = float(func.evalf(subs={x: x_i}))
        
        table.append([
            i, round(x_i, 6), round(f_p, 6), round(f_pp, 6),
            round(x_next, 6), round(abs_err, 6),
            tol_display, error_status
        ])
        
        if abs_err < tol:
            break
        
        x_i = x_next
        
    # Return the last calculated x_next (which is the minimizer)
    return table, x_next

# ---- Run and Display Results ----
if st.button("Run Newton-Raphson Optimization") and parsed_func and tolerance is not None:
    try:
        table, minimizer = newton_raphson_full(f, f_prime, f_double_prime, init_guess, tolerance, max_iters, tolerance_str)
        
        if table:
            df = pd.DataFrame(table, columns=[
                "Iteration", "$x_i$", "$f'(x_i)$", "$f''(x_i)$", "$x_{i+1}$", "$|x_{i+1} - x_i|$", "$\\epsilon$ (Tol)", "Error < Tol?"
            ])
            st.subheader("Newton-Raphson Iteration Table")
            # Use st.dataframe with escape=False to render LaTeX-like column headers
            st.dataframe(df, hide_index=True, use_container_width=True, height=min(len(df) * 35 + 40, 400))
            
            final_f_val = float(f.evalf(subs={x: minimizer}))
            st.success(f"Estimated minimizer: $x = {minimizer:.6f}$, $f(x) = {final_f_val:.6f}$")

            # ---- Plot Function & Steps ----
            xs = np.linspace(left, right, 400)
            
            # Use a safe evaluation method to avoid plotting issues
            ys = []
            for val in xs:
                try:
                    ys.append(float(f.evalf(subs={x: val})))
                except (TypeError, ValueError):
                    ys.append(np.nan) # Append NaN for non-plottable points
            
            plt.figure(figsize=(10,5))
            plt.plot(xs, ys, label="$f(x)$", color="blue", linewidth=2)
            
            if len(table):
                # Filter out the initial guess if the loop broke immediately
                it_xs = [row[1] for row in table]
                it_ys = [float(f.evalf(subs={x: val})) for val in it_xs]
                plt.scatter(it_xs, it_ys, color="red", s=55, label="Iterations")
                plt.plot(it_xs, it_ys, color="red", linestyle="--", alpha=0.5)
            
            # Plot the final minimizer location
            plt.scatter(minimizer, final_f_val, color="green", marker='o', s=100, label="Minimizer")
            
            plt.xlabel("$x$", fontsize=12)
            plt.ylabel("$f(x)$", fontsize=12)
            plt.title("Function and Newton-Raphson Steps", fontsize=15)
            plt.legend()
            st.subheader("Function Plot (with Iteration Steps)")
            st.pyplot(plt, use_container_width=True)

            st.info("""
            - **Iteration:** Step number
            - **$x_i$**: Current $\\text{x}$ value
            - **$f'(x_i)$**: First derivative (gradient) at step
            - **$f''(x_i)$**: Second derivative (Hessian) used for update
            - **$x_{i+1}$**: Next approximation of $\\text{x}$
            - **$|x_{i+1} - x_i|$**: Absolute error for iteration
            - **$\\epsilon$ (Tol)**: Your typed tolerance value
            - **Error < Tol?**: Displays TRUE or FALSE
            """)
        else:
            st.error("No iterations performed. Check initial guess and function domain.")

    except Exception as e:
        st.error(f"An unexpected error occurred during calculation or plotting: {e}")

st.markdown("""
**Instructions:** - Use calculator buttons for quick input, or type your function directly.
- **Implicit multiplication** is now fully supported.
- Adjust all other settings and click 'Run' for stepwise results and visualization!
""")