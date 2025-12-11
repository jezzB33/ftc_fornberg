import streamlit as st
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# PART 1: THE CORE LOGIC (Your Scripts)
# ==========================================

# --- From ftc_recast_gp_weights.py ---
def _gauss_solve(A, b):
    """Solve Ax = b with partial pivoting."""
    n = len(A)
    M = [row[:] for row in A]
    y = b[:]
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[piv][k]) < 1e-30:
            return None # Handle singularity gracefully in UI
        if piv != k:
            M[k], M[piv] = M[piv], M[k]
            y[k], y[piv] = y[piv], y[k]
        for i in range(k+1, n):
            m = M[i][k] / M[k][k]
            if m != 0.0:
                for j in range(k, n):
                    M[i][j] -= m * M[k][j]
                y[i] -= m * y[k]
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        s = y[i]
        for j in range(i+1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x

def ftc_recast_weights(nodes, x_star=None):
    """Generate weights for arbitrary nodes."""
    n = len(nodes)
    if n < 2: return []
    if x_star is None: x_star = sum(nodes) / n

    A = []
    b = []
    # Row 0: Sum of weights = 0
    A.append([1.0 for _ in nodes])
    b.append(0.0)
    # Integrated moments
    for k in range(1, n):
        row = []
        for xj in nodes:
            dx = xj - x_star
            row.append((dx**k) / k)
        A.append(row)
        b.append(1.0 if k == 1 else 0.0)

    return _gauss_solve(A, b)

# --- From ftc_fornberg_recast.py ---
def ftc_stencil(order=2):
    """Return hardcoded efficient stencils."""
    if order == 2: return [-0.5, 0.0, 0.5]
    elif order == 4: return [1/12, -2/3, 0.0, 2/3, -1/12]
    elif order == 6: return [-1/60, 3/20, -3/4, 0.0, 3/4, -3/20, 1/60]
    return []

# ==========================================
# PART 2: THE STREAMLIT UI
# ==========================================

st.set_page_config(page_title="FTC-Recast Stencil Engine", layout="wide", page_icon="⚡")

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    h1 {color: #0e1117;}
    div.stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ FTC-Recast Finite Difference Engine")
st.markdown("""
**An interactive showcase of the FTC-Recast method for numerical differentiation.**
This engine generates high-fidelity derivative stencils for arbitrary grids and executes them with high efficiency.
""")

# Layout: 3 Tabs
tab1, tab2, tab3 = st.tabs([
    "🎛️ Dynamic Stencil Generator", 
    "📉 Accuracy & Convergence", 
    "🚀 Efficiency Profiling"
])

# -----------------------------------------------------------------------------
# TAB 1: DYNAMIC GENERATOR (Flexibility Showcase)
# -----------------------------------------------------------------------------
with tab1:
    col_ctrl, col_viz = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Configuration")
        st.info("Demonstrates the solver's ability to handle **non-uniform (messy) grids**.")
        
        # Function Selection
        func_name = st.selectbox("Test Function", ["sin(x)", "exp(x)", "tanh(x)", "x^2 + 2x"])
        x_star = st.number_input("Evaluation Point (x*)", value=1.0, step=0.1)
        
        # Node Configuration
        st.write("**Node Offset Configuration** (relative to x*)")
        node_preset = st.radio("Grid Type", ["Uniform", "Random Perturbation", "Custom"], horizontal=True)
        
        if node_preset == "Uniform":
            offsets = [-1.0, 0.0, 1.0]
        elif node_preset == "Random Perturbation":
            # Add random noise to uniform grid to simulate mesh drift
            noise = np.random.uniform(-0.2, 0.2, 3)
            offsets = sorted([-1.0 + noise[0], 0.0 + noise[1], 1.0 + noise[2]])
        else:
            custom_str = st.text_input("Enter comma-separated offsets", "-1.5, -0.1, 0.5")
            try:
                offsets = sorted([float(x) for x in custom_str.split(',')])
            except:
                st.error("Invalid input")
                offsets = [-1.0, 0.0, 1.0]

        # Calculate Weights (The "Secret Sauce" Solver)
        nodes = [x_star + o for o in offsets]
        weights = ftc_recast_weights(nodes, x_star)
        
    with col_viz:
        if weights is None:
            st.error("Singular system detected. Try different nodes.")
        else:
            # Prepare Data for Plotting
            funcs = {
                "sin(x)": (np.sin, np.cos),
                "exp(x)": (np.exp, np.exp),
                "tanh(x)": (np.tanh, lambda x: 1 - np.tanh(x)**2),
                "x^2 + 2x": (lambda x: x**2 + 2*x, lambda x: 2*x + 2)
            }
            f, df = funcs[func_name]
            
            # 1. Visualization: The Function & Tangent
            x_plot = np.linspace(min(nodes)-0.5, max(nodes)+0.5, 100)
            y_plot = f(x_plot)
            
            # Calculate approx derivative
            f_nodes = [f(x) for x in nodes]
            approx_deriv = sum(w*y for w, y in zip(weights, f_nodes))
            true_deriv = df(x_star)
            
            # Tangent line: y = f(x*) + f'(x*)(x - x*)
            tangent = f(x_star) + approx_deriv * (x_plot - x_star)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot 1: Function + Tangent
            ax1.plot(x_plot, y_plot, 'k-', alpha=0.3, label='f(x)')
            ax1.plot(x_plot, tangent, 'r--', label="FTC Approximation")
            ax1.scatter(nodes, f_nodes, color='blue', s=80, zorder=5, label='Stencil Nodes')
            ax1.scatter([x_star], [f(x_star)], color='red', marker='x', s=100, zorder=6, label='x*')
            ax1.set_title(f"Approximating f'({x_star}) using {len(nodes)} arbitrary nodes")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: The Weights (Stem plot)
            markerline, stemlines, baseline = ax2.stem(nodes, weights, basefmt="black")
            plt.setp(stemlines, 'color', 'blue')
            plt.setp(markerline, 'color', 'blue')
            ax2.set_title("Generated Stencil Weights (Note asymmetry for non-uniform grids)")
            ax2.set_xlabel("x")
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Approx. Derivative", f"{approx_deriv:.6f}")
            c2.metric("True Derivative", f"{true_deriv:.6f}")
            c3.metric("Error", f"{abs(approx_deriv - true_deriv):.2e}", delta_color="inverse")

# -----------------------------------------------------------------------------
# TAB 2: ACCURACY (Scientific Verification)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Convergence Verification")
    st.markdown("""
    This test verifies that the **Order 2, 4, and 6** stencils converge at the theoretically predicted rates.
    We plot the Error vs Step Size ($h$) on a log-log scale.
    """)
    
    col_acc_1, col_acc_2 = st.columns([1, 3])
    
    with col_acc_1:
        st.write("settings")
        orders_to_test = st.multiselect("Orders to Plot", [2, 4, 6], default=[2, 4, 6])
        n_points = st.slider("Resolution", 10, 50, 20)
    
    with col_acc_2:
        # Run Convergence Test
        h_vals = np.logspace(-1, -3.5, n_points)
        results = {}
        
        # Using sin(x) at x=1.0 for test
        exact = np.cos(1.0)
        
        fig2, ax_conv = plt.subplots(figsize=(8, 5))
        
        for order in orders_to_test:
            errors = []
            coeffs = ftc_stencil(order)
            radius = len(coeffs) // 2
            
            for h in h_vals:
                # Generate uniform grid around x=1.0
                test_nodes = [1.0 + (i - radius)*h for i in range(len(coeffs))]
                f_vals = [np.sin(x) for x in test_nodes]
                
                # Apply stencil
                d_approx = sum(c * fv for c, fv in zip(coeffs, f_vals)) / h
                errors.append(abs(d_approx - exact))
            
            # Plot
            ax_conv.loglog(h_vals, errors, 'o-', label=f"Order {order}")
            
            # Calculate empirical slope (order of accuracy)
            if len(errors) > 1:
                slope = np.polyfit(np.log(h_vals), np.log(errors), 1)[0]
                results[order] = slope

        ax_conv.set_xlabel("Step Size (h)")
        ax_conv.set_ylabel("Absolute Error (|f' - approx|)")
        ax_conv.grid(True, which="both", alpha=0.3)
        ax_conv.legend()
        ax_conv.invert_xaxis() # Convention: smaller h to the right
        st.pyplot(fig2)
        
        # Display empirical slopes
        st.write("### Empirical Convergence Rates")
        cols = st.columns(len(results))
        for i, (ord_key, slope) in enumerate(results.items()):
            cols[i].metric(f"Order {ord_key} Slope", f"{slope:.2f}", help="Should be close to the Order number")

# -----------------------------------------------------------------------------
# TAB 3: PROFILING (Efficiency Showcase)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Performance Profiling")
    st.markdown("""
    This demonstrates the advantage of the **FTC-Recast architecture**:
    1. **Solver Mode:** Calculate weights dynamically (flexible, but slower).
    2. **Stencil Mode:** Apply pre-calculated weights (rigid, but extremely fast).
    """)
    
    iterations = st.slider("Number of Derivative Calculations", 1000, 100000, 50000)
    
    if st.button("🚀 Run Benchmark"):
        # 1. Benchmark Solver (The "Expensive" way)
        # We simulate recalculating weights for a changing grid 
        start_t = time.time()
        for _ in range(iterations):
            # Trivial 3-point solve
            _ = ftc_recast_weights([-0.1, 0.0, 0.1], x_star=0.0)
        dur_solve = time.time() - start_t
        
        # 2. Benchmark Hardcoded (The "Optimized" way)
        start_t = time.time()
        coeffs = ftc_stencil(2) # Pre-fetched
        sample_vals = [0.5, 0.6, 0.7] # Dummy data
        for _ in range(iterations):
            # Pure arithmetic application
            val = coeffs[0]*sample_vals[0] + coeffs[1]*sample_vals[1] + coeffs[2]*sample_vals[2]
        dur_apply = time.time() - start_t
        
        # Visualization
        st.write("---")
        c1, c2, c3 = st.columns(3)
        
        c1.metric("Dynamic Solver Time", f"{dur_solve:.4f} s")
        c2.metric("Pre-calc Stencil Time", f"{dur_apply:.4f} s")
        
        speedup = dur_solve / dur_apply if dur_apply > 0 else 0
        c3.metric("Speedup Factor", f"{speedup:.1f}x", delta="Faster", delta_color="normal")
        
        # Bar Chart
        perf_data = pd.DataFrame({
            "Method": ["Dynamic Solver", "Pre-calc Stencil"],
            "Time (s)": [dur_solve, dur_apply]
        })
        st.bar_chart(perf_data, x="Method", y="Time (s)")
