# ftc_feature_scanner.py
# Automated Input Stream & Feature Range Identification
# -----------------------------------------------------

import math
import random
import time
import statistics

# --- CORE LOGIC (Imported from your previous environment) ---

def _gauss_solve(A, b):
    """Solve Ax = b with partial pivoting."""
    n = len(A)
    M = [row[:] for row in A]
    y = b[:]
    try:
        for k in range(n):
            piv = max(range(k, n), key=lambda i: abs(M[i][k]))
            if abs(M[piv][k]) < 1e-15: # slightly tighter check for stress testing
                raise ValueError("Singular")
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
    except ValueError:
        return None # Signal failure

def ftc_recast_weights(nodes, x_star):
    n = len(nodes)
    if n < 2: return None
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
        A.append(1.0 if k == 1 else 0.0)
    return _gauss_solve(A, b)

# --- SCANNER ENGINE ---

def run_input_stream():
    """
    Cycles through parameter ranges to identify the operational envelope.
    """
    print(f"{'='*60}")
    print(f"FTC-RECAST FEATURE RANGE SCANNER")
    print(f"{'='*60}")
    print(f"{'TEST SCENARIO':<25} | {'STATUS':<10} | {'MAX ERR':<10} | {'STABILITY (Max W)':<18}")
    print("-" * 65)

    # Setup true function for validation
    f = math.sin
    df = math.cos
    x_center = 1.0 # Evaluation point

    # ---------------------------------------------------------
    # STREAM 1: Grid Jitter (Testing Robustness to Non-Uniformity)
    # ---------------------------------------------------------
    # We take a uniform grid and add increasing amounts of random noise.
    # Feature Identified: Ability to handle "messy" real-world data.
    
    jitter_levels = [0.0, 0.1, 0.5, 0.9, 0.99] # % of step size h
    h = 0.1
    base_offsets = [-1, 0, 1] # 3-point stencil

    for jit in jitter_levels:
        errors = []
        max_weights = []
        failures = 0
        
        # Run 100 monte carlo simulations per level
        for _ in range(100):
            # Construct perturbed nodes
            nodes = []
            for i in base_offsets:
                # noise in range [-jit*h/2, +jit*h/2]
                noise = random.uniform(-jit*h*0.49, jit*h*0.49) 
                nodes.append(x_center + (i*h) + noise)
            
            # Sort nodes (solver doesn't strictly require it, but good practice)
            nodes.sort()
            
            # Solve
            w = ftc_recast_weights(nodes, x_center)
            if w is None:
                failures += 1
                continue
            
            # Validation
            approx = sum(wi * f(xi) for wi, xi in zip(w, nodes))
            err = abs(approx - df(x_center))
            
            errors.append(err)
            max_weights.append(max(abs(wi) for wi in w))

        # Analyze Stream Results
        if failures < 100:
            avg_err = statistics.mean(errors)
            avg_max_w = statistics.mean(max_weights)
            status = "PASS" if failures == 0 else f"UNSTABLE ({failures}%)"
            print(f"Jitter {int(jit*100)}% {' ':<14} | {status:<10} | {avg_err:.2e}   | {avg_max_w:.2f}")
        else:
            print(f"Jitter {int(jit*100)}% {' ':<14} | FAIL       | N/A        | N/A")

    print("-" * 65)

    # ---------------------------------------------------------
    # STREAM 2: Skew Tolerance (Testing Evaluation Point Location)
    # ---------------------------------------------------------
    # We move x* from the center of the stencil to the far outside.
    # Feature Identified: Transitions from Centered -> Forward/Backward -> Extrapolation.
    
    # Stencil fixed at [0, 0.1, 0.2] (relative to 0)
    # We move evaluation point x* from 0.1 (center) to 0.5 (way outside)
    
    grid_nodes = [0.0, 0.1, 0.2]
    eval_points = [0.1, 0.0, -0.1, -0.5, -1.0] 
    labels = ["Centered", "Edge (Fwd)", "Outside", "Far Out", "Extreme"]
    
    for label, x_eval in zip(labels, eval_points):
        # Shift absolute nodes so x_eval is relative
        # Actually easier: fix nodes, change x_star
        nodes_abs = [x + 1.0 for x in grid_nodes] # Grid at [1.0, 1.1, 1.2]
        x_target = 1.0 + x_eval # Target relative to grid start
        
        w = ftc_recast_weights(nodes_abs, x_target)
        
        if w:
            approx = sum(wi * f(xi) for wi, xi in zip(w, nodes_abs))
            err = abs(approx - df(x_target))
            max_w = max(abs(wi) for wi in w)
            print(f"Skew: {label:<16} | PASS       | {err:.2e}   | {max_w:.2f}")
        else:
            print(f"Skew: {label:<16} | FAIL       | -          | -")

    print("-" * 65)

    # ---------------------------------------------------------
    # STREAM 3: Node Collapse (Singularity Proximity)
    # ---------------------------------------------------------
    # What happens when two nodes get dangerously close?
    # Feature Identified: Numerical stability limit.
    
    separations = [1e-1, 1e-4, 1e-8, 1e-12, 1e-15]
    
    for sep in separations:
        # Nodes: [-h, 0, epsilon] -> 0 and epsilon are colliding
        nodes = [x_center - 0.1, x_center, x_center + sep]
        
        w = ftc_recast_weights(nodes, x_center)
        
        if w:
            # Check if weights explode
            max_w = max(abs(wi) for wi in w)
            stability = "STABLE" if max_w < 1e6 else "EXPLODING"
            print(f"Collapse Sep {sep:.0e} {' ':<4} | {stability:<10} | -          | {max_w:.2e}")
        else:
            print(f"Collapse Sep {sep:.0e} {' ':<4} | SINGULAR   | -          | -")

    print(f"{'='*60}")

if __name__ == "__main__":
    run_input_stream()
