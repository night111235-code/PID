import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import odeint

st.set_page_config(page_title="PID Tuning Simulator", layout="wide")

# -------------------------------
# Process model (FOPDT-like)
# -------------------------------
def fopdt_process(y, t, u, taup, Kp, T0=23.0):
    # y: current PV (temperature); u: OP (manipulated), Kp: process gain, taup: time constant
    dydt = (1.0 / taup) * (-(y - T0) + Kp * u)
    return dydt

def simulate_pid(
    Kc: float,
    Ki: float,
    Kd: float,
    Kp: float,
    taup: float,
    thetap: float,
    tf: float,
    n: int,
    sp_profile: list,
    op_min: float = 0.0,
    op_max: float = 100.0,
):
    """
    Simulate PID with a time-delayed OP on a first-order-plus-dead-time process.
    sp_profile: list of (t_start, value) tuples, e.g. [(0, 23), (5, 35), (10, 40)] in same time units as tf
    """
    t = np.linspace(0.0, tf, n)
    dt = t[1] - t[0]

    # Build SP from profile
    SP = np.zeros(n)
    current_val = sp_profile[0][1]
    idx = 0
    for i, ti in enumerate(t):
        # move to the last profile entry whose start time <= ti
        while idx + 1 < len(sp_profile) and ti >= sp_profile[idx + 1][0]:
            idx += 1
            current_val = sp_profile[idx][1]
        SP[i] = current_val

    # Allocate arrays
    P = np.zeros(n)
    I = np.zeros(n)
    D = np.zeros(n)
    e = np.zeros(n)
    OP = np.zeros(n)
    PV = np.ones(n) * sp_profile[0][1]  # start near first SP
    y0 = PV[0]
    iae = 0.0

    # dead-time as integer steps
    delay_steps = max(0, int(round(thetap / dt)))

    for i in range(1, n):
        # pick delayed OP
        u_delayed = OP[i - delay_steps] if i - delay_steps >= 0 else OP[0]

        # integrate process one step
        ts = [t[i - 1], t[i]]
        y = odeint(fopdt_process, y0, ts, args=(u_delayed, taup, Kp))
        y0 = float(y[1])
        PV[i] = y0

        # metrics
        e[i] = SP[i] - PV[i]
        iae += abs(e[i])

        # PID terms
        P[i] = Kc * e[i]
        I[i] = I[i - 1] + Ki * e[i] * dt
        D[i] = Kd * (PV[i] - PV[i - 1]) / dt  # derivative on PV (measurement)

        # controller output
        OP[i] = P[i] + I[i] + D[i]

        # saturation with simple anti-windup (hold integral when saturated)
        if OP[i] > op_max:
            OP[i] = op_max
            I[i] = I[i - 1]
        if OP[i] < op_min:
            OP[i] = op_min
            I[i] = I[i - 1]

    return t, SP, PV, OP, P, I, D, e, iae


# -------------------------------
# UI
# -------------------------------
st.title("PID Tuning Simulator (FOPDT)")
st.caption("Interactive tuner for a first-order process with dead time. Adjust Kc, Ki, Kd to minimize IAE.")

with st.sidebar:
    st.header("Controller (PID)")
    Kc = st.slider("Kc (Proportional gain)", 0.0, 50.0, 12.0, 0.1)
    Ki = st.slider("Ki (Integral factor, 1/sec-like)", 0.0, 20.0, 0.10, 0.01)
    Kd = st.slider("Kd (Derivative factor, sec-like)", 0.0, 20.0, 0.0, 0.1)

    st.header("Process (FOPDT)")
    Kp = st.slider("Process gain (Kp)", 0.1, 5.0, 0.9, 0.1)
    taup = st.slider("Time constant (τp, sec)", 10.0, 1000.0, 300.0, 5.0)
    thetap = st.slider("Dead time (θ, sec)", 0.0, 200.0, 15.0, 1.0)
    T0 = st.number_input("Ambient/Base temperature (°C)", value=23.0, step=0.5)

    st.header("Simulation")
    tf = st.slider("Final time (sec)", 50.0, 5000.0, 1000.0, 10.0)
    n = st.slider("Number of points", 200, 5000, 1001, 1)

    st.header("Setpoint profile")
    sp0 = st.number_input("SP0 (from t=0)", value=23.0, step=0.5)
    step1_t = st.number_input("Step 1 time (sec)", value=50.0, step=1.0, min_value=0.0, max_value=float(tf))
    step1_val = st.number_input("SP at Step 1", value=35.0, step=0.5)
    step2_t = st.number_input("Step 2 time (sec)", value=500.0, step=1.0, min_value=0.0, max_value=float(tf))
    step2_val = st.number_input("SP at Step 2", value=40.0, step=0.5)

    st.caption("Tip: Set Ki≈1/τi and Kd≈τd if you think in time constants; here Ki and Kd are used directly like in your notebook.")

# Build SP profile like your notebook (23 → 35 → 40)
sp_profile = sorted(
    [(0.0, sp0), (step1_t, step1_val), (step2_t, step2_val)],
    key=lambda x: x[0]
)

# Run simulation
t, SP, PV, OP, P, I, D, e, iae = simulate_pid(
    Kc=Kc, Ki=Ki, Kd=Kd, Kp=Kp, taup=taup, thetap=thetap, tf=tf, n=n, sp_profile=sp_profile
)

# -------------------------------
# Plots (Matplotlib)
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
(ax1, ax2), (ax3, ax4) = axes

# PV vs SP
ax1.plot(t, SP, "k-", linewidth=2, label="Setpoint (SP)")
ax1.plot(t, PV, "r:", linewidth=2, label="Process Variable (PV)")
ax1.set_ylabel("Temperature (°C)")
ax1.set_title(f"IAE = {iae:.2f} | Kc={Kc:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")
ax1.legend(loc="best")
ax1.grid(True, alpha=0.3)

# PID terms
ax2.plot(t, P, "g.-", linewidth=1.5, label="P = Kc * e(t)")
ax2.plot(t, I, "b-", linewidth=1.5, label="I = ∫ Ki * e(t) dt")
ax2.plot(t, D, "r--", linewidth=1.5, label="D = Kd * dPV/dt")
ax2.set_title("PID Terms")
ax2.legend(loc="best")
ax2.grid(True, alpha=0.3)

# Error
ax3.plot(t, e, "m--", linewidth=1.5, label="Error e = SP - PV")
ax3.set_ylabel("ΔT (°C)")
ax3.set_xlabel("Time (s)")
ax3.legend(loc="best")
ax3.grid(True, alpha=0.3)

# OP
ax4.plot(t, OP, "b--", linewidth=1.5, label="Controller Output (OP)")
ax4.set_xlabel("Time (s)")
ax4.legend(loc="best")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# -------------------------------
# Details / footer
# -------------------------------
with st.expander("Show simulation details"):
    st.write(
        f"""
- **Model**: FOPDT with dead time θ using a delayed OP buffer  
- **Anti-windup**: integral held when OP hits limits (0–100)  
- **IAE**: Integral of |error| over the simulation  
- **Notes**: This uses derivative on PV (measurement), which is common for noise robustness.
        """
    )

st.info("Try increasing Kc and Ki gradually to reduce IAE, then add a bit of Kd to tame overshoot.")
