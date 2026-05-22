"""
Half-car (bounce + pitch) suspension model.

Board model
-----------
Two suspension corners, front 'a' and rear 'b', each a spring k and damper c.
The sprung body is a rigid beam of length (wheelbase) L, mass m, pitch inertia I.

Corner kinematics (small-angle, from the board):
    a = x - (L/2) * theta      (front attachment point)
    b = x + (L/2) * theta      (rear attachment point)
    x     = (a + b) / 2        (heave / bounce)
    theta = (b - a) / L        (pitch, sin(theta) ~ theta)

Road input f(t): front wheel sees f(t), rear wheel sees f(t - tau)
with delay tau = L / V (wheelbase / vehicle speed). Here f(t) is a
linear sine sweep (chirp) so we can read off the resonances.

Equations of motion are M q'' + C q' + K q = F(t), q = [x, theta].
We derive M, C, K symbolically with sympy, then integrate with solve_ivp.
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. SYMBOLIC DERIVATION OF THE EQUATIONS OF MOTION
# ----------------------------------------------------------------------
t = sp.symbols('t', real=True)
m, I, k, c, L = sp.symbols('m I k c L', positive=True)
x, th = sp.symbols('x theta', cls=sp.Function)
fa, fb = sp.symbols('f_a f_b')          # front / rear road inputs

x_t, th_t = x(t), th(t)

# corner displacements & velocities in terms of body DOFs
a   = x_t - (L/2)*th_t
b   = x_t + (L/2)*th_t
ad  = sp.diff(a, t)
bd  = sp.diff(b, t)

# Suspension force at each corner (spring + damper), road acts through spring & damper.
# Force pushing body UP at a corner = k*(road - corner) + c*(roaddot - cornerdot).
# For the sweep we drive through the spring term k*f (matches the board's k f(t) RHS),
# damper referenced to ground motion of the corner.
Fa = k*(fa - a) + c*(0 - ad)
Fb = k*(fb - b) + c*(0 - bd)

# Newton (heave) and Euler (pitch about CG). Front corner at -L/2, rear at +L/2.
eq_heave = sp.Eq(m*sp.diff(x_t, t, 2), Fa + Fb)
eq_pitch = sp.Eq(I*sp.diff(th_t, t, 2), (-L/2)*Fa + (L/2)*Fb)

# Assemble M, C, K, and the forcing vector
q   = sp.Matrix([x_t, th_t])
qd  = q.diff(t)
qdd = q.diff(t, 2)

lhs = sp.Matrix([eq_heave.lhs - eq_heave.rhs,
                 eq_pitch.lhs - eq_pitch.rhs])
lhs = sp.expand(lhs)

M = sp.zeros(2); C = sp.zeros(2); K = sp.zeros(2)
for i in range(2):
    for j in range(2):
        M[i, j] = lhs[i].coeff(qdd[j])
        C[i, j] = lhs[i].coeff(qd[j])
        K[i, j] = lhs[i].coeff(q[j])

# forcing (collect the road-input terms). lhs = M q'' + C q' + K q - F = 0,
# so the road terms in lhs are (-F); negate to recover F on the RHS.
F = -sp.Matrix([lhs[i].coeff(fa)*fa + lhs[i].coeff(fb)*fb for i in range(2)])

print("M =\n", sp.pretty(M))
print("\nC =\n", sp.pretty(C))
print("\nK =\n", sp.pretty(K))
print("\nF (road forcing) =\n", sp.pretty(F))

# ----------------------------------------------------------------------
# 2. NUMERICAL PARAMETERS  (representative passenger car, half-car)
# ----------------------------------------------------------------------
params = {
    m: 730.0,      # sprung mass of half car [kg]
    I: 820.0,      # pitch inertia [kg m^2] (gives distinct pitch mode)
    k: 30000.0,    # suspension stiffness per corner [N/m]
    c: 2500.0,     # suspension damping per corner [N s/m]
    L: 2.6,        # wheelbase [m]
}
Mn = np.array(M.subs(params)).astype(float)
Cn = np.array(C.subs(params)).astype(float)
Kn = np.array(K.subs(params)).astype(float)
Ln = float(L.subs(params))
Minv = np.linalg.inv(Mn)

# undamped natural frequencies (eigenvalues of M^-1 K)
w2 = np.linalg.eigvals(Minv @ Kn)
wn = np.sqrt(np.sort(w2.real))
print("\nNatural frequencies [Hz]:", wn / (2*np.pi))

# ----------------------------------------------------------------------
# 3. ROAD INPUT: linear sine sweep (chirp), rear delayed by tau = L / V
# ----------------------------------------------------------------------
V      = 10.0                       # vehicle speed [m/s]
tau    = Ln / V                     # front-to-rear time delay [s]
amp    = 0.02                       # 2 cm road amplitude
f0, f1 = 0.2, 6.0                   # sweep from 0.2 to 6 Hz
Tend   = 40.0

def chirp(tt):
    # instantaneous phase of a linear sweep
    rate = (f1 - f0) / Tend
    ph = 2*np.pi*(f0*tt + 0.5*rate*tt**2)
    return amp * np.sin(ph) * (tt > 0)

def road_front(tt): return chirp(tt)
def road_rear(tt):  return chirp(tt - tau)

# forcing as a fast numeric function F(fa, fb)
F_func = sp.lambdify((fa, fb, *params.keys()), F.subs(params), 'numpy')

def Fvec(tt):
    return np.array(F_func(road_front(tt), road_rear(tt),
                           *[params[s] for s in params]), float).flatten()

# ----------------------------------------------------------------------
# 4. INTEGRATE:  state z = [x, theta, xdot, thetadot]
# ----------------------------------------------------------------------
def rhs(tt, z):
    qv  = z[:2]
    qdv = z[2:]
    acc = Minv @ (Fvec(tt) - Cn @ qdv - Kn @ qv)
    return np.concatenate([qdv, acc])

tspan = (0, Tend)
teval = np.linspace(0, Tend, 8000)
sol = solve_ivp(rhs, tspan, [0, 0, 0, 0], t_eval=teval,
                method='RK45', rtol=1e-8, atol=1e-10, max_step=0.01)

x_sol, th_sol = sol.y[0], sol.y[1]
a_sol = x_sol - (Ln/2)*th_sol
b_sol = x_sol + (Ln/2)*th_sol

# ----------------------------------------------------------------------
# 5. ANALYTIC FREQUENCY RESPONSE (FRF) magnitude vs frequency
# ----------------------------------------------------------------------
freqs = np.linspace(0.1, 6.0, 1200)
mag_x = np.zeros_like(freqs)
mag_th = np.zeros_like(freqs)
# RHS gain: with both wheels in phase, F per unit road = K @ [1,...] structure.
# Build the road->force map B so that F = B * f_road (front=rear=1 for FRF).
B = np.array([[float(F[0].coeff(fa).subs(params)), float(F[0].coeff(fb).subs(params))],
              [float(F[1].coeff(fa).subs(params)), float(F[1].coeff(fb).subs(params))]])
for i, fr in enumerate(freqs):
    w = 2*np.pi*fr
    Z = -w**2*Mn + 1j*w*Cn + Kn                       # dynamic stiffness
    H_in  = np.linalg.solve(Z, B @ np.array([1.0,  1.0]))  # in-phase  -> bounce
    H_anti = np.linalg.solve(Z, B @ np.array([-1.0, 1.0])) # anti-phase -> pitch
    mag_x[i]  = abs(H_in[0])
    mag_th[i] = abs(H_anti[1])

# ----------------------------------------------------------------------
# 6. PLOTS
# ----------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 8))

ax[0, 0].plot(sol.t, x_sol*1000, lw=1)
ax[0, 0].set_title("Body bounce  x(t)")
ax[0, 0].set_xlabel("time [s]"); ax[0, 0].set_ylabel("heave [mm]")

ax[0, 1].plot(sol.t, np.degrees(th_sol), color='tab:orange', lw=1)
ax[0, 1].set_title("Body pitch  theta(t)")
ax[0, 1].set_xlabel("time [s]"); ax[0, 1].set_ylabel("pitch [deg]")

ax[1, 0].plot(sol.t, a_sol*1000, label="front a", lw=1)
ax[1, 0].plot(sol.t, b_sol*1000, label="rear b", lw=1, alpha=0.8)
ax[1, 0].set_title("Corner displacements")
ax[1, 0].set_xlabel("time [s]"); ax[1, 0].set_ylabel("disp [mm]"); ax[1, 0].legend()

ax[1, 1].plot(freqs, mag_x/np.max(mag_x), label="bounce |X/F|")
ax[1, 1].plot(freqs, mag_th/np.max(mag_th), label="pitch |Θ/F|")
for f_hz in wn/(2*np.pi):
    ax[1, 1].axvline(f_hz, ls='--', color='gray', alpha=0.6)
ax[1, 1].set_title("Frequency response (resonances dashed)")
ax[1, 1].set_xlabel("frequency [Hz]"); ax[1, 1].set_ylabel("normalized magnitude")
ax[1, 1].legend()

plt.tight_layout()
plt.savefig("half_car_results.png", dpi=130)
print("\nSaved figure -> half_car_results.png")