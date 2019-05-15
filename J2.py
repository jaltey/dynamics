# Integrate the motion of a test particle around a central object,
# when the central object has some oblateness characterized by J2
# Planar case only for now!

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

pi=np.pi

# Set units
G=1.
au=1.
Msun=1.
yr=2*pi
Rsun=0.00465047*au

# Equations of motion with a J2
def deriv(X, t):
    
    r, v, phi = X
    phi = np.arctan2(np.sin(phi),np.cos(phi))
    vdot = h*h*np.power(r,-3) - GM*np.power(r,-2) - 1.5*K*np.power(r,-4)
    rdot = v
    phidot = h*np.power(r,-2)
    
    return [rdot,vdot,phidot]

# Central object mass, radius, and J2
Mcen=1.0*Msun
Rcen=1.0*Rsun
GM=G*Mcen
J2=1e-2

# Orbit parameters (semi-major axis, eccentricity, mean motion)
a=0.1*au
e=0.8
n=np.sqrt(GM*np.power(a,-3))

# Initial position, velocity, and angular momentum
# Start at pericenter
r = a*(1-e)
h = np.sqrt(GM*a*(1-e*e))*np.sqrt(1+0.5*J2*np.power(Rcen/a,2)*(3.+e*e)/np.power(1-e*e,2))
vr = 0.
vt = h/r
phi = 0.

# Initial conditions for integrator
X0 = np.array([r, vr, phi])

# Integration time and sampling
tf = 1*yr
N = 5000
time = np.linspace(0,tf,N)

# Integration
K = GM*Rcen*Rcen*J2
sol = odeint(deriv, X0, time)
    
r,vr,phi = sol.T[:3]
phi = np.arctan2(np.sin(phi),np.cos(phi))

# Plot
fig,ax=plt.subplots(figsize=(5,4))
ax.plot(time/yr,r, "C0-", lw=4)
ax.set_xlim(0,tf/yr)
ax.set_xlabel("time (years)")
ax.set_ylabel("r (au)")
plt.tight_layout()
plt.show()

