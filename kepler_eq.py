# Solve Kepler's Equation using a modified Newton-Raphson scheme
# Murray & Dermott (1999), section 2.4

import numpy as np
import matplotlib.pyplot as plt


# Set units
G=1.
au=1.
Msun=1.
yr=2*np.pi

def KeplerEq(M,e):

    # First make sure that the mean anomaly is in the range 0<=M<=2*pi
    while (np.abs(M) > 2.*np.pi):
        M-= 2.*np.pi*np.sign(M)
    if (M < 0.0):
        M+= 2*np.pi

    # Initial guest for E
    k=0.85
    ecc_anom = M + np.sign(np.sin(M)) * k * e

    # Start iteration
    i=0
    imax=100 # Max number of iterations
    tol=1e-10 # tolerance requirement
    
    while(i<imax):

        # fn's are n-th derivatives of f0
        f0 = ecc_anom - e*np.sin(ecc_anom) - M
        f1 = 1.0 - e*np.cos(ecc_anom)
        f2 = e*np.sin(ecc_anom)
        f3 = -e*np.cos(ecc_anom)

        delta1 = -f0/f1
        delta2 = -f0/(f1 + delta1*f2/2.)
        delta3 = -f0/(f1 + delta2*f2/2. + delta2*delta2*f3/6.)

        ecc_anom = ecc_anom + delta3

        err = ecc_anom - e*np.sin(ecc_anom) - M
        
        if (np.abs(err) < tol):
            break
        
        i+=1
    
    return ecc_anom

# Now we can plot orbital radius as a function of time
tf=5*yr
N=1000
time=np.linspace(0,tf,N)

# Assume a test particle at a=1au around the Sun, with eccentricity e
e = 0.8
a = 1*au
n = np.sqrt(G*Msun*np.power(a,-3))

# Get mean anomaly, knowing time of pericenter passage
tau = 0.
M = n*(time-tau)

# Get eccentric anomaly
ecc_anom = [KeplerEq(M[i],e) for i in range(0,N)]

# Deduce true anomaly
true_anom = 2 * np.arctan( np.sqrt((1+e)/(1-e)) * np.tan(np.divide(ecc_anom,2)) )

# Finally, deduce radius
r = a * (1 - e*e) / (1 + e*np.cos(true_anom))

# And plot
fig,ax=plt.subplots(figsize=(5,4))
ax.plot(time/yr,r, "C0-", lw=4)
ax.set_xlim(0,tf/yr)
ax.set_xlabel("time (years)")
ax.set_ylabel("r (au)")
plt.tight_layout()
plt.show()
