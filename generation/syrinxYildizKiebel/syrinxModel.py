import numpy as np
import matplotlib.pyplot as plt

# ----- PART 1: Setup -----

# constants
kappa1 = 6.0/5.0
kappa2 = 1.0
kappa3 = 1.0/5.0
lambd = 1.0/8.0
N = 8
n = 5
f1 = 0.06
f2 = 2*0.06
f3 = 3*0.06
f4 = 4*0.06
f5 = 5*0.06
p0 = 4700
p1 = 7000
k0 = 7.6e8
k1 = 7e8
b = 1000
c = 1e8

# scaling functions
S = lambda x: 1.0/(1+np.exp(-x))
G = lambda x: np.array([np.exp(x_i)/np.sum(np.exp(x)) for x_i in x])

# connectivity matrix for heteroclinic orbit
rho = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i == j:
            rho[i,j] = 0.0
        elif j == i+1:
            rho[i,j] = 1.5
        elif j == i-1:
            rho[i,j] = 0.5
        else:
            rho[i,j] = 1.0
# this additional stuff was needed, although not obvious from the paper
rho[7,0] = 1.5
rho[0,7] = 0.5

# noise sampling functions
w_1 = lambda n: np.random.multivariate_normal(np.zeros(n),np.diag(n*[np.exp(-12)]))
w_2 = lambda n: np.random.multivariate_normal(np.zeros(n),np.diag(n*[np.exp(-16)]))
w_3 = lambda n: np.random.multivariate_normal(np.zeros(n),np.diag(n*[np.exp(-16)]))
w_4 = lambda n: np.random.multivariate_normal(np.zeros(n),np.diag(n*[np.exp(-16)]))

# connectivity matrix for hopfield network
W = np.array([  [-0.7634, -0.6390, -0.2995,  0.9744,  0.5732],
                [ 0.5654, -0.0437,  0.4338, -0.8218, -0.5850],
                [ 0.5896, -0.0325, -0.6591,  0.7162,  0.9779],
                [-0.8933,  0.5817, -0.6501, -0.2264, -0.1209],
                [-0.9154, -0.2407, -0.4713, -0.4636, -0.5809]])
# # second version
# W = np.array([    [-0.9325,  0.8345, -0.8415, -0.5558,  0.2149],
#               [ 0.8703, -0.8299, -0.8011,  0.7111,  0.3601],
#               [ 0.8152,  0.6341, -0.8081,  0.9685,  0.7585],
#               [-0.2917, -0.1723, -0.2294, -0.1480, -0.7595],
#               [-0.4141, -0.5107, -0.9282,  0.9574, -0.5053]])

# activition function for hopfield network
phi = lambda x: np.tanh(x)

# strange magic number for inverse engineering of the hopfield input
a = 0.2

# matrix for change rate of RA ensembles
A = np.diag([a]*n)

# strange constant for input function, in S1 just called "c"
c_I = a/np.tanh(1)

# attractors for the hopfield network, showing which RA ensembles should be active (1) or inactive (-1) for all HVC ensembles
x_star = [
    np.array([-1, 1,-1, 1,-1]),
    np.array([ 1,-1, 1,-1, 1]),
    np.array([ 1,-1,-1, 1,-1]),
    np.array([-1, 1,-1, 1,-1]),
    np.array([ 1, 1, 1,-1,-1]),
    np.array([-1, 1, 1, 1,-1]),
    np.array([-1,-1,-1,-1, 1]),
    np.array([ 1, 1, 1, 1, 1])
]

# inverse engineered input function for HVC ensemble k, such that the hopfield network will converge to the specific binary vector
I_k = lambda k: c_I * phi(x_star[k]) - W @ phi(x_star[k])

# combined input function over all HVC ensembles
I = lambda v: sum([v[k]*I_k(k) for k in range(N)])






# ----- PART 2: Simulation -----

### init euler loop ###

timespan = 500
stepsize = 0.1
sampleLength = timespan/stepsize
t_coll = np.zeros(sampleLength)

### init HVC ###

# start values were chosen from the plots in the paper
x3 = np.zeros(N)-8
x3[0] = 0

v3 = np.zeros(N)

x3_coll = np.zeros((sampleLength,N))
v3_coll = np.zeros((sampleLength,N))

### init RA ###

x2 = np.zeros(n)
v2 = np.zeros(n)
w2 = np.zeros(n)

x2_coll = np.zeros((sampleLength,n))
v2_coll = np.zeros((sampleLength,n))
w2_coll = np.zeros((sampleLength,n))

### actual loop ###

print("Eulering around...")
for i, t in enumerate(np.arange(0,timespan,stepsize)):

    # collect times and plot progress

    t_coll[i] = t
    percent = i*100.0/(sampleLength)
    if percent%5==0:
        print("{}%".format(percent))

    # HVC equations

    dx3 = kappa3*(-lambd*x3 - rho @ S(x3) + 1) + w_1(N)
    x3 += stepsize*dx3
    x3_coll[i,:] = x3

    v3 = G(x3) + w_2(N)
    v3_coll[i,:] = v3

    # RA equations

    dx2 = kappa2 * (-A @ x2 + W @ phi(x2) + I(v3)) + w_1(n)
    x2 += stepsize*dx2
    x2_coll[i, :] = x2

    v2 = 0.5*x2 + 0.5 + w_2(n)
    v2_coll[i, :] = v2

    w2 = G(x2) + w_3(n)
    w2_coll[i, :] = w2


# ----- PART 3: Plotting -----

# helper function for plotting vertical lines corresponding to the specific HVC activation sequence intervals
def axlines():
    for i in range(N-1):
        # numbers were read from the plot
        plt.axvline(72+i*65, color='red')

# plt HVC stuff

plt.figure("HVC - x3")
for i in range(N):
  plt.plot(t_coll, x3_coll[:,i])

plt.figure("HVC - v3")
for i in range(N):
  plt.plot(t_coll, v3_coll[:,i])

# plot RA stuff

plt.figure("RA - x2")
for i in range(n):
    plt.subplot(n,1,i+1)
    plt.plot(t_coll, x2_coll[:,i])
    axlines()

plt.figure("RA - v2 and w2")
for i in range(n):
    plt.subplot(n,1,i+1)
    plt.plot(t_coll, v2_coll[:,i])
    plt.plot(t_coll, w2_coll[:,i])
    axlines()


plt.show()


# ----- PART 4: Remarks/Tests/Other -----


### 1) ###

# Strange way of making RA-stage work without differential equations...
# substitute I(v) above with the following:

I_k = [
  np.array([-1.0,1.0,-1.0,1.0,-1.0]),
  np.array([1.0,-1.0,1.0,-1.0,1.0]),
  np.array([1.0,-1.0,-1.0,1.0,-1.0]),
  np.array([-1.0,1.0,-1.0,1.0,-1.0]),
  np.array([1.0,1.0,1.0,-1.0,-1.0]),
  np.array([-1.0,1.0,1.0,1.0,-1.0]),
  np.array([-1.0,-1.0,-1.0,-1.0,1.0]),
  np.array([1.0,1.0,1.0,1.0,1.0])
]
I = lambda v: sum([v[k]*I_k[k] for k in range(N)])

# then later in the euler loop adjust the RA update for x2 to:
x2 = I(v3)






