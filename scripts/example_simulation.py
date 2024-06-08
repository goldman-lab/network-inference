import sys
sys.path.insert(1, '../src/')
from ZData import *
import warnings

warnings.simplefilter('ignore', category=numba.core.errors.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=numba.core.errors.NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=numba.core.errors.NumbaPerformanceWarning)

def get_block_diagonal_params(N,nb,readout_frac):
    b = np.zeros(N,dtype=int)
    nint = int((1.0 - readout_frac) * N / (nb - 1))
    for i in range(nb):
        b[i*nint:(i+1)*nint] = i
    m = np.zeros((nb,nb))
    s = np.zeros((nb,nb))
    for i in range(nb-1):
        m[i,i] = 1.0/nint
        m[nb-1,i] = 1.0/nint
        s[i,i] = 0.5/np.sqrt(nint)
        s[nb-1,i] = 0.25/np.sqrt(nint)
    return b,m,s

def get_block_cyclic_params(N,nc,nr,readout_frac):
    nb = 3 * nc + nr
    b = np.zeros(N,dtype=int)
    nint = int((1.0 - readout_frac) * N / (3*nc))
    nread = N - 3*nc*nint
    for i in range(3*nc):
        b[i*nint:(i+1)*nint] = i
    for i in range(nr):
        b[3*nc*nint + i*nread:3*nc*nint + (i+1)*nread] = i
    m = np.zeros((nb,nb))
    s = np.zeros((nb,nb))
    for i in range(nc):
        m[i*3+1,i*3] = 1.0/nint
        m[i*3+2,i*3+1] = 1.0/nint
        m[i*3,i*3+2] = 1.0/nint
        s[i*3+1,i*3] = 0.5/np.sqrt(nint)
        s[i*3+2,i*3+1] = 0.5/np.sqrt(nint)
        s[i*3,i*3+2] = 0.5/np.sqrt(nint)
        for j in range(nr):
            rando = np.random.randint(3)
            m[nb-1-j,i*3:(i+1)*3] = 1.0/nint
            m[nb-1-j,i*3+rando] = 0
            s[nb-1-j,i*3:(i+1)*3] = 0.25/np.sqrt(nint)
            s[nb-1-j,i*3+rando] = 0
    return b,m,s

confile = sys.argv[1]
outfile = sys.argv[2]
data = ZData(confile,dtype='sim')
b,m,s = get_block_diagonal_params(data.N,3,0.2)
data.sim.generate_weight_matrix('block random',block_id=b,mean=m,std=s,sign_constraints=1,leading_eigenvalues=[0.995])
y,v = ZSim.sorted_eigs(data.sim.W)
while np.real(y[n-1]) < 0.9:
    data.sim.generate_weight_matrix('block random',block_id=b,mean=m,std=s,sign_constraints=1,leading_eigenvalues=[0.995])
    y,v = ZSim.sorted_eigs(data.sim.W)
data.sim.assign_cirf_taus()
data.sim.simulate(data)
data.estimate_firing_rates()
data.save(outfile)

'''
data = ZData(confile,dtype='sim')
#  block-diagonal matrices
for n in range(2,5):
    b,m,s = get_block_diagonal_params(data.N,n+1,0.2)

    for i in range(5):
        data.sim.generate_weight_matrix('block random',block_id=b,mean=m,std=s,sign_constraints=1,leading_eigenvalues=[0.995])
        y,v = ZSim.sorted_eigs(data.sim.W)
        while np.real(y[n-1]) < 0.9:
            data.sim.generate_weight_matrix('block random',block_id=b,mean=m,std=s,sign_constraints=1,leading_eigenvalues=[0.995])
            y,v = ZSim.sorted_eigs(data.sim.W)
        data.sim.assign_cirf_taus()
        data.nstims = 2
        data.ntrials = 10 * np.ones(3,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'allexc/w{n}bd_{data.N}.{i}.2stim.fitdata')

        data.nstims = 5
        data.ntrials = 10 * np.ones(6,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'allexc/w{n}bd_{data.N}.{i}.5stim.fitdata')

        data.nstims = 20
        data.ntrials = 10 * np.ones(21,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'allexc/w{n}bd_{data.N}.{i}.20stim.testdata')

#  3-cycle matrices
for n in range(1,3):
    b,m,s = get_block_cyclic_params(data.N,n,2,0.1)

    for i in range(5):
        data.sim.generate_weight_matrix('block random',block_id=b,mean=m,std=s,sign_constraints=1,leading_eigenvalues=[0.995])
        y,v = ZSim.sorted_eigs(data.sim.W)
        while np.real(y[n-1]) < 0.9:
            data.sim.generate_weight_matrix('block random',block_id=b,mean=m,std=s,sign_constraints=1,leading_eigenvalues=[0.995])
            y,v = ZSim.sorted_eigs(data.sim.W)
        data.sim.assign_cirf_taus()
        data.nstims = 2
        data.ntrials = 10 * np.ones(3,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'allexc/w{n}c_{data.N}.{i}.2stim.fitdata')

        data.nstims = 5
        data.ntrials = 10 * np.ones(6,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'allexc/w{n}c_{data.N}.{i}.5stim.fitdata')

        data.nstims = 20
        data.ntrials = 10 * np.ones(21,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'allexc/w{n}c_{data.N}.{i}.20stim.testdata')

        

#  orthogonal matrices
data = ZData(confile,dtype='sim')
le = np.array([0.995,0.98,0.97,0.95,0.9,0.8])
for n in range(1,6):
    for i in range(5):
        data.sim.generate_weight_matrix('orthogonal',leading_eigenvalues=np.delete(le,np.s_[n:5],axis=0))
        data.sim.assign_cirf_taus()
        data.nstims = 2
        data.ntrials = 10 * np.ones(3,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'free/w{n}o_{data.N}.{i}.2stim.fitdata')

        data.nstims = 5
        data.ntrials = 10 * np.ones(6,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'free/w{n}o_{data.N}.{i}.5stim.fitdata')

        data.nstims = 20
        data.ntrials = 10 * np.ones(21,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'free/w{n}o_{data.N}.{i}.20stim.testdata')

#  uniform random matrices
data = ZData(confile,dtype='sim')
le = np.array([0.995,0.98,0.97,0.95,0.9,0.8])
m = 1.0 / data.N
s = 0.5 / np.sqrt(data.N)
for n in range(1,6):
    for i in range(5):
        data.sim.generate_weight_matrix('uniform random',mean=m,std=s,sign_constraints=None,leading_eigenvalues=np.delete(le,np.s_[n:5],axis=0))
        data.sim.assign_cirf_taus()
        data.nstims = 2
        data.ntrials = 10 * np.ones(3,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'free/w{n}u_{data.N}.{i}.2stim.fitdata')

        data.nstims = 5
        data.ntrials = 10 * np.ones(6,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'free/w{n}u_{data.N}.{i}.5stim.fitdata')

        data.nstims = 20
        data.ntrials = 10 * np.ones(21,dtype=int)
        data.sim.simulate(data)
        data.estimate_firing_rates()
        data.save(f'free/w{n}u_{data.N}.{i}.20stim.testdata')
'''