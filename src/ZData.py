import pickle
import numpy as np
import math
import scipy.optimize
import scipy.linalg
import scipy.stats
import copy
import time
import random
import numba

class ZData():
    def __init__(self,confile='',dtype='data'):
        self.N = 25
        self.dt_samp = 0.05
        self.dt_int = 0.05
        self.nstims = 0
        self.ntrials = [1]
        self.sac_period = [1.8,2]
        self.stim_period = [2.5,3]
        self.measurements = None
        self.measurement_times = None
        self.firing_rate_estimate = None
        self.t_int = None
        self.average_trials = True
        self.rate_estimation_method='2dpen'
        self.cirf_tau_estimate = 1.5
        self.multiexp_sac_taus = []
        self.multiexp_stim_taus = []
        self.multiexp_coef_pen = 0
        self.twodpen_fixation = 10
        self.twodpen_input = 0.5
        self.r2_method = 'avg'
        self.verbosity = 1
        self.conserve_storage=True
        self.base_directory=''
        self.sim = None
        if dtype == 'sim':
            self.sim = ZSim()
        self.configure(confile)
        self.centroid = np.zeros((2,self.N))
        return

    def verbalize(self,message,frivolity):
        if self.verbosity > frivolity:
            print(message)
        return

    def configure(self,confile):
        sim_commands = []
        try:
            phil = open(confile,'r')
            commands = phil.read().split('\n')
            sim_commands = []
            for command in commands:
                command = command.split('#')[0] #discard comments
                command = command.replace(' ','')
                #print(command)
                command = command.split('=')
                command[0] = command[0].lower()
                if command[0] == 'n':
                    self.N = int(command[1])
                elif command[0] == 'sampledt':
                    self.dt_samp = float(command[1])
                elif command[0] == 'interpolateddt':
                    self.dt_int = float(command[1])
                elif command[0] == 'nstims':
                    self.nstims = int(command[1])
                elif command[0] == 'ntrials':
                    #command[1] = command[1].replace('[','').replace(']','')
                    command[1] = command[1].translate({ord(c): None for c in '[]()'})
                    command[1] = command[1].translate({ord(c): ord(',') for c in ':;|/'})
                    counts = command[1].split(',')
                    self.ntrials = np.array([int(count) for count in counts],dtype=int)
                elif command[0] == 'saccadeperiod':
                    command[1] = command[1].translate({ord(c): None for c in '[]()'})
                    command[1] = command[1].translate({ord(c): ord(',') for c in ':;|/'})
                    times = command[1].split(',')
                    self.sac_period = [float(times[0]),float(times[1])]
                elif command[0] == 'stimperiod':
                    command[1] = command[1].translate({ord(c): None for c in '[]()'})
                    command[1] = command[1].translate({ord(c): ord(',') for c in ':;|/'})
                    times = command[1].split(',')
                    self.stim_period = [float(times[0]),float(times[1])]
                elif command[0] == 'rateestimation':
                    self.rate_estimation_method = command[1].lower()
                elif command[0] == 'r2method':
                    self.r2_method = command[1].lower()
                elif command[0] == 'cirftaudeconv':
                    self.cirf_tau_estimate = float(command[1])
                elif command[0] == 'averagetrials':
                    self.average_trials = (command[1].lower() == 'true')
                elif command[0] == 'multiexpsactaus':
                    command[1] = command[1].translate({ord(c): None for c in '[]()'})
                    command[1] = command[1].translate({ord(c): ord(',') for c in ':;|/'})
                    taus = command[1].split(',')
                    self.multiexp_sac_taus = [float(tau) for tau in taus]
                elif command[0] == 'multiexpstimtaus':
                    command[1] = command[1].translate({ord(c): None for c in '[]()'})
                    command[1] = command[1].translate({ord(c): ord(',') for c in ':;|/'})
                    taus = command[1].split(',')
                    self.multiexp_stim_taus = [float(tau) for tau in taus]
                elif command[0] == 'multiexpcoefpen':
                    self.multiexp_coef_pen = float(command[1])
                elif command[0] == '2dpenfixation':
                    self.twodpen_fixation = float(command[1])
                elif command[0] == '2dpeninput':
                    self.twodpen_input = float(command[1])
                elif command[0] == 'verbosity':
                    self.verbosity = int(command[1])
                elif command[0] == 'basedir':
                    self.base_directory = command[1]
                    if '/' in self.base_directory and not self.base_directory[-1] == '/':
                        self.base_directory += '/'
                    elif '\\' in self.base_directory and not self.base_directory[-1] == '\\':
                        self.base_directory += '\\'

                elif command[0] in ['simtau','simdt','simw','simduration','saccadesize','stimsize','neuronalnoise','measurementnoise',
                                    'measurement','synapticnonlinearity','ratenonlinearity','stimsignconstraint']:
                    sim_commands.append(command)
        except FileNotFoundError:
            print('No or invalid configuration file provided. Using defaults...')
        if not self.sim == None:
            self.sim.N = self.N
            self.sim.verbosity = self.verbosity
            self.sim.configure(sim_commands)
            if self.sim.measured_quantity == 'rate':
                self.rate_estimation_method = 'none'
        return
    
    def save(self,outname):
        if self.conserve_storage:
            self.measurements = None
            self.measurement_times = None
        f = open(f'{self.base_directory}{outname}','wb')
        pickle.dump(self,f)
        f.close()

    @staticmethod
    def load(infile):
        f = open(infile,'rb')
        data = pickle.load(f)
        f.close()
        return data

    def estimate_firing_rates(self):
        pt1 = time.monotonic()

        duration = self.dt_samp * (np.shape(self.measurement_times)[0] - 1)
        self.T = int(duration / self.dt_int) + 1
        self.t_int = np.linspace(0,duration,self.T)
        prev_trials = np.zeros(self.nstims + 1,dtype=int)
        for i in range(self.nstims):
            prev_trials[i+1] = prev_trials[i] + self.ntrials[i]
        tot_trials = np.sum(self.ntrials)
        self.firing_rate_estimate = np.zeros((self.T,self.N,tot_trials))
        for i in range(tot_trials):
            for j in range(self.N):
                    self.firing_rate_estimate[:,j,i] = np.interp(self.t_int,self.measurement_times[:,i],self.measurements[:,j,i])
        if self.average_trials:
            avgs = []
            for i in range(self.nstims+1):
                avgs.append(np.mean(self.firing_rate_estimate[:,:,prev_trials[i]:prev_trials[i]+self.ntrials[i]],axis=2)[None,:,:])
            self.firing_rate_estimate = np.concatenate(avgs,axis=0)
        if self.rate_estimation_method == 'multiexp':
            self.estimate_multiexp()
        elif self.rate_estimation_method == '2dpen':
            self.estimate_2dpen()

        pt2 = time.monotonic()
        self.verbalize(f'Time to estimate rates: {pt2-pt1} seconds',9)
        return

    def estimate_multiexp(self):
        I_sac = np.zeros(self.T)
        t1 = np.argmin(abs(self.t_int - self.sac_period[0]))
        if self.t_int[t1] - self.sac_period[0] > 0:
            t1 -= 1
        t2 = np.argmin(abs(self.t_int - self.sac_period[1]))
        if self.t_int[t2] - self.sac_period[1] < 0:
            t2 += 1
        I_sac[t1:t2] = 1

        I_stim = np.zeros(self.T)
        t1 = np.argmin(abs(self.t_int - self.stim_period[0]))
        if self.t_int[t1] - self.stim_period[0] > 0:
            t1 -= 1
        t2 = np.argmin(abs(self.t_int - self.stim_period[1]))
        if self.t_int[t2] - self.stim_period[1] < 0:
            t2 += 1
        I_stim[t1:t2] = 1

        t = np.linspace(0,(self.T - 1)*self.dt_int,self.T)
        cirf = self.dt_int * np.exp(-t/self.cirf_tau_estimate)
        nexp_sac = len(self.multiexp_sac_taus)
        sac_exps = []
        for i in range(nexp_sac):
            sac_exps.append(self.dt_int * np.exp(-t/self.multiexp_sac_taus[i]))
        nexp_stim = len(self.multiexp_stim_taus)
        stim_exps = []
        for i in range(nexp_stim):
            stim_exps.append(self.dt_int * np.exp(-t/self.multiexp_stim_taus[i]))
        
        rout = np.zeros(((self.nstims + 1) * self.T,nexp_sac + self.nstims * nexp_stim))
        fout = np.zeros(((self.nstims + 1) * self.T + 1,nexp_sac + self.nstims * nexp_stim))
        for i in range(nexp_sac):
            a = np.convolve(I_sac,sac_exps[i])
            for j in range(self.nstims + 1):
                rout[j*self.T:(j+1)*self.T,i] = a[0:self.T]
            a = np.convolve(a,cirf)
            for j in range(self.nstims + 1):
                fout[j*self.T:(j+1)*self.T,i] = a[0:self.T]
        for i in range(nexp_stim):
            a = np.convolve(I_stim,stim_exps[i])
            for j in range(1,self.nstims + 1):
                rout[j*self.T:(j+1)*self.T,nexp_sac + (j-1)*nexp_stim + i] = a[0:self.T]
            a = np.convolve(a,cirf)
            for j in range(1,self.nstims + 1):
                fout[j*self.T:(j+1)*self.T,nexp_sac + (j-1)*nexp_stim + i] = a[0:self.T]
        fout[-1,:] = self.multiexp_coef_pen
        
        sta = np.zeros(((self.nstims + 1) * self.T + 1,self.N))
        for j in range(self.nstims + 1):
            sta[j*self.T:(j+1)*self.T,:] = self.firing_rate_estimate[j,:,:]
        
        coefs = np.zeros((nexp_sac + self.nstims * nexp_stim,self.N))
        lb = np.zeros(nexp_sac + self.nstims * nexp_stim)
        lb[nexp_sac:] = -999 * np.ones(self.nstims * nexp_stim)
        ub = 999 * np.ones(nexp_sac + self.nstims * nexp_stim)
        for n in range(self.N):
            res = scipy.optimize.lsq_linear(fout,sta[:,n],bounds=(lb,ub),method='trf')
            coefs[:,n] = res.x[:]
        #coefs = np.linalg.pinv(fout) @ sta
        fit_rates = np.dot(rout,coefs)
        #fit_fluor = np.dot(fout,coefs)
        for j in range(self.nstims + 1):
            #fit_rates[j*self.T:(j+1)*self.T-1,:] = ((1.9/0.05) * (fit_fluor[1:self.T,:] - fit_fluor[0:160,:]) + fit_fluor[0:160,:])
            self.firing_rate_estimate[j,:,:] = fit_rates[j*self.T:(j+1)*self.T,:]
        return

    def estimate_2dpen(self):
        cirf = np.zeros((self.T,self.T))
        t_back = np.linspace((self.T-1)*self.dt_int,0,self.T)
        exptb = self.dt_int * np.exp(-t_back/self.cirf_tau_estimate)
        for j in range(self.T):
            cirf[j,0:j+1] = exptb[-j-1:]
        pen_2d = np.zeros((self.T,self.T))
        vec_2d = np.ones(3)
        vec_2d[1] = -2
        t1 = np.argmin(abs(self.t_int - self.sac_period[0]))
        if self.t_int[t1] - self.sac_period[0] > 0:
            t1 -= 1
        t2 = np.argmin(abs(self.t_int - self.stim_period[1]))
        if self.t_int[t2] - self.stim_period[1] < 0:
            t2 += 1
        for i in range(t1):
            pen_2d[i,i:i+3] = self.twodpen_fixation * vec_2d
        for i in range(t1,t2):
            pen_2d[i,i:i+3] = self.twodpen_input * vec_2d
        for i in range(t2,self.T-2):
            pen_2d[i,i:i+3] = self.twodpen_fixation * vec_2d
        A = np.concatenate([cirf,pen_2d],axis=0)
        z = np.zeros(2*self.T)
        #lb = np.zeros(self.T)
        lb = -99 * np.ones(self.T)
        ub = 99 * np.ones(self.T)
        for i in range(self.nstims+1):
            for j in range(self.N):
                z[0:self.T] = self.firing_rate_estimate[i,:,j]
                res = scipy.optimize.lsq_linear(A,z,bounds=(lb,ub),method='trf',tol=1e-10)
                self.firing_rate_estimate[i,:,j] = res.x[:]
        return

    def get_r2(self,t_min=0):
        it = int(t_min/self.dt_int)
        X_est = self.firing_rate_estimate[:,it:,:]
        X = self.sim.simulated_rates[:,it:,:]
        X_mean = np.mean(X,axis=1)
        if self.r2_method == 'avg':
            r2 = 1 - np.sum((X - X_est)**2,axis=(0,1))/np.sum((X - X_mean[:,None,:])**2,axis=(0,1))
            return np.mean(r2)
        return 1 - np.sum((X - X_est)**2)/np.sum((X - X_mean[:,None,:])**2)

class ZSim():
    def __init__(self):
        self.N = 100
        self.tau = 0.1
        self.W = None
        self.synaptic_nonlinearity = ZSim.identity
        self.rate_nonlinearity = ZSim.identity
        self.dt = 0.001
        self.sim_duration = 8
        self.I_sac = np.ones(self.N) / np.sqrt(self.N)
        self.I_stims = None
        self.I_tonic = np.zeros(self.N)
        self.sac_size = 20
        self.stim_size = 10
        self.stim_sign_constraints = None
        self.neuronal_noise = 0
        self.simulated_rates = None
        self.measurement_noise = 0
        self.cirf_taus = None
        self.measured_quantity = 'fluor'
        self.verbosity = 1
        return

    def verbalize(self,message,frivolity):
        if self.verbosity > frivolity:
            print(message)
        return

    def configure(self,commands):
        for command in commands:
            if command[0] == 'n':
                self.N = int(command[1])
            elif command[0] == 'simtau':
                self.tau = float(command[1])
            elif command[0] == 'simdt':
                self.dt = float(command[1])
            elif command[0] == 'simw':
                self.load_W(command[1])
            elif command[0] == 'simduration':
                self.sim_duration = float(command[1])
            elif command[0] == 'saccadesize':
                self.sac_size = float(command[1])
            elif command[0] == 'stimsize':
                self.stim_size = float(command[1])
            elif command[0] == 'stimsignconstraint':
                command[1] = command[1].translate({ord(c): None for c in '[]()'})
                command[1] = command[1].translate({ord(c): ord(',') for c in ':;|/'})
                signs = command[1].split(',')
                ns = len(signs)
                if ns == 1:
                    self.stim_sign_constraints = int(signs)
                elif ns == self.N:
                    self.stim_sign_constraints = np.array([int(s) for s in signs])
            elif command[0] == 'neuronalnoise':
                self.neuronal_noise = float(command[1])
            elif command[0] == 'measurementnoise':
                self.measurement_noise = float(command[1])
            elif command[0] == 'measurement':
                self.measured_quantity = command[1].lower()
            elif command[0] == 'synapticnonlinearity':
                fun = command[1].lower()
                if fun == 'relu':
                    self.synaptic_nonlinearity = ZSim.relu
                    self.synaptic_nonlinearity_prime = ZSim.relu_prime
                elif fun == 'saturatedlinear':
                    self.synaptic_nonlinearity = ZSim.saturated_linear
                    self.synaptic_nonlinearity_prime = ZSim.saturated_linear_prime
            elif command[0] == 'ratenonlinearity':
                fun = command[1].lower()
                if fun == 'relu':
                    self.rate_nonlinearity_primenonlinearity = ZSim.relu
                    self.rate_nonlinearity_prime = ZSim.relu_prime
                elif fun == 'saturatedlinear':
                    self.rate_nonlinearity = ZSim.saturated_linear
                    self.rate_nonlinearity_prime = ZSim.saturated_linear_prime
        return

    def load_W(self,infile='',matrix=None):
        if matrix is None:
            phil = open(infile,'rb')
            self.W = pickle.load(phil)
            phil.close()
        else:
            self.W = copy.deepcopy(matrix)
        n = np.shape(self.W)[0]
        if not n == self.N:
            self.N = n
        y,v = ZSim.sorted_eigs(self.W)
        vr = np.real(v[:,0])
        self.I_sac = vr * np.sign(np.sum(vr))
        if not np.shape(self.I_tonic)[0] == self.N:
            self.I_tonic = np.zeros(self.N)
        return

    def generate_weight_matrix(self,method,**kwargs):
        if method == 'orthogonal':
            self.W = ZSim.generate_orthogonal_matrix(self.N,kwargs['leading_eigenvalues'])
        elif method == 'uniform random':
            self.W = ZSim.generate_uniform_random_matrix(self.N,kwargs['mean'],kwargs['std'],kwargs['sign_constraints'],kwargs['leading_eigenvalues'])
        elif method == 'block random':
            self.W = ZSim.generate_block_random_matrix(self.N,kwargs['block_id'],kwargs['mean'],kwargs['std'],kwargs['sign_constraints'],kwargs['leading_eigenvalues'])
            self.block_id = kwargs['block_id']
        y,v = ZSim.sorted_eigs(self.W)
        vr = np.real(v[:,0])
        self.I_sac = vr * np.sign(np.sum(vr))
        if not np.shape(self.I_tonic)[0] == self.N:
            self.I_tonic = np.zeros(self.N)
        return

    @staticmethod
    def sorted_eigs(M):
        y,v = np.linalg.eig(M)
        vtmp = copy.deepcopy(y)
        n = len(y)
        for i in range(n):
            for j in range(i+1,n):
                if np.real(y[j]) > np.real(y[i]):
                    tmp = y[i]
                    y[i] = y[j]
                    y[j] = tmp
                    vtmp[:] = v[:,i]
                    v[:,i] = v[:,j]
                    v[:,j] = vtmp[:]
        return y,v
    
    @staticmethod
    def generate_orthogonal_matrix(N,leading_eigenvalues):
        n = len(leading_eigenvalues)
        y = np.array(leading_eigenvalues)
        if n < N:
            y = np.zeros(N)
            for i in range(n):
                y[i] = leading_eigenvalues[i]
            filler_max = np.min(abs(leading_eigenvalues))
            y[n:] = (filler_max / 2) * np.random.randn(N-n)
            y[n:][y[n:] > filler_max] = filler_max
        E = scipy.stats.ortho_group.rvs(dim=N)
        return np.real(E @ np.diag(y) @ E.T)

    @staticmethod
    def generate_uniform_random_matrix(N,mean,std,sign_constraints,leading_eigenvalues):
        W = mean + std * np.random.randn(N,N)
        if not sign_constraints is None:
            W *= (W * sign_constraints > 0)
            np.fill_diagonal(W,0)
            y,E = np.linalg.eig(W)
            W *= leading_eigenvalues[0] / np.max(np.real(y))
        else:
            y,E = ZSim.sorted_eigs(W)
            y *= leading_eigenvalues[0] / np.real(y[0])
            real_counter = 1
            while real_counter < N and abs(np.imag(y[real_counter])) > 1e-10:
                real_counter += 2
            imag_counter = 1
            while imag_counter < N and abs(np.imag(y[imag_counter])) < 1e-10:
                imag_counter += 1
            for i in range(1,len(leading_eigenvalues)):
                if abs(np.imag(leading_eigenvalues[i])) > 1e-10:
                    while imag_counter < N and abs(np.imag(y[imag_counter])) < 1e-10:
                        imag_counter += 1
                    if imag_counter < N:
                        y[imag_counter] = leading_eigenvalues[i]
                        imag_counter += 1
                        y[imag_counter] = np.conjugate(leading_eigenvalues[i])
                        imag_counter += 1

                else:
                    while real_counter < N and abs(np.imag(y[real_counter])) > 1e-10:
                        real_counter += 2
                    if real_counter < N:
                        y[real_counter] = leading_eigenvalues[i]
                        real_counter += 1
                    elif imag_counter < N:
                        y[imag_counter] = leading_eigenvalues[i] + np.imag(y[imag_counter])
                        imag_counter += 1
                        y[imag_counter] = leading_eigenvalues[i] + np.imag(y[imag_counter])
                        imag_counter += 1
            W = np.real(E @ np.diag(y) @ np.linalg.pinv(E))
        return W
    
    @staticmethod
    def generate_block_random_matrix(N,block_id,mean,std,sign_constraints,leading_eigenvalues):
        W = np.zeros((N,N))
        nb = np.max(block_id) - np.min(block_id) + 1
        bid = np.unique(block_id)
        for i in range(nb):
            bi = (block_id == bid[i])
            ni = np.sum(bi)
            for j in range(nb):
                bj = (block_id == bid[j])
                nj = np.sum(bj)
                W[np.ix_(bi,bj)] = mean[i,j] + std[i,j] * np.random.randn(ni,nj)
        if not sign_constraints is None:
            W *= (W * sign_constraints > 0)
            np.fill_diagonal(W,0)
            y,E = np.linalg.eig(W)
            W *= leading_eigenvalues[0] / np.max(np.real(y))
        else:
            y,E = ZSim.sorted_eigs(W)
            y *= leading_eigenvalues[0] / np.real(y[0])
            real_counter = 1
            imag_counter = 1
            for i in range(1,len(leading_eigenvalues)):
                if abs(np.imag(leading_eigenvalues[i])) > 1e-10:
                    while abs(np.imag(y[imag_counter])) < 1e-10:
                        imag_counter += 1
                    y[imag_counter] = leading_eigenvalues[i]
                    imag_counter += 1
                    y[imag_counter] = np.conjugate(leading_eigenvalues[i])
                    imag_counter += 1

                else:
                    while abs(np.imag(y[real_counter])) > 1e-10:
                        real_counter += 2
                    y[real_counter] = leading_eigenvalues[i]
                    real_counter += 1
            W = np.real(E @ np.diag(y) @ np.linalg.pinv(E))
        return W

    def assign_cirf_taus(self,fixed_cirf=-1):
        if fixed_cirf > 0:
            self.cirf_taus = fixed_cirf * np.ones(self.N)
        else:
            self.cirf_taus = np.zeros(self.N)
            for i in range(self.N):
                self.cirf_taus[i] = ZSim.cdf_inverse(np.random.rand())
                if self.cirf_taus[i] < 0.7:
                    self.cirf_taus[i] = 0.7
                if self.cirf_taus[i] > 3.98:
                    self.cirf_taus[i] = 3.98
        return

    @staticmethod
    def cdf_inverse(x):
        return 1.5 - 0.3*np.log(1.0/x - 1.0)/np.log(3)

    def generate_stimulations(self,n):
        self.I_stims = np.random.randn(n,self.N)
        if not self.stim_sign_constraints == None:
            for i in range(n):
                self.I_stims[i,:] = abs(self.I_stims[i,:]) * self.stim_sign_constraints
        for i in range(n):
            self.I_stims[i,:] = self.I_stims[i,:] / np.linalg.norm(self.I_stims[i,:])
        return

    def generate_tonic(self,mean,std):
        self.I_tonic = mean + std * np.random.randn(self.N)
        return
    
    def simulate(self,data):
        pt1 = time.monotonic()

        self.generate_stimulations(data.nstims)
        T_gen = int(self.sim_duration/self.dt) + 1
        self.t_sac = np.zeros(T_gen)
        self.t_sac[int(data.sac_period[0]/self.dt):int(data.sac_period[1]/self.dt)+1] = self.sac_size
        self.t_stim = np.zeros(T_gen)
        end_of_input = int(data.stim_period[1]/self.dt)+1
        self.t_stim[int(data.stim_period[0]/self.dt):end_of_input] = self.stim_size
        self.simulated_rates = np.concatenate([self.sim_stim(s,data.ntrials[s],T_gen) for s in range(data.nstims+1)],axis=2)

        pt2 = time.monotonic()
        pt3=pt2
        T_samp = int(self.sim_duration/data.dt_samp)
        prev_trials = np.zeros(data.nstims + 1,dtype=int)
        for i in range(data.nstims):
            prev_trials[i+1] = prev_trials[i] + data.ntrials[i]
        tot_trials = np.sum(data.ntrials)
        data.measurements = np.zeros((T_samp,self.N,tot_trials))
        data.measurement_times = np.zeros((T_samp,tot_trials))
        b = int(data.dt_samp/self.dt)
        t = np.linspace(0,self.sim_duration,T_gen)
        if self.measured_quantity == 'fluor':
            t_back = np.linspace(self.sim_duration,0,T_gen)
            c = ZSim.rate_to_fluor(self.N,self.dt,T_gen,t_back,self.cirf_taus,data.nstims,data.ntrials,self.simulated_rates)
            pt3 = time.monotonic()
            offset = np.random.randint(b,size=tot_trials)
            for i in range(tot_trials):
                data.measurements[:,:,i] = c[offset[i]:offset[i] + b*T_samp:b,:,i]
                data.measurement_times[:,i] = t[offset[i]:offset[i] + b*T_samp:b]
            data.measurements += self.measurement_noise * np.mean(c[end_of_input:,:,:],axis=0) * np.random.randn(T_samp,self.N,tot_trials)
        elif self.measured_quantity == 'rate':
            offset = np.random.randint(b,size=tot_trials)
            n = self.measurement_noise * np.random.randn(T_samp,self.N,tot_trials)
            for i in range(tot_trials):
                data.measurements[:,:,i] = self.simulated_rates[offset[i]:offset[i] + b*T_samp:b,:,i] + n[:,:,i]
                data.measurement_times[:,i] = t[offset[i]:offset[i] + b*T_samp:b]
        if data.average_trials:
            b = int(data.dt_int/self.dt)
            avgs = []
            for i in range(data.nstims+1):
                avgs.append(np.mean(self.simulated_rates[:,:,prev_trials[i]:prev_trials[i]+data.ntrials[i]],axis=2)[None,:T_gen-1:b,:])
            self.simulated_rates = np.concatenate(avgs,axis=0)
        
        pt4 = time.monotonic()
        self.verbalize(f'Time to simulate rates: {pt2-pt1} seconds',9)
        self.verbalize(f'Time to make measurements: {pt3-pt2} + {pt4-pt3} seconds',9)
        return

    def sim_stim(self,s,nt,T):
        r = np.zeros((T,self.N,nt))
        I_stim = np.zeros(self.N)
        if s > 0:
            I_stim = self.I_stims[s-1]
        dt_tau = self.dt / self.tau
        srdt_tau = np.sqrt(self.dt) / self.tau
        for i in range(1,T):
            n = np.random.randn(self.N,nt) * self.neuronal_noise
            I_tot = self.I_tonic + self.t_sac[i-1]*self.I_sac + self.t_stim[i-1]*I_stim
            dr = dt_tau * (-r[i-1,:,:] + self.W @ self.synaptic_nonlinearity(r[i-1,:,:]) + I_tot[:,None]) + srdt_tau * n
            r[i,:,:] = self.rate_nonlinearity(r[i-1,:,:] + dr)
        return r
    
    @staticmethod
    @numba.njit
    def rate_to_fluor(N,dt,T,t_back,taus,nstims,ntrials,r):
        c = np.zeros(np.shape(r))
        cirf = np.zeros((T,T))
        for i in range(N):
            decay = dt*np.exp(-t_back/taus[i])
            for j in range(T):
                cirf[j,0:j+1] = decay[-j-1:]
            c[:,i,:] = cirf @ r[:,i,:]
        return c
 
    #####################################
    ##### Available Non-linearities #####
    #####################################

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def saturated_linear(x):
        return x / (1 + x)
