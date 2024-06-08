import pickle
import numpy as np
import math
import scipy.optimize
import scipy.linalg
import copy
import time
import random
from ZData import *

class ZFit():
    def __init__(self,confile):
        self.fit_method = 'lsq'
        self.allow_autapse = False
        self.allow_tonic = False
        self.allow_inhibition = False
        self.obey_dale = True
        self.max_inhibitory_fraction = 0.2
        self.inflate = self.inflate_no_autapse
        self.residuals = self.residuals_full
        self.jacobian = self.jacobian_full_no_autapse
        self.add_nuclear_norm_gradient = self.add_nuclear_norm_gradient_no_autapse
        self.wmax = 999.9
        self.bmax = 999.9
        self.rmax = 999.9
        self.rmin = -999.9
        self.regularizations = {'None' : 0}
        self.prior = 'uniform'
        self.include_stims = True
        self.synaptic_nonlinearity = ZSim.identity
        self.synaptic_nonlinearity_prime = ZFit.identity_prime
        self.rate_nonlinearity = ZSim.identity
        self.rate_nonlinearity_prime = ZFit.identity_prime
        self.start_time = 0.5
        self.start_index = 0
        self.tau = 0.1
        self.observed_fraction = 1.0
        self.observed_cells = None
        self.whiten = True
        self.clma_min_iterations = 1
        self.clma_max_iterations = 1
        self.clma_min_vrel = 1e-6
        self.clma_max_vrel = 0.01
        self.r2_method = 'avg'
        self.verbosity = 1
        self.base_directory = ''
        self.configure(confile)

        self.W = None
        self.B = None
        self.r0s = None
        self.R2 = None
        return
    
    def verbalize(self,message,frivolity):
        if self.verbosity > frivolity:
            print(message)
        return

    def configure(self,confile):
        try:
            phil = open(confile,'r')
            commands = phil.read().split('\n')
            for command in commands:
                command = command.split('#')[0] #discard comments
                command = command.replace(' ','')
                #print(command)
                command = command.split('=')
                command[0] = command[0].lower().translate({ord(c): None for c in '_-.'})
                if command[0] == 'fitmethod':
                    self.verbalize(f'Setting fit method to {command[1]}...',4)
                    self.fit_method = command[1]
                elif command[0] == 'miniter':
                    self.verbalize(f'Setting minimum CLMA iterations to {command[1]}...',4)
                    self.clma_min_iterations = int(command[1])
                elif command[0] == 'maxiter':
                    self.verbalize(f'Setting maximum CLMA iterations to {command[1]}...',4)
                    self.clma_max_iterations = int(command[1])
                elif command[0] == 'allowautapse':
                    self.allow_autapse = (command[1].lower() == 'true')
                    self.verbalize(f'Setting autapses allowed to {self.allow_autapse}...',4)
                elif command[0] == 'allowtonic':
                    self.allow_tonic = (command[1].lower() == 'true')
                    self.verbalize(f'Setting tonic input allowed to {self.allow_tonic}...',4)
                elif command[0] == 'allowinhibition':
                    self.allow_inhibition = (command[1].lower() == 'true')
                    self.verbalize(f'Setting inhibition allowed to {self.allow_inhibition}...',4)
                elif command[0] == 'obeydale':
                    self.obey_dale = (command[1].lower() == 'true')
                    self.verbalize(f'Setting Dales Law enforcement to {self.obey_dale}...',4)
                elif command[0] == 'maxinhibition':
                    self.verbalize(f'Setting maximum fraction of inhibitory cells to {command[1]}...',4)
                    self.max_inhibitory_fraction = float(command[1])
                elif command[0] == 'maxweight':
                    self.verbalize(f'Setting maximum connnection weight to {command[1]}...',4)
                    self.wmax = float(command[1])
                elif command[0] == 'maxtonic':
                    self.verbalize(f'Setting maximum tonic input to {command[1]}...',4)
                    self.bmax = float(command[1])
                elif command[0] == 'maxrate':
                    self.verbalize(f'Setting maximum initial firing rate to {command[1]}...',4)
                    self.rmax = float(command[1])
                elif command[0] == 'minrate':
                    self.verbalize(f'Setting minimum initial firing rate to {command[1]}...',4)
                    self.rmin = float(command[1])
                elif command[0] == 'regularization':
                    command[1] = command[1].translate({ord(c): None for c in '[]()'})
                    command[1] = command[1].translate({ord(c): ord(',') for c in ':;|/'})
                    self.verbalize(f'Adding regularization: {command[1]}...',4)
                    reg = command[1].split(',')
                    self.regularizations[reg[0].upper()] = float(reg[1])
                elif command[0] == 'prior':
                    self.verbalize(f'Setting prior to {command[1]}...',4)
                    self.prior = command[1]
                elif command[0] == 'includestims':
                    self.include_stims = (command[1].lower() == 'true')
                    self.verbalize(f'Setting inclusion of stimulations to {self.include_stims}...',4)
                elif command[0] == 'synapticnonlinearity':
                    fun = command[1].lower()
                    self.verbalize(f'Setting synaptic nonlinearity to {fun}...',4)
                    if fun == 'relu':
                        self.synaptic_nonlinearity = ZSim.relu
                        self.synaptic_nonlinearity_prime = ZFit.relu_prime
                    elif fun == 'saturatedlinear':
                        self.synaptic_nonlinearity = ZSim.saturated_linear
                        self.synaptic_nonlinearity_prime = ZFit.saturated_linear_prime
                elif command[0] == 'ratenonlinearity':
                    fun = command[1].lower()
                    self.verbalize(f'Setting rate nonlinearity to {fun}...',4)
                    if fun == 'relu':
                        self.rate_nonlinearity_primenonlinearity = ZSim.relu
                        self.rate_nonlinearity_prime = ZFit.relu_prime
                    elif fun == 'saturatedlinear':
                        self.rate_nonlinearity = ZSim.saturated_linear
                        self.rate_nonlinearity_prime = ZFit.saturated_linear_prime
                elif command[0] == 'start':
                    self.verbalize(f'Setting start time to {command[1]}...',4)
                    self.start_time = float(command[1])
                elif command[0] == 'tau':
                    self.verbalize(f'Setting intrinsic time constant to {command[1]}...',4)
                    self.tau = float(command[1])
                elif command[0] == 'observedfraction':
                    self.verbalize(f'Setting observed fraction of the network to {command[1]}...',4)
                    self.observed_fraction = float(command[1])
                elif command[0] == 'observedcells':
                    command[1] = command[1].translate({ord(c): None for c in '[]()'})
                    command[1] = command[1].translate({ord(c): ord(',') for c in ':.-;|/'})
                    self.verbalize(f'Setting observed cells to: {command[1]}...',4)
                    cells = command[1].split(',')
                    self.observed_cells = [int(c) for c in cells]
                elif command[0] == 'whiten':
                    self.whiten = (command[1].lower() == 'true')
                    self.verbalize(f'Setting data whitening to {self.whiten}...',4)
                elif command[0] == 'r2method':
                    self.verbalize(f'Setting R^2 reporting method to {command[1]}...',4)
                    self.r2_method = command[1]
                elif command[0] == 'verbosity':
                    self.verbalize(f'Setting verbosity to {command[1]}...',4)
                    self.verbosity = int(command[1])
                elif command[0] == 'basedir':
                    self.verbalize(f'Setting base directory to {command[1]}...',4)
                    self.base_directory = command[1]
                    if '/' in self.base_directory and not self.base_directory[-1] == '/':
                        self.base_directory += '/'
                    elif '\\' in self.base_directory and not self.base_directory[-1] == '\\':
                        self.base_directory += '\\'
        except FileNotFoundError:
            print('No or invalid configuration file provided. Using defaults...')
        if self.allow_autapse:
            self.inflate = self.inflate_include_autapse
            self.jacobian_full_allow_autapse
            self.add_nuclear_norm_gradient = self.add_nuclear_norm_gradient_allow_autapse
        if not self.allow_inhibition:
            self.obey_dale = True
        if not (self.synaptic_nonlinearity == ZSim.identity and self.rate_nonlinearity == ZSim.identity):
            self.whiten = False
        return
    
    def load(self,infile='',data=None):
        fitdata = data
        if data is None:
            f = open(f'{self.base_directory}{infile}','rb')
            fitdata = pickle.load(f)
            f.close()
        self.train_rates = fitdata.firing_rate_estimate
        self.dt = fitdata.dt_int
        self.start_index = int((fitdata.stim_period[1] + self.start_time) / self.dt)
        self.rc_pos = fitdata.centroid[0,:]
        self.ml_pos = fitdata.centroid[1,:]
        if not fitdata.sim is None:
            self.true_W = fitdata.sim.W
        
        nstim,T,self.N = np.shape(self.train_rates)
        self.nstims = nstim - 1
        self.T = T - self.start_index
        if not self.observed_cells is None and len(self.observed_cells) < self.N:
            self.observed_fraction = len(self.observed_cells) / self.N
            self.N = len(self.observed_cells)
            self.train_rates = self.train_rates[:,:,self.observed_cells]
            self.rc_pos = self.rc_pos[self.observed_cells]
            self.ml_pos = self.ml_pos[self.observed_cells]
        elif self.observed_fraction < 1:
            cells = list(range(self.N))
            self.N = int(self.observed_fraction * self.N)
            self.observed_cells = random.sample(cells,self.N)
            self.observed_cells.sort()
            self.train_rates = self.train_rates[:,:,self.observed_cells]
            self.rc_pos = self.rc_pos[self.observed_cells]
            self.ml_pos = self.ml_pos[self.observed_cells]
        else:
            self.observed_cells = list(range(self.N))
        self.scale_factor = np.ones(self.N)
        if self.whiten:
            self.scale_factor = np.max(np.mean(abs(self.train_rates[:,self.start_index:,:]),axis=1),axis=0)
            msf = 1.0 / np.mean(self.scale_factor)
            self.scale_factor = 1.0 / self.scale_factor
            self.scale_factor[self.scale_factor > 10*msf] = 10*msf
            #self.scale_factor[self.scale_factor > 10] = 10
        self.verbalize(f'Whitening scale factors:\n{self.scale_factor}',10)
        self.tau_eff = np.zeros(self.N)
        for k in range(self.N):
            t = np.linspace(0,(self.T-1)*self.dt,self.T)
            denom = self.train_rates[0,self.start_index,k]
            if abs(denom) > 1e-10:
                logratio = np.log(self.train_rates[0,self.start_index:,k]/denom)
                self.tau_eff[k] = -np.dot(t,logratio)/np.dot(logratio,logratio)
        self.dx = np.zeros((self.N,self.N))
        self.W_prior = np.zeros((self.N,self.N))
        self.penalty_matrix = np.ones((self.N,self.N))
        self.penalty_matrix_inh = np.ones((self.N,self.N))
        found_prior_matrix = False
        if '.' in self.prior:
            try:
                f = open(self.prior,'rb')
                self.W_prior = pickle.load(f)
                f.close()
                found_prior_matrix = True
            except:
                pass
        if not found_prior_matrix and not self.prior == 'uniform':
            if 'tau' in self.prior:
                for k in range(self.N):
                    for l in range(self.N):
                        dx[k,l] = self.tau_eff[k] - self.tau_eff[l]
            elif 'rc' in self.prior:
                for k in range(self.N):
                    for l in range(self.N):
                        dx[k,l] = self.rc_pos[k] - self.rc_pos[l]
            elif 'ml' in self.prior:
                for k in range(self.N):
                    for l in range(self.N):
                        dx[k,l] = self.ml_pos[k] - self.ml_pos[l]
            if np.percentile(self.dx,95) > 0:
                self.dx = self.dx / np.percentile(self.dx,95)
            if 'local' in self.prior:
                adx = abs(self.dx)
                self.penalty_matrix = self.N * adx / np.linalg.norm(adx)
                self.penalty_matrix_inh = 1 - adx
                self.penalty_matrix_inh = self.penalty_matrix_inh - np.min(self.penalty_matrix_inh)
                self.penalty_matrix_inh = self.N * self.penalty_matrix_inh / np.linalg.norm(self.penalty_matrix_inh)
            elif 'distal' in self.prior:
                adx = abs(self.dx)
                self.penalty_matrix_inh = self.N * adx / np.linalg.norm(adx)
                self.penalty_matrix = 1 - adx
                self.penalty_matrix = self.penalty_matrix - np.min(self.penalty_matrix)
                self.penalty_matrix = self.N * self.penalty_matrix / np.linalg.norm(self.penalty_matrix)
            elif 'feedforward' in self.prior:
                self.penalty_matrix = np.max(self.dx) - self.dx
                self.penalty_matrix = self.N * self.penalty_matrix / np.linalg.norm(self.penalty_matrix)
                self.penalty_matrix_inh = self.dx - np.min(self.dx)
                self.penalty_matrix_inh = self.N * self.penalty_matrix_inh / np.linalg.norm(self.penalty_matrix_inh)
            elif 'feedback' in self.prior:
                self.penalty_matrix = self.dx - np.min(self.dx)
                self.penalty_matrix = self.N * self.penalty_matrix / np.linalg.norm(self.penalty_matrix)
                self.penalty_matrix_inh = np.max(self.dx) - self.dx
                self.penalty_matrix_inh = self.N * self.penalty_matrix_inh / np.linalg.norm(self.penalty_matrix_inh)
        return

    @staticmethod
    def load_fit(infile):
        f = open(infile,'rb')
        fit = pickle.load(f)
        f.close()
        return fit

    def save(self,outfile):
        self.train_rates = None
        self.Rt = None
        self.dR = None
        self.SRt = None
        self.Rt_flat = None
        self.Ji = None
        self.x0 = None
        self.lower_bounds = None
        self.upper_bounds = None
        self.penalty_flat = None
        self.prior_flat = None
        f = open(f'{self.base_directory}{outfile}','wb')
        pickle.dump(self,f)
        f.close()
        return

    def fit(self):
        self.ntrials = 1
        if self.include_stims:
            self.ntrials += self.nstims
        
        T_x = self.ntrials*(self.T-1)
        self.Rt = np.zeros((T_x,self.N))
        Rtp1 = np.zeros((T_x,self.N))
        for i in range(self.ntrials):
            self.Rt[i*(self.T-1):(i+1)*(self.T-1),:] = self.train_rates[i,self.start_index:-1,:] * self.scale_factor
            Rtp1[i*(self.T-1):(i+1)*(self.T-1),:] = self.train_rates[i,self.start_index+1:,:] * self.scale_factor
        self.dt_tau = self.dt / self.tau
        self.dR = (Rtp1 - self.Rt)/self.dt_tau + self.Rt
        self.SRt = self.synaptic_nonlinearity(self.Rt)
        self.Rt = (self.Rt * 1e8).astype(int).astype(float) * 1e-8
        self.SRt = (self.SRt * 1e8).astype(int).astype(float) * 1e-8
        self.dR = (self.dR * 1e8).astype(int).astype(float) * 1e-8

        self.Rt_flat = np.zeros(self.ntrials * self.T * self.N)
        for i in range(self.ntrials):
            for j in range(self.N):
                self.Rt_flat[i*self.N+j::self.ntrials*self.N] = self.train_rates[i,self.start_index:,j] * self.scale_factor[j]
            #for j in range(self.T):
            #    self.Rt_flat[i*self.N+j*self.N*self.ntrials:(i+1)*self.N+j*self.N*self.ntrials] = self.train_rates[i,self.start_index+j,:] * self.scale_factor
        self.Rt_flat = (self.Rt_flat * 1e8).astype(int).astype(float) * 1e-8

        reg_types = list(self.regularizations.keys())
        for reg in reg_types:
            self.regularizations[reg] *= self.ntrials

        self.W = np.zeros((self.N,self.N))
        self.B = np.zeros(self.N)
        self.r0s = np.zeros((self.ntrials,self.N))
        if not self.obey_dale:
            #fit with no sign constraints
            if 'NUC' in reg_types:
                self.cost = self.fit_next(np.zeros(self.N),'clma')
            else:
                self.cost = self.fit_next(np.zeros(self.N),'free')
            if self.fit_method == 'clma':
                self.cost = self.fit_full(np.zeros(self.N,dtype=int))
        else:
            #fit all positive matrix
            sign_constraints = np.ones(self.N,dtype=int)
            if 'NUC' in reg_types:
                self.cost = self.fit_next(sign_constraints,'clma')
            else:
                self.cost = self.fit_next(sign_constraints,'dale')
            if self.fit_method == 'clma':
                self.cost = self.fit_full(sign_constraints)
            if self.allow_inhibition:
                #try fitting matrices with small fraction of inhibitory cells and keep if better than all positive fit
                best_W = copy.deepcopy(self.W)
                best_B = copy.deepcopy(self.B)
                best_r0s = copy.deepcopy(self.r0s)
                best_cost = self.cost
                neglim = int(self.max_inhibitory_fraction * self.N)
                test_inhibitory_groupings = []
                for selection_method in range(3):
                    if 'NUC' in reg_types:
                        self.fit_next(np.zeros(self.N),'clma')
                    else:
                        self.fit_next(np.zeros(self.N),'free')
                    index_list = []
                    sign_constraints = np.zeros(self.N,dtype=int)
                    for ineg in range(neglim):
                        if selection_method == 0:
                            col_sums = np.sum((self.W < 0),axis=0)
                            col_sums[sign_constraints < 0] = -1
                            index = np.argmax(col_sums)
                        else:
                            if selection_method == 1:
                                col_sums = np.sum(self.W,axis=0)
                            else:
                                col_sums = np.min(self.W,axis=0) 
                            col_sums[sign_constraints < 0] = 2*np.max(col_sums)
                            index = np.argmin(col_sums)
                        index_list.append(index)
                        index_list.sort()
                        sign_constraints[index] = -1
                        if 'NUC' in reg_types:
                            self.fit_next(sign_constraints,'clma')
                        else:
                            self.fit_next(sign_constraints,'free')
                        if not (index_list in test_inhibitory_groupings):
                            test_inhibitory_groupings.append(copy.deepcopy(index_list))
                best_group = []
                sign_constraints = np.ones(self.N,dtype=int)
                for in_group in test_inhibitory_groupings:
                    self.verbalize(f'Testing inhibitory grouping: {in_group}',1)
                    for i in in_group:
                        sign_constraints[i] = -1
                    new_cost = 999999.9
                    if 'NUC' in reg_types:
                        new_cost = self.fit_next(sign_constraints,'clma')
                    else:
                        new_cost = self.fit_next(sign_constraints,'dale')
                    if self.fit_method == 'clma':
                        new_cost = self.fit_full(sign_constraints)#,iteration_limit_override=30)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_W[:,:] = self.W[:,:]
                        best_B[:] = self.B[:]
                        best_r0s[:,:] = self.r0s[:,:]
                        best_group = in_group
                    for i in in_group:
                        sign_constraints[i] = 1
                self.verbalize(f'Best inhibitory grouping: {best_group}',0)
                self.W = best_W
                self.B = best_B
                self.r0s = best_r0s
                self.cost = best_cost
                #if self.fit_method == 'clma':
                #    for i in best_group:
                #        sign_constraints[i] = -1
                #    self.cost = self.fit_full(sign_constraints)
        '''scale fitted parameters to account for data whitening'''
        for i in range(self.N):
            self.W[i,:] = self.W[i,:] * self.scale_factor / self.scale_factor[i]
        self.B = self.B / self.scale_factor
        self.r0s = self.r0s / self.scale_factor

        for reg in reg_types:
            self.regularizations[reg] /= self.ntrials

        self.R2 = self.get_r2()
        self.verbalize(f'train R^2: {self.R2}',0)
        y,v = ZSim.sorted_eigs(self.W)
        self.verbalize(f'Leading eigenvalues of fit matrix:\n{y[0:8]}...',1)
        return

    #####################################
    ##### Available Non-linearities #####
    #####################################

    @staticmethod
    def identity_prime(x):
        return np.ones(np.shape(x))

    @staticmethod
    def relu_prime(x):
        return np.ones(np.shape(x)) * (x > 0)

    @staticmethod
    def saturated_linear_prime(x):
        return (1 + x)**(-1) - (x * (1 + x)**(-2))

    ###############################################
    ##### Next-time-step Regression Functions #####
    ###############################################

    def fit_next(self,sign_constraints,ftype,rerolls=2):
        self.verbalize(f'Fitting next time step cost function...',1)
        if ftype == 'free':
            return self.fit_lsq_free(sign_constraints)
        elif ftype == 'dale':
            return self.fit_lsq_dale(sign_constraints)
        elif ftype == 'clma':
            #self.init_clma_next(sign_constraints)
            if self.allow_tonic:
                self.residuals = self.residuals_next_tonic
                if self.allow_autapse:
                    self.jacobian = self.jacobian_next_allow_autapse_tonic
                else:
                    self.jacobian = self.jacobian_next_no_autapse_tonic
            else:
                self.residuals = self.residuals_next
                if self.allow_autapse:
                    self.jacobian = self.jacobian_next_allow_autapse
                else:
                    self.jacobian = self.jacobian_next_no_autapse
            nx = (self.N + int(self.allow_tonic))*self.N
            self.x0 = np.zeros(nx)
            self.prior_flat = np.zeros(nx)
            self.penalty_flat = np.zeros(nx)
            self.lower_bounds = np.zeros(nx)
            self.upper_bounds = np.zeros(nx)
            for icol in range(self.N):
                if sign_constraints[icol] < 1:
                    self.lower_bounds[icol*self.N:(icol+1)*self.N] = -self.wmax*self.scale_factor/self.scale_factor[icol]
                else:
                    self.lower_bounds[icol*self.N:(icol+1)*self.N] = 0
                if sign_constraints[icol] > -1:
                    self.upper_bounds[icol*self.N:(icol+1)*self.N] = self.wmax*self.scale_factor/self.scale_factor[icol]
                else:
                    self.upper_bounds[icol*self.N:(icol+1)*self.N] = 0
                self.penalty_flat[icol*self.N:(icol+1)*self.N] = self.penalty_matrix[:,icol]
                self.prior_flat[icol*self.N:(icol+1)*self.N] = self.W_prior[:,icol]*self.scale_factor/self.scale_factor[icol]
            xindex = self.N**2
            if self.allow_tonic:
                self.lower_bounds[xindex:xindex+self.N] = -self.bmax * self.scale_factor
                self.upper_bounds[xindex:xindex+self.N] = self.bmax * self.scale_factor

            if not self.allow_autapse:
                cut = [i*self.N + i for i in range(self.N)]
                self.x0 = np.delete(self.x0,cut,axis=0)
                self.lower_bounds = np.delete(self.lower_bounds,cut,axis=0)
                self.upper_bounds = np.delete(self.upper_bounds,cut,axis=0)
                self.penalty_flat = np.delete(self.penalty_flat,cut,axis=0)
                self.prior_flat = np.delete(self.prior_flat,cut,axis=0)
            
            self.clma_max_vrel = 1.0
            cost = self.fit_clma(eta=0.5e-10)
            for i in range(rerolls):
                self.x0 += 0.01 * np.random.randn(len(self.x0))
                cost = self.fit_clma(eta=0.5e-10)

            xindex = self.inflate(self.x0)
            if self.allow_tonic:
                self.B[:] = self.x0[xindex:xindex+self.N]
                xindex += self.N
            for i in range(self.ntrials):
                self.r0s[i,:] = self.Rt_flat[i*self.N:(i+1)*self.N]
            return cost
        return -1

    def fit_lsq_free(self,sign_constraints):
        T_x = self.ntrials*(self.T-1)
        X = copy.deepcopy(self.SRt)
        reg_types = list(self.regularizations.keys())
        if 'L1' in reg_types:
            X = np.concatenate([X,np.zeros((1,self.N))],axis=0)
        if 'L2' in reg_types:
            X = np.concatenate([X,np.zeros((self.N,self.N))],axis=0)
        X = np.concatenate([X,-X],axis=1)
        lb = np.zeros(2*self.N)
        ub = np.zeros(2*self.N)
        if self.allow_tonic:
            X = np.concatenate([X,np.zeros((np.shape(X)[0],1))],axis=1)
            lb = np.zeros(2*self.N+1)
            ub = np.zeros(2*self.N+1)
        ub[0:self.N] = self.wmax / self.scale_factor
        ub[self.N:2*self.N] = self.wmax / self.scale_factor
        for i in range(self.N):
            if sign_constraints[i] < 0:
                ub[i] = 1e-10
        dr = np.zeros(T_x + int('L1' in reg_types) + (self.N * int('L2' in reg_types)))
        cost = 0
        for i in range(self.N):
            dr[0:T_x] = self.dR[:,i]
            lb[0:2*self.N] *= self.scale_factor[i]
            ub[0:2*self.N] *= self.scale_factor[i]
            if self.allow_tonic:
                X[0:T_x,2*self.N] = (abs(self.Rt[:,i]) > 1e-10).astype(float)
                lb[2*self.N] = -self.bmax * self.scale_factor[i]
                ub[2*self.N] = self.bmax * self.scale_factor[i]
            pindex = T_x
            if 'L1' in reg_types:
                X[pindex,0:self.N] = np.sqrt(self.regularizations['L1']*self.penalty_matrix[i,:]*self.scale_factor/self.scale_factor[i])
                X[pindex,self.N:2*self.N] = np.sqrt(self.regularizations['L1']*self.penalty_matrix_inh[i,:]*self.scale_factor/self.scale_factor[i])
                pindex += 1
            if 'L2' in reg_types:
                for j in range(self.N):
                    X[pindex+j,j] = np.sqrt(self.regularizations['L2']*self.penalty_matrix[i,j])*self.scale_factor[j]/self.scale_factor[i]
                    X[pindex+j,self.N+j] = np.sqrt(self.regularizations['L2']*self.penalty_matrix_inh[i,j])*self.scale_factor[j]/self.scale_factor[i]
                    dr[pindex+j] = X[pindex+j,j] * self.W_prior[i,j]*self.scale_factor[i]/self.scale_factor[j]
            if self.allow_autapse:
                res = scipy.optimize.lsq_linear(X,dr,bounds=(lb,ub),method='trf',tol=1e-10)
                self.W[i,:] = res.x[0:self.N] - res.x[self.N:2*self.N]
                if self.allow_tonic:
                    self.B[i] = res.x[2*self.N]
                cost += res.cost
            else:
                Xi = np.delete(X,[i,self.N+i],axis=1)
                lbi = np.delete(lb,[i,self.N+i],axis=0)
                ubi = np.delete(ub,[i,self.N+i],axis=0)
                res = scipy.optimize.lsq_linear(Xi,dr,bounds=(lbi,ubi),method='trf',tol=1e-10)
                self.W[i,:i] = res.x[0:i] - res.x[self.N-1:self.N-1+i]
                self.W[i,i+1:] = res.x[i:self.N-1] - res.x[self.N-1+i:2*self.N-2]
                if self.allow_tonic:
                    self.B[i] = res.x[2*self.N-2]
                cost += res.cost
            lb[0:2*self.N] /= self.scale_factor[i]
            ub[0:2*self.N] /= self.scale_factor[i]
        for i in range(self.ntrials):
            self.r0s[i,:] = self.Rt_flat[i*self.N:(i+1)*self.N]
        cost_data = 0.5 * np.sum((self.dR - self.SRt @ self.W.T)**2)
        self.verbalize(f'LSQ - Final Cost (Data, Reg, Tot): ({cost_data}, {cost - cost_data}, {cost})',0)
        return cost

    def fit_lsq_dale(self,sign_constraints):
        T_x = self.ntrials*(self.T-1)
        X = copy.deepcopy(self.SRt)
        reg_types = list(self.regularizations.keys())
        if 'L1' in reg_types:
            X = np.concatenate([X,np.zeros((1,self.N))],axis=0)
        if 'L2' in reg_types:
            X = np.concatenate([X,np.zeros((self.N,self.N))],axis=0)
        lb = np.zeros(self.N)
        ub = np.zeros(self.N)
        if self.allow_tonic:
            X = np.concatenate([X,np.zeros((np.shape(X)[0],1))],axis=1)
            lb = np.zeros(self.N+1)
            ub = np.zeros(self.N+1)
        lb[0:self.N] = -self.wmax * (sign_constraints < 1).astype(float) / self.scale_factor
        ub[0:self.N] = self.wmax * (sign_constraints > -1).astype(float) / self.scale_factor
        dr = np.zeros(T_x + int('L1' in reg_types) + (self.N * int('L2' in reg_types)))
        cost = 0
        for i in range(self.N):
            dr[0:T_x] = self.dR[:,i]
            lb[0:self.N] *= self.scale_factor[i]
            ub[0:self.N] *= self.scale_factor[i]
            if self.allow_tonic:
                X[0:T_x,self.N] = (abs(self.Rt[:,i]) > 1e-10).astype(float)
                lb[self.N] = -self.bmax * self.scale_factor[i]
                ub[self.N] = self.bmax * self.scale_factor[i]
            pindex = T_x
            if 'L1' in reg_types:
                X[pindex,0:self.N] = np.sqrt(self.regularizations['L1']*self.penalty_matrix[i,:]*sign_constraints*self.scale_factor/self.scale_factor[i])
                pindex += 1
            if 'L2' in reg_types:
                for j in range(self.N):
                    X[pindex+j,j] = np.sqrt(self.regularizations['L2']*self.penalty_matrix[i,j])*self.scale_factor[j]/self.scale_factor[i]
                    dr[pindex+j] = X[pindex+j,j] * self.W_prior[i,j]
            if self.allow_autapse:
                res = scipy.optimize.lsq_linear(X,dr,bounds=(lb,ub),method='trf',tol=1e-10)
                self.W[i,:] = res.x[0:self.N]
                if self.allow_tonic:
                    self.B[i] = res.x[self.N]
                cost += res.cost
            else:
                Xi = np.delete(X,i,axis=1)
                lbi = np.delete(lb,i,axis=0)
                ubi = np.delete(ub,i,axis=0)
                res = scipy.optimize.lsq_linear(Xi,dr,bounds=(lbi,ubi),method='trf',tol=1e-10)
                self.W[i,:i] = res.x[0:i]
                self.W[i,i+1:] = res.x[i:self.N-1]
                if self.allow_tonic:
                    self.B[i] = res.x[self.N-1]
                cost += res.cost
            lb[0:self.N] /= self.scale_factor[i]
            ub[0:self.N] /= self.scale_factor[i]
        for i in range(self.ntrials):
            self.r0s[i,:] = self.Rt_flat[i*self.N:(i+1)*self.N]
        cost_data = 0.5 * np.sum((self.dR - self.SRt @ self.W.T)**2)
        self.verbalize(f'LSQ - Final Cost (Data, Reg, Tot): ({cost_data}, {cost - cost_data}, {cost})',0)
        return cost
    
    #################################################
    ##### Full Time Series Regression Functions #####
    #################################################

    def fit_full(self,sign_constraints,iteration_limit_override=-1,rerolls=0):
        self.verbalize(f'Fitting full time series cost function...',1)
        if self.allow_tonic:
            self.residuals = self.residuals_full_tonic
            if self.allow_autapse:
                self.jacobian = self.jacobian_full_allow_autapse_tonic
            else:
                self.jacobian = self.jacobian_full_no_autapse_tonic
        else:
            self.residuals = self.residuals_full
            if self.allow_autapse:
                self.jacobian = self.jacobian_full_allow_autapse
            else:
                self.jacobian = self.jacobian_full_no_autapse
        y,v = np.linalg.eig(self.W)
        ymax = np.max(abs(np.real(y)))
        #print(ymax)
        ylim = np.exp(np.log(100)/self.T)
        if ymax > ylim:
            self.W = 0.99*self.W/ymax
        nx = (self.N + int(self.allow_tonic) + 1 + int(self.include_stims)*self.nstims)*self.N
        self.x0 = np.zeros(nx)
        self.prior_flat = np.zeros(nx)
        self.penalty_flat = np.zeros(nx)
        self.lower_bounds = np.zeros(nx)
        self.upper_bounds = np.zeros(nx)
        for icol in range(self.N):
            self.x0[icol*self.N:(icol+1)*self.N] = self.W[:,icol]#self.true_W[:,icol] * self.scale_factor / self.scale_factor[icol]
            if sign_constraints[icol] < 1:
                self.lower_bounds[icol*self.N:(icol+1)*self.N] = -self.wmax*self.scale_factor/self.scale_factor[icol]
            else:
                self.lower_bounds[icol*self.N:(icol+1)*self.N] = 0
            if sign_constraints[icol] > -1:
                self.upper_bounds[icol*self.N:(icol+1)*self.N] = self.wmax*self.scale_factor/self.scale_factor[icol]
            else:
                self.upper_bounds[icol*self.N:(icol+1)*self.N] = 0
            self.penalty_flat[icol*self.N:(icol+1)*self.N] = self.penalty_matrix[:,icol]
            self.prior_flat[icol*self.N:(icol+1)*self.N] = self.W_prior[:,icol]*self.scale_factor/self.scale_factor[icol]
        xindex = self.N**2
        if self.allow_tonic:
            self.x0[xindex:xindex+self.N] = self.B[:]
            self.lower_bounds[xindex:xindex+self.N] = -self.bmax * self.scale_factor
            self.upper_bounds[xindex:xindex+self.N] = self.bmax * self.scale_factor
            xindex += self.N
        self.x0[xindex:xindex+self.N] = self.r0s[0,:]
        self.lower_bounds[xindex:xindex+self.N] = self.rmin * self.scale_factor
        self.upper_bounds[xindex:xindex+self.N] = self.rmax * self.scale_factor
        if self.include_stims:
            xindex += self.N
            for i in range(self.nstims):
                self.x0[xindex:xindex+self.N] = self.r0s[i+1,:]
                self.lower_bounds[xindex:xindex+self.N] = self.rmin * self.scale_factor
                self.upper_bounds[xindex:xindex+self.N] = self.rmax * self.scale_factor
                xindex += self.N

        if not self.allow_autapse:
            cut = [i*self.N + i for i in range(self.N)]
            self.x0 = np.delete(self.x0,cut,axis=0)
            self.lower_bounds = np.delete(self.lower_bounds,cut,axis=0)
            self.upper_bounds = np.delete(self.upper_bounds,cut,axis=0)
            self.penalty_flat = np.delete(self.penalty_flat,cut,axis=0)
            self.prior_flat = np.delete(self.prior_flat,cut,axis=0)
        
        self.yi = np.zeros(self.N * self.T * self.ntrials)
        self.syi = np.zeros(self.N * self.T * self.ntrials)
        self.sypi = np.zeros(self.N * self.T * self.ntrials)
        self.WS = np.zeros((self.ntrials * self.N,self.ntrials * self.N))
        self.BS = np.zeros(self.ntrials * self.N)
        #self.clma_min_vrel = 1e-10
        self.clma_max_vrel = 0.01
        cost = self.fit_clma(eta=0.0005,iteration_limit_override=iteration_limit_override)
        for i in range(rerolls):
            self.x0 += 0.001 * np.random.randn(len(self.x0))
            cost = self.fit_clma(eta=0.0005)
        self.yi = None
        self.syi = None
        self.sypi = None
        self.WS = None
        self.BS = None

        xindex = self.inflate(self.x0)
        #for icol in range(self.N):
        #    self.W[:,icol] = self.x0[icol*self.N:(icol+1)*self.N]
        #xindex = self.N**2
        if self.allow_tonic:
            self.B[:] = self.x0[xindex:xindex+self.N]
            xindex += self.N
        self.r0s[0,:] = self.x0[xindex:xindex+self.N]
        if self.include_stims:
            xindex += self.N
            for i in range(self.nstims):
                self.r0s[i+1,:] = self.x0[xindex:xindex+self.N]
                xindex += self.N
        return cost

    ###########################################
    ##### Residual and Jacobian Functions #####
    ###########################################

    def residuals_next(self,x):
        N = self.N
        T_tot = self.ntrials * (self.T-1)
        self.inflate(x)
        #for i in range(N):
        #    self.W[:,i] = self.x0[i*N:(i+1)*N]
        r = self.dR - self.SRt @ self.W.T
        return np.reshape(r,(T_tot*N,),'F')

    def jacobian_next_allow_autapse(self):
        N = self.N
        T_tot = self.ntrials * (self.T-1)
        for i in range(N):
            for j in range(N):
                self.Ji[i*T_tot:(i+1)*T_tot,j*N+i] = -self.SRt[:,j]

    def residuals_next_tonic(self,x):
        N = self.N
        T_tot = self.ntrials * (self.T-1)
        xindex = self.inflate(x)
        #for i in range(N):
        #    self.W[:,i] = self.x0[i*N:(i+1)*N]
        #xindex = N**2
        self.B[:] = self.x0[xindex:xindex+N]
        r = self.dR - self.SRt @ self.W.T - self.B
        return np.reshape(r,(T_tot*N,),'F')

    def jacobian_next_allow_autapse_tonic(self):
        N = self.N
        T_tot = self.ntrials * (self.T-1)
        for i in range(N):
            for j in range(N):
                self.Ji[i*T_tot:(i+1)*T_tot,j*N+i] = -self.SRt[:,j]
            self.Ji[i*T_tot:(i+1)*T_tot,N**2+i] = -1

    def jacobian_next_no_autapse(self):
        N = self.N
        T_tot = self.ntrials * (self.T-1)
        for j in range(N):
            for i in range(j):
                self.Ji[i*T_tot:(i+1)*T_tot,j*(N-1)+i] = -self.SRt[:,j]
            for i in range(j+1,N):
                self.Ji[i*T_tot:(i+1)*T_tot,j*(N-1)+i-1] = -self.SRt[:,j]

    def jacobian_next_no_autapse_tonic(self):
        N = self.N
        T_tot = self.ntrials * (self.T-1)
        for j in range(N):
            for i in range(j):
                self.Ji[i*T_tot:(i+1)*T_tot,j*(N-1)+i] = -self.SRt[:,j]
            for i in range(j+1,N):
                self.Ji[i*T_tot:(i+1)*T_tot,j*(N-1)+i-1] = -self.SRt[:,j]
            self.Ji[i*T_tot:(i+1)*T_tot,N*(N-1)+i] = -1

    def residuals_full(self,x):
        T = self.T
        S = self.ntrials
        N = self.N
        xindex = self.inflate(x)
        #for i in range(N):
        #    self.W[:,i] = self.x0[i*N:(i+1)*N]
        #xindex = N**2
        for j in range(S):
            self.WS[j*N:(j+1)*N,j*N:(j+1)*N] = self.W[:,:]
        r = np.zeros(N*T*S)
        self.yi[:S*N] = x[xindex:xindex+N*S]
        r[:S*N] = self.Rt_flat[:S*N] - self.yi[:S*N]
        self.syi[:S*N] = self.synaptic_nonlinearity(self.yi[:S*N])
        self.sypi[:S*N] = self.synaptic_nonlinearity_prime(self.yi[:S*N])
        for k in range(1,int(T)):
            self.yi[k*S*N:(k+1)*S*N] = self.rate_nonlinearity((1 - self.dt_tau) * self.yi[(k-1)*S*N:k*S*N] + self.dt_tau * self.WS @ self.syi[(k-1)*S*N:k*S*N])
            self.syi[k*S*N:(k+1)*S*N] = self.synaptic_nonlinearity(self.yi[k*S*N:(k+1)*S*N])
            self.sypi[k*S*N:(k+1)*S*N] = self.synaptic_nonlinearity_prime(self.yi[k*S*N:(k+1)*S*N])
            r[k*S*N:(k+1)*S*N] = self.Rt_flat[k*S*N:(k+1)*S*N] - self.yi[k*S*N:(k+1)*S*N]
        #for k in range(10):
        #    #print(k,r[k*S*N:(k+1)*S*N])
        #    print(k,X_pred[0,k,:])
        return r

    def jacobian_full_allow_autapse(self):
        S = self.ntrials
        T = self.T
        N = self.N

        self.Ji[:S*N,-S*N:] = -np.identity(S*N)
        syi2 = np.reshape(self.syi,(S*T,N))
        for i in range(N):
            for l in range(S):
                self.Ji[(S+l)*N+i::S*N,i:N**2+i:N] = -self.dt_tau*syi2[l:((T-1)*S+l):S,:]
        for k in range(1,T):
            self.Ji[k*S*N:(k+1)*S*N,:] += (1-self.dt_tau) * self.Ji[(k-1)*S*N:k*S*N,:] + self.dt_tau * self.WS @ (self.sypi[(k-1)*S*N:k*S*N,None] * self.Ji[(k-1)*S*N:k*S*N,:])
            yhat = (1 - self.dt_tau) * self.yi[(k-1)*S*N:k*S*N] + self.dt_tau * self.WS @ self.syi[(k-1)*S*N:k*S*N]
            self.Ji[k*S*N:(k+1)*S*N,:] = self.Ji[k*S*N:(k+1)*S*N,:] * self.rate_nonlinearity_prime(yhat)[:,None]
        return

    def residuals_full_tonic(self,x):
        T = self.T
        S = self.ntrials
        N = self.N
        xindex = self.inflate(x)
        #for i in range(N):
        #    self.W[:,i] = self.x0[i*N:(i+1)*N]
        #xindex = N**2
        self.B[:] = x[xindex:xindex+N]
        xindex += N
        for j in range(S):
            self.WS[j*N:(j+1)*N,j*N:(j+1)*N] = self.W[:,:]
            self.BS[j*N:(j+1)*N] = self.B[:]
        r = np.zeros(N*T*S)
        self.yi[:S*N] = x[xindex:xindex+N*S]
        r[:S*N] = self.Rt_flat[:S*N] - self.yi[:S*N]
        self.syi[:S*N] = self.synaptic_nonlinearity(self.yi[:S*N])
        self.sypi[:S*N] = self.synaptic_nonlinearity_prime(self.yi[:S*N])
        for k in range(1,int(T)):
            self.yi[k*S*N:(k+1)*S*N] = self.rate_nonlinearity((1 - self.dt_tau) * self.yi[(k-1)*S*N:k*S*N] + self.dt_tau * self.WS @ self.syi[(k-1)*S*N:k*S*N] + self.dt_tau * self.BS)
            self.syi[k*S*N:(k+1)*S*N] = self.synaptic_nonlinearity(self.yi[k*S*N:(k+1)*S*N])
            self.sypi[k*S*N:(k+1)*S*N] = self.synaptic_nonlinearity_prime(self.yi[k*S*N:(k+1)*S*N])
            r[k*S*N:(k+1)*S*N] = self.Rt_flat[k*S*N:(k+1)*S*N] - self.yi[k*S*N:(k+1)*S*N]
        return r

    def jacobian_full_allow_autapse_tonic(self):
        S = self.ntrials
        T = self.T
        N = self.N

        self.Ji[:S*N,-S*N:] = -np.identity(S*N)
        syi2 = np.reshape(self.syi,(S*T,N))
        for i in range(N):
            for l in range(S):
                self.Ji[(S+l)*N+i::S*N,i:N**2+i:N] = -self.dt_tau*syi2[l:((T-1)*S+l):S,:]
            self.Ji[S*N+i::N,N**2+i] = -self.dt_tau
        for k in range(1,T):
            self.Ji[k*S*N:(k+1)*S*N,:] += (1-self.dt_tau) * self.Ji[(k-1)*S*N:k*S*N,:] + self.dt_tau * self.WS @ (self.sypi[(k-1)*S*N:k*S*N,None] * self.Ji[(k-1)*S*N:k*S*N,:])
            yhat = (1 - self.dt_tau) * self.yi[(k-1)*S*N:k*S*N] + self.dt_tau * self.WS @ self.syi[(k-1)*S*N:k*S*N] + self.dt_tau * self.BS
            self.Ji[k*S*N:(k+1)*S*N,:] = self.Ji[k*S*N:(k+1)*S*N,:] * self.rate_nonlinearity_prime(yhat)[:,None]
        return

    def jacobian_full_no_autapse(self):
        S = self.ntrials
        T = self.T
        N = self.N

        #for i in range(N):
        #    for l in range(S):
        #        Ji[l*N+i:(T*S+l)*N+i:S*N,(N-1+l)*N:(N+l)*N] = -M_series[:,i,:]
        #        for j in range(i):
        #            Ji[(S+l)*N+i::S*N,j*(N-1)+i-1] = -dt_tau*yi[l*N+j:((T-1)*S+l)*N+j:S*N]*dx0dxi[j*(N-1)+i-1]
        #        for j in range(i+1,N):
        #            Ji[(S+l)*N+i::S*N,j*(N-1)+i] = -dt_tau*yi[l*N+j:((T-1)*S+l)*N+j:S*N]*dx0dxi[j*(N-1)+i]
        #for k in range(2,T):
        #    Ji[k*S*N:(k+1)*S*N,:(N-1)*N] += np.dot(MS,Ji[(k-1)*S*N:k*S*N,:(N-1)*N])

        self.Ji[:S*N,-S*N:] = -np.identity(S*N)
        #syi2 = np.reshape(self.syi,(S*T,N))
        for i in range(N):
            for j in range(i):
                self.Ji[S*N+i::N,j*(N-1)+i-1] = -self.dt_tau*self.syi[j:(T-1)*S*N+j:N]
            for j in range(i+1,N):
                self.Ji[S*N+i::N,j*(N-1)+i] = -self.dt_tau*self.syi[j:(T-1)*S*N+j:N]
            #for l in range(S):
            #    self.Ji[(S+l)*N+i::S*N,i-1:i*(N-1)+i-1:N-1] = -self.dt_tau*syi2[l:((T-1)*S+l):S,:i]
            #    self.Ji[(S+l)*N+i::S*N,(i+1)*(N-1)+i:N*(N-1)+i:N-1] = -self.dt_tau*syi2[l:((T-1)*S+l):S,i+1:]
        for k in range(1,T):
            self.Ji[k*S*N:(k+1)*S*N,:] += (1-self.dt_tau) * self.Ji[(k-1)*S*N:k*S*N,:] + self.dt_tau * self.WS @ (self.sypi[(k-1)*S*N:k*S*N,None] * self.Ji[(k-1)*S*N:k*S*N,:])
            yhat = (1 - self.dt_tau) * self.yi[(k-1)*S*N:k*S*N] + self.dt_tau * self.WS @ self.syi[(k-1)*S*N:k*S*N]
            self.Ji[k*S*N:(k+1)*S*N,:] = self.Ji[k*S*N:(k+1)*S*N,:] * self.rate_nonlinearity_prime(yhat)[:,None]
        return

    def jacobian_full_no_autapse_tonic(self):
        S = self.ntrials
        T = self.T
        N = self.N

        self.Ji[:S*N,-S*N:] = -np.identity(S*N)
        syi2 = np.reshape(self.syi,(S*T,N))
        for i in range(N):
            for j in range(i):
                self.Ji[S*N+i::N,j*(N-1)+i-1] = -self.dt_tau*self.syi[j:(T-1)*S*N+j:N]
            for j in range(i+1,N):
                self.Ji[S*N+i::N,j*(N-1)+i] = -self.dt_tau*self.syi[j:(T-1)*S*N+j:N]
            #for l in range(S):
            #    self.Ji[(S+l)*N+i::S*N,i-1:i*(N-1)+i-1:N-1] = -self.dt_tau*syi2[l:((T-1)*S+l):S,:i]
            #    self.Ji[(S+l)*N+i::S*N,(i+1)*(N-1)+i:N*(N-1)+i:N-1] = -self.dt_tau*syi2[l:((T-1)*S+l):S,i+1:]
            self.Ji[S*N+i::N,N*(N-1)+i] = -self.dt_tau
        for k in range(1,T):
            self.Ji[k*S*N:(k+1)*S*N,:] += (1-self.dt_tau) * self.Ji[(k-1)*S*N:k*S*N,:] + self.dt_tau * self.WS @ (self.sypi[(k-1)*S*N:k*S*N,None] * self.Ji[(k-1)*S*N:k*S*N,:])
            yhat = (1 - self.dt_tau) * self.yi[(k-1)*S*N:k*S*N] + self.dt_tau * self.WS @ self.syi[(k-1)*S*N:k*S*N] + self.dt_tau * self.BS
            self.Ji[k*S*N:(k+1)*S*N,:] = self.Ji[k*S*N:(k+1)*S*N,:] * self.rate_nonlinearity_prime(yhat)[:,None]
        return

    ##############################
    ##### CLMA Fit Functions #####
    ##############################

    def fit_clma(self,eta=0.0005,iteration_limit_override=-1):
        '''doc string'''
        S = self.ntrials
        T = self.T
        N = self.N
        '''enforce consistency of initial guess with bounds'''
        nx = np.shape(self.x0)[0]
        
        for i in range(nx):
            buffer = 0.00001# * (self.upper_bounds[i] - self.lower_bounds[i])
            if self.x0[i] - self.lower_bounds[i] < buffer:
                self.x0[i] = self.lower_bounds[i] + buffer
            if self.upper_bounds[i] - self.x0[i] < buffer:
                self.x0[i] = self.upper_bounds[i] - buffer
            if self.upper_bounds[i] - self.lower_bounds[i] < 2e-10:
                self.x0[i] = (self.upper_bounds[i] + self.lower_bounds[i]) / 2
        reg_types = list(self.regularizations.keys())
        '''calculate initial residuals and cost'''
        avg_tres = 0
        tstart = time.monotonic()
        ri = self.residuals(self.x0)
        avg_tres += time.monotonic() - tstart
        Ci_data = 0.5*np.sum(np.multiply(ri,ri))
        Ci_reg  = 0
        sf_flat = np.ones(nx)
        nuc = None
        if self.allow_autapse:
            for icol in range(N):
                sf_flat[icol*N:(icol+1)*N] = self.scale_factor[icol] / self.scale_factor
        else:
            for icol in range(N):
                sf_flat[icol*(N-1):(icol+1)*(N-1)] = np.delete(self.scale_factor[icol] / self.scale_factor,icol,axis=0)
        if 'L1' in reg_types:
            Ci_reg += self.regularizations['L1'] * np.sum(self.penalty_flat * sf_flat * abs(self.x0 - self.prior_flat))
        if 'L2' in reg_types:
            Ci_reg += self.regularizations['L2'] * 0.5 * np.sum(self.penalty_flat * sf_flat**2 * (self.x0 - self.prior_flat)**2)
        if 'NUC' in reg_types:
            nuc = np.linalg.norm(self.W,'nuc')
            Ci_reg += self.regularizations['NUC'] * nuc #np.sum(s)
        Ci = Ci_data + Ci_reg
        self.verbalize(f'CLMA - Starting Cost (Data, Reg, Tot): ({Ci_data}, {Ci_reg}, {Ci})',0)
        '''set configuration'''
        jlim = self.clma_max_iterations
        if iteration_limit_override > 0:
            jlim = iteration_limit_override
        maxD = 0 #if greater than 0, allows for some cost increasing steps to get out of local minima
        avg_tjac = 0
        avg_tstep = 0
        avg_tinv = 0
        '''initialize iterated variables'''
        xi = np.log(np.divide(self.x0 - self.lower_bounds,self.upper_bounds - self.x0)) #transform to project bounds to (-inf,+inf)
        exi = np.exp(xi)
        exip1 = exi + 1
        xip1 = np.zeros(nx)
        vi = np.zeros(nx)
        vip1 = np.zeros(nx)
        vnorm = 1
        vrel = 1
        rip1 = 0
        nr = 1 #number of times residuals have been calculated
        ninv = 0
        Cip1 = Ci + 1
        dC = 1 #change in cost from most recent iteration
        D = maxD
        self.Ji = np.zeros((np.shape(ri)[0],nx))
        pen_term = np.zeros(nx)
        nJ = 0 #number of times Jacobian has been calculated
        I = np.identity(nx)

        '''main loop'''
        while (vrel > self.clma_min_vrel and (dC < 0 or dC > 1e-4) and nJ < jlim) or nJ < self.clma_min_iterations:
            tstart = time.monotonic()
            self.verbalize(f'iteration: {nJ}',99)

            '''caculate Jacobian'''
            dx0dxi = np.divide(np.multiply(self.upper_bounds-self.lower_bounds,exi),np.multiply(exip1,exip1))
            self.Ji = 0 * self.Ji
            self.jacobian()
            self.Ji = np.multiply(self.Ji,dx0dxi)
        
            tjac = time.monotonic() - tstart
            nJ += 1
            JT = np.transpose(self.Ji)
        
            '''calculate regularization gradients'''
            pen_term = 0 * pen_term
            if 'L1' in reg_types:
                pen_term += self.regularizations['L1'] * self.penalty_flat * sf_flat * np.sign(self.x0 - self.prior_flat)
            if 'L2' in reg_types:
                pen_term += self.regularizations['L2'] * self.penalty_flat * sf_flat**2 * (self.x0 - self.prior_flat)
            if 'NUC' in reg_types:
                pen_term = self.add_nuclear_norm_gradient(nuc,pen_term)
            pen_term *= dx0dxi
        
            g = np.dot(JT,self.Ji) + np.outer(pen_term,pen_term)

            '''search for good step'''
            threshold = np.sqrt(2 * Ci / (T*S*N))
            while (1-D)*Cip1 > Ci and vrel > self.clma_min_vrel:
                eta = eta * 2
                ginv = 0
                tstart = time.monotonic()
                try:
                    ginv = np.linalg.pinv(g + eta*I) #weighted combination of gradient descent and Newton's method
                except:
                    print('LM step failed, taking gradient descent step')
                    ginv = (1.0/eta)*I
                avg_tinv += time.monotonic() - tstart
                ninv += 1
                '''compute step velocity'''
                vip1 = -np.dot(ginv,np.dot(JT,ri) + pen_term)
                vnorm = np.linalg.norm(vip1)
                '''compute directional 2nd derivative for step in velocity direction to get step acceleration'''
                onent = xi+vip1/10
                if np.max(abs(onent)) > 200:
                    continue 
                ez = np.exp(onent)
                z = np.divide(self.lower_bounds + self.upper_bounds*ez,ez+1)
                tstart = time.monotonic()
                rstep = self.residuals(z)
                avg_tres += time.monotonic() - tstart
                vvHi = 2*(rstep - ri - np.dot(self.Ji,vip1/10))*10**2
                nr += 1
                a = -np.dot(np.dot(ginv,JT),vvHi)
                anorm = np.linalg.norm(a)
                '''compare velocity magnitude current size of fitted parameters'''
                vrel = vnorm/(math.sqrt(nx)*np.mean(np.abs(xi)))
                '''
                compare acceleration and velocity
                if acceleration is large relative to velocity and velocity is large, try again with more weight on gradient descent
                if velocity is small but acceleration is still large relative to velocity, just step along velocity vector
                otherwise, step is a combination of velocity and acceleration
                ''' 
                step = vip1
                if 2*anorm > 0.75*vnorm:
                    if vrel > self.clma_max_vrel:
                        continue
                else:
                    step = vip1 + 0.5*a
                '''make the step'''
                xip1 = xi + step
                #print(np.mean(abs(xi)),np.mean(abs(xip1)))
                if np.max(abs(xip1)) > 200:
                    continue
                exi = np.exp(xip1)
                exip1 = exi + 1
                '''transform back to original coordinates to calculate residuals'''
                self.x0 = np.divide(np.add(self.lower_bounds,np.multiply(self.upper_bounds,exi)),exip1)
                tstart = time.monotonic()
                rip1 = self.residuals(self.x0)
                avg_tres += time.monotonic() - tstart
                nr += 1
                '''calculate updated costs'''
                if np.mean(rip1) > threshold:# or np.std(rip1) > threshold:
                    if self.verbosity > 99:
                        # redudant check, but don't want to construct this string unless necessary 
                        self.verbalize(f'eta: {eta}, pre-cost: {Ci}, post-cost: {Cip1}, change: {(Ci - Cip1)/Ci}, vrel: {vrel}, anorm: {anorm}, vnorm: {vnorm}, step size: {np.linalg.norm(rstep-ri)}',99)
                    continue
                Ci_data = 0.5*np.dot(rip1,rip1)
                Ci_reg  = 0
                if 'L1' in reg_types:
                    Ci_reg += self.regularizations['L1'] * np.sum(self.penalty_flat * sf_flat * abs(self.x0 - self.prior_flat))
                if 'L2' in reg_types:
                    Ci_reg += 0.5 * self.regularizations['L2'] * np.sum(self.penalty_flat * sf_flat**2 * (self.x0 - self.prior_flat)**2)
                if 'NUC' in reg_types:
                    u,s,vt = np.linalg.svd(self.W)
                    nuc = np.linalg.norm(self.W,'nuc')
                    Ci_reg += self.regularizations['NUC'] * nuc #np.sum(s)
                Cip1 = Ci_data + Ci_reg
                '''update tolerance for cost increases
                when velocity direction is changing a lot between iterations,
                cost-increasing steps steps should be rejected'''
                D = max(0,min(maxD,np.dot(vi,vip1/vnorm)))
                if self.verbosity > 99:
                    # redudant check, but don't want to construct this string unless necessary
                    self.verbalize(f'eta: {eta}, pre-cost: {Ci}, post-cost: {Cip1}, change: {(Ci - Cip1)/Ci}, vrel: {vrel}, anorm: {anorm}, vnorm: {vnorm}, step size: {np.linalg.norm(rstep-ri)}',99)
            '''if the step produced an unacceptable change in cost, roll it back'''
            if (1-D)*Cip1 > Ci:
                exi = np.exp(xi)
                self.x0 = np.divide(np.add(self.lower_bounds,np.multiply(self.upper_bounds,exi)),exi+1)
                Ci_data = 0.5*np.dot(ri,ri)
                Ci_reg  = 0
                if 'L1' in reg_types:
                    Ci_reg += self.regularizations['L1'] * np.sum(self.penalty_flat * sf_flat * abs(self.x0 - self.prior_flat))
                if 'L2' in reg_types:
                    Ci_reg += 0.5 * self.regularizations['L2'] * np.sum(self.penalty_flat * sf_flat**2 * (self.x0 - self.prior_flat)**2)
                if 'NUC' in reg_types:
                    self.inflate(self.x0)
                    nuc = np.linalg.norm(self.W,'nuc')
                    Ci_reg += self.regularizations['NUC'] * nuc
                Cip1 = Ci_data + Ci_reg
                continue
            '''update iterated variables'''
            xi = xip1
            xip1 = []
            ri = rip1
            rip1 = []
            vi = vip1 / vnorm
            vip1 = []
            dC = (Ci - Cip1)/Ci
            Ci = Cip1
            Cip1 += 1
            '''update config'''
            maxD = max(0,maxD-0.02) #at each iteration, reduce tolerance for cost increases
            D = maxD
            eta = eta / 10 #after each successful step, shift weighting back toward Newton's method
            '''record time performance'''
            tstep = time.monotonic() - tstart - tjac
            avg_tjac += tjac
            avg_tstep += tstep
        self.verbalize(f'CLMA - Final Cost (Data, Reg, Tot): ({Ci_data}, {Ci_reg}, {Ci})',0)
        self.verbalize(f'Jacobian Evaluations: {nJ} ({avg_tjac/nJ} s/jacobian)',1)
        self.verbalize(f'Residual Evaluations: {nr} ({avg_tres/nr} s/jacobian)',1)
        self.verbalize(f'Matrix Inversions: {ninv} ({avg_tinv/ninv} s/jacobian)',1)
        return Ci

    def generate_prediction(self,r0,T):
        X_pred = np.zeros((T,self.N))
        X_pred[0,:] = r0[:]
        for k in range(1,T):
            SX = self.synaptic_nonlinearity(X_pred[k-1,:])
            X_pred[k,:] = self.rate_nonlinearity((1-self.dt_tau) * X_pred[k-1,:] + self.dt_tau * self.W @ SX + self.dt_tau * self.B)
        return X_pred

    def get_r2(self):
        X_pred = np.zeros((self.ntrials,self.T,self.N))
        X_pred[:,0,:] = self.r0s[:,:]
        X = np.zeros((self.ntrials,self.T,self.N))
        for i in range(self.ntrials):
            X[i,:,:] = self.train_rates[i,self.start_index:,:]
            for k in range(1,self.T):
                SX = self.synaptic_nonlinearity(X_pred[i,k-1,:])
                X_pred[i,k,:] = self.rate_nonlinearity((1-self.dt_tau) * X_pred[i,k-1,:] + self.dt_tau * self.W @ SX + self.dt_tau * self.B)
        X_mean = np.mean(X,axis=1)
        if self.r2_method == 'avg':
            r2 = 1 - np.sum((X - X_pred)**2,axis=(0,1))/np.sum((X - X_mean[:,None,:])**2,axis=(0,1))
            return np.mean(r2)
        return 1 - np.sum((X - X_pred)**2)/np.sum((X - X_mean[:,None,:])**2)

    def inflate_include_autapse(self,x):
        n = self.N**2
        self.W = np.reshape(x[:n],(self.N,self.N),'F')
        return n

    def inflate_no_autapse(self,x):
        n = self.N * (self.N-1)
        for i in range(self.N):
            self.W[:i,i] = x[i*(self.N-1):i*(self.N-1)+i]
            self.W[i+1:,i] = x[i*(self.N-1)+i:(i+1)*(self.N-1)]
        return n

    def add_nuclear_norm_gradient_allow_autapse(self,nuc,pen_term):
        for i in range(self.N):
            for j in range(self.N):
                self.W[i,j] += 1e-5
                pen_term[j*self.N+i] += self.regularizations['NUC'] * (np.linalg.norm(self.W,'nuc') - nuc)/1e-5
                self.W[i,j] -= 1e-5
        return pen_term

    def add_nuclear_norm_gradient_no_autapse(self,nuc,pen_term):
        for j in range(self.N):
            for i in range(j):
                self.W[i,j] += 1e-5
                pen_term[j*(self.N-1)+i] += self.regularizations['NUC'] * (np.linalg.norm(self.W,'nuc') - nuc)/1e-5
                self.W[i,j] -= 1e-5
            for i in range(j+1,self.N):
                self.W[i,j] += 1e-5
                pen_term[j*(self.N-1)+i-1] += self.regularizations['NUC'] * (np.linalg.norm(self.W,'nuc') - nuc)/1e-5
                self.W[i,j] -= 1e-5
        return pen_term
