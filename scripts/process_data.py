import sys
sys.path.insert(1, '../src/')
from ZData import *
import scipy.io

confile = sys.argv[1]
datfile = sys.argv[2]

# load experimental data
raw_data = scipy.io.loadmat(datfile)

# initialize data objects for each fish
fish_data = []
for i in range(4):
    fish_data.append(ZData(confile)) 

# indices for accessing each imaging plane of each fish
# 18 total planes: 4 in fish 1, 5 in fish 2, 4 in fish 3, and 5 in fish 4
fishID = [0,0,0,0,1,1,1,1,1,2,2,2,2,3,3,3,3,3]
planeID = [0,1,2,3,0,1,2,3,5,0,1,2,3,0,1,2,3,4]

# these are the cells in each plane that passed the selection criteria set by Kayvon long ago
# to able to make exact comparisons to previous fitting results, run with backward_compatible_selection = True
# setting this to False will enable applying fresh selection criteria
backward_compatibility_whitelist = [[28],[0,3,5,6,7,9,10,13,14,16,17,20,21],[0,2,5,23],[0],[0,1,2,4],[0,1,2,3,5],[3,5,8],[1,2,4,5,6,10,16,24],[0,1,2],[0,1,3,4,8,10,18],[0,1,2,3,4,6,8,9,10,13,14,16],[0,1,2,3,5,6],[0,6],[0,1,2,6,8],[8,16,18],[1,2,3,4,5],[0,1,3,8,11],[0,1,2]]
backward_compatible_selection = True

N_passed_tot = 0
N_tot = 0
for p in range(18): # loop over imaging planes
    plane_data = ZData(confile)
    stim1_data = raw_data['fish'][0][fishID[p]][0][0][0][0][0][planeID[p]] # plane data for stim in rostral position
    stim2_data = raw_data['fish'][0][fishID[p]][0][0][1][0][0][planeID[p]] # plane data for stim in caudal position
    # collect all fluorescence traces for saccades with stim off, rostral stim on, and caudal stim on, in that order
    # result is (T, N_cells, N_saccades) array
    plane_data.measurements = np.concatenate([stim1_data[20],stim2_data[20],stim1_data[21],stim2_data[21]],axis=2)
    plane_data.ntrials[0] = np.shape(stim1_data[20])[2] + np.shape(stim2_data[20])[2] # N_saccades with stim off
    plane_data.ntrials[1] = np.shape(stim1_data[21])[2] # N_saccades with rostral stim on
    plane_data.ntrials[2] = np.shape(stim2_data[21])[2] # N_saccades with caudal stim on
    plane_data.centroid = raw_data['fish'][0][fishID[p]][0][0][0][0][0][planeID[p]][12]
    avg_fluor = (stim1_data[24] + stim2_data[24]) / 2 # average fluorescence over all saccades with stim off
    corr = stim1_data[19] # correlation of fluorescence with eye position

    if p == 9: # one plane has a couple corrupted entries that need to be removed
        plane_data.measurements = np.delete(plane_data.measurements,[3,8],1)
        plane_data.centroid = np.delete(plane_data.centroid,[3,8],1)
        avg_fluor = np.delete(avg_fluor,[3,8],1)
        corr = np.delete(corr,[3,8],1)

    T,plane_data.N,L = np.shape(plane_data.measurements)
    plane_data.measurement_times = np.outer(np.linspace(0,(T-1)*plane_data.dt_samp,T),np.ones(L)) # data already interpolated to common time points
    plane_data.estimate_firing_rates()
    if backward_compatible_selection:
        plane_data.N = len(backward_compatibility_whitelist[p])
        plane_data.firing_rate_estimate = plane_data.firing_rate_estimate[:,:,backward_compatibility_whitelist[p]]
        plane_data.centroid = plane_data.centroid[:,backward_compatibility_whitelist[p]]
    else:
        N_passed = plane_data.N
        # if applying new selection criteria, the ones currently in use are thresholds on peak fluorescence, peak estimated firing rate, and correlation with eye position
        for i in range(plane_data.N-1,-1,-1):
            p90 = np.percentile(avg_fluor[:,i],90)
            if np.max(corr[:,i]) < 0.2:
                print('Neuron {0:d} failed cor > 0.2'.format(i))
                plane_data.firing_rate_estimate = np.delete(plane_data.firing_rate_estimate,i,2)
                plane_data.centroid = np.delete(plane_data.centroid,i,1)
                N_passed = N_passed - 1
            elif p90 < 0.07:
                print('Neuron {0:d} failed p90 > 0.07'.format(i))
                plane_data.firing_rate_estimate = np.delete(plane_data.firing_rate_estimate,i,2)
                plane_data.centroid = np.delete(plane_data.centroid,i,1)
                N_passed = N_passed - 1
            elif np.max(plane_data.firing_rate_estimate[0,67:,i]) < 0.04:
                print('Neuron {0:d} failed rate > 0.04'.format(i))
                plane_data.firing_rate_estimate = np.delete(plane_data.firing_rate_estimate,i,2)
                plane_data.centroid = np.delete(plane_data.centroid,i,1)
                N_passed = N_passed - 1
            else:
                print('Neuron {0:d} passed'.format(i))
        print('{0:d}/{1:d} neurons in plane {2:d} passed cuts'.format(N_passed,plane_data.N,p))
        N_tot += plane_data.N
        N_passed_tot += N_passed
        plane_data.N = N_passed

    # add data for the cells in the current imaging plane that passed selection criteria to the associated fish
    if fish_data[fishID[p]].firing_rate_estimate is None:
        fish_data[fishID[p]].N = plane_data.N
        fish_data[fishID[p]].firing_rate_estimate = plane_data.firing_rate_estimate
        fish_data[fishID[p]].centroid = plane_data.centroid
    else:
        fish_data[fishID[p]].N += plane_data.N
        fish_data[fishID[p]].firing_rate_estimate = np.concatenate([fish_data[fishID[p]].firing_rate_estimate,plane_data.firing_rate_estimate],axis=2)
        fish_data[fishID[p]].centroid = np.concatenate([fish_data[fishID[p]].centroid,plane_data.centroid],axis=1)

if not backward_compatible_selection:
    print('{0:d}/{1:d} neurons passed cuts'.format(N_passed_tot,N_tot))
for i in range(4):
    print(f'Saving data for {fish_data[i].N} neurons in fish {i+1}')
    fish_data[i].save(f'fish{i+1}.fitdata')