import numpy as np
import graph_tool.all as gt

def runStochasticBlockModel(W,nb):
    N = np.shape(W)[0]
    es = W.T.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(es))
    ew = g.new_edge_property("double")
    ew.a = abs(W.T[es])
    g.ep['eff_weight'] = ew
    SBM = gt.minimize_blockmodel_dl(g,state_args=dict(B=nb,recs=[g.ep["eff_weight"]],rec_types=["real-normal"]),multilevel_mcmc_args=dict(B_min=nb,B_max=nb))
    maxE = SBM.entropy()
    minE = SBM.entropy()
    new_extremum = True
    count = 0
    boredom = 0
    # loop to run several SBMs and keep best
    # keep going until the number of boring runs (no new maximum or minimum) or number of total runs hits limit
    while boredom < 1 and count < 10:
        new_extremum = False
        #print(count,minE)
        # for each run, do 10 SBMs
        for i in range(10):
            test_SBM = gt.minimize_blockmodel_dl(g,state_args=dict(B=nb,recs=[g.ep["eff_weight"]],rec_types=["real-normal"]),multilevel_mcmc_args=dict(B_min=nb,B_max=nb))
            if test_SBM.entropy() > maxE:
                maxE = test_SBM.entropy()
                new_extremum = True
            elif test_SBM.entropy() < minE:
                minE = test_SBM.entropy()
                SBM = test_SBM
                new_extremum = True
        count += 1
        # increment total count, and if no new maximum or minimum, increment boredom count
        if not new_extremum:
            boredom += 1
    #print('Final:',minE)
    # block labels are arbitrary numbers depending on which cell the block started with
    # so need to convert to standard labels running from 0 to nb-1
    blockLabels = np.unique(SBM.get_blocks().a)
    new_sbmid = -np.ones(N)
    for i in range(nb):
        new_sbmid += (i+1)*(SBM.get_blocks().a == blockLabels[i])
    return new_sbmid