import sys
sys.path.insert(1, '../src/')
from ZFit import *

confile = sys.argv[1]
penalty_strengths = [0.001,0.01,0.1,1.0,10.0]

for ps in penalty_strengths:
    for i in range(5):
        for j in range(2,5):
            fit = ZFit(confile)
            fit.regularizations['NUC'] = ps
            fit.load(f'allexc/w{j}bd_25.{i}.2stim.fitdata')
            fit.fit()
            fit.save(f'allexc/fits/reg_scan/w{j}bd_25.{i}.2stim.nuc({ps}).fit')