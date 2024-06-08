import sys
sys.path.insert(1, '../src/')
from ZFit import *

confile = sys.argv[1]
#infile = sys.argv[2]
#outfile = sys.argv[3]

#fit = ZFit(confile)
#fit.load(infile)
#fit.fit()
#fit.save(outfile)

'''
for i in range(4):
    fit = ZFit(confile)
    fit.load(f'real/fish{i+1}.fitdata')
    fit.fit()
    fit.save(f'real/fits/fish{i+1}.nuc.fit')
'''

for i in range(5):
    for j in range(2,5):
        fit = ZFit(confile)
        fit.include_stims = False
        fit.load(f'allexc/w{j}bd_25.{i}.2stim.fitdata')
        fit.fit()
        fit.save(f'allexc/fits/w{j}bd_25.{i}.0stim.nuc.fit')
        fit = ZFit(confile)
        fit.load(f'allexc/w{j}bd_25.{i}.2stim.fitdata')
        fit.fit()
        fit.save(f'allexc/fits/w{j}bd_25.{i}.2stim.nuc.fit')
        fit = ZFit(confile)
        fit.load(f'allexc/w{j}bd_25.{i}.5stim.fitdata')
        fit.fit()
        fit.save(f'allexc/fits/w{j}bd_25.{i}.5stim.nuc.fit')

for i in range(5):
    for j in range(1,6):
        fit = ZFit(confile)
        fit.obey_dale = False
        fit.allow_inhibition = True
        fit.include_stims = False
        fit.load(f'free/w{j}o_25.{i}.2stim.fitdata')
        fit.fit()
        fit.save(f'free/fits/w{j}o_25.{i}.0stim.nuc.fit')
        fit = ZFit(confile)
        fit.obey_dale = False
        fit.allow_inhibition = True
        fit.load(f'free/w{j}o_25.{i}.2stim.fitdata')
        fit.fit()
        fit.save(f'free/fits/w{j}o_25.{i}.2stim.nuc.fit')
        fit = ZFit(confile)
        fit.obey_dale = False
        fit.allow_inhibition = True
        fit.load(f'free/w{j}o_25.{i}.5stim.fitdata')
        fit.fit()
        fit.save(f'free/fits/w{j}o_25.{i}.5stim.nuc.fit')

for i in range(5):
    for j in range(1,6):
        fit = ZFit(confile)
        fit.obey_dale = False
        fit.allow_inhibition = True
        fit.include_stims = False
        fit.load(f'free/w{j}u_25.{i}.2stim.fitdata')
        fit.fit()
        fit.save(f'free/fits/w{j}u_25.{i}.0stim.nuc.fit')
        fit = ZFit(confile)
        fit.obey_dale = False
        fit.allow_inhibition = True
        fit.load(f'free/w{j}u_25.{i}.2stim.fitdata')
        fit.fit()
        fit.save(f'free/fits/w{j}u_25.{i}.2stim.nuc.fit')
        fit = ZFit(confile)
        fit.obey_dale = False
        fit.allow_inhibition = True
        fit.load(f'free/w{j}u_25.{i}.5stim.fitdata')
        fit.fit()
        fit.save(f'free/fits/w{j}u_25.{i}.5stim.nuc.fit')

for i in range(5):
    for j in range(1,2):
        fit = ZFit(confile)
        fit.include_stims = False
        fit.load(f'allexc/w{j}c_25.{i}.2stim.fitdata')
        fit.fit()
        fit.save(f'allexc/fits/w{j}c_25.{i}.0stim.nuc.fit')
        fit = ZFit(confile)
        fit.load(f'allexc/w{j}c_25.{i}.2stim.fitdata')
        fit.fit()
        fit.save(f'allexc/fits/w{j}c_25.{i}.2stim.nuc.fit')
        fit = ZFit(confile)
        fit.load(f'allexc/w{j}c_25.{i}.5stim.fitdata')
        fit.fit()
        fit.save(f'allexc/fits/w{j}c_25.{i}.5stim.nuc.fit')
