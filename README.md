# network-inference
Overview
------------------------------------------------------------
This package is used for fitting neural recording data. To give a high level overview of how it is structured, data is stored in a ZData object. This data can loaded from a .mat file (when processing raw experimental data), from a previously saved ZData object, or generated using a ZSim object (also defined in ZData.py). A ZFit object performs the fit and stores the results. The configuration of each of these objects will be described in detail below and can be edited either directly in python scripts or through human-readable configuration files. The most common usage for a script running several simulations/fits would be to set initial configuration using a configuration file and make any small changes needed for different simulations/fits in the script. The Jupyter notebook StimPaperPlots.ipynb contains useful functions for plotting recording data, weight matrices, and fit results.

Package Directory
------------------------------------------------------------
The contents of this package are organized as follows:

main directory
|--README - you're reading this right now
|--conda_environment.yml - for installing dependencies using conda
|--requirements.txt - for installing dependencies using pip
|--configs
   |--example_sim_config.conf - example configuration file for simulation
   |--example_fit_config.conf - example configuration file for fitting
   |--raw_data_config.conf - configuration file for processing real data
|--data - for housing example data/fits. Unpack example_data.tar.gz to populate this folder with example data
   |--real - holds real data and fits
   |--allexc - holds simulated data and fits for excitatory matrices
   |--dale - holds simulated data and fits for Dale's Law obeying matrices
   |--free - holds simulated data and fits for matrices with no sign constraints
|--notebooks
   |--PlotNotebook.ipynb - Jupyter notebook containing useful plotting functions
|--scripts
   |--process_data.py - loads fluorescence data and generates ZData objects containing firing rate estimates for each fish
   |--example_simulation.py - example script to run simulations
   |--example_fit.py - example script to fit data
   |--example_regularization_scan.py - example script that performs a bunch of fits while varying the regularization
|--src
   |--ZData.py - defines classes for simulating and storing data
   |--ZFit.py - defines classes for fitting data
   |--ZCluster.py - contains method for fitting SBM to weight matrices


Dependencies
------------------------------------------------------------
First, install the latest stable version of python on your machine (exact version doesn't matter, just needs to be 3.x. If using Anaconda, make sure it's Anaconda3). I've collected dependencies into files to make them easy to install. If using Anaconda, just run

conda env create -f conda_environment.yml

to create and environment called venv (you can change this name on the first line of the .yml file) with all of the dependencies installed and run

conda activate env

to start the environment. If not using Anaconda, you can install almost all of the dependencies by running

pip install -r requirements.txt

This can also be done in a virtual environment by first running

python3 -m venv <name>
source <name>/bin/activate

Note, all of the commands provided here are for a bash shell on Linux (and are therefore suitable for lab/campus computing clusters). If using something else, like PowerShell in Windows, some commands will be a little different. The one dependency that can't be installed via pip is graph-tool, so if you're not using Anaconda, go to graph-tool documentation web site (https://graph-tool.skewed.de/installation.html) and follow the installation instructions for your system. Dependency on graph-tool is isolated to ZCluster.py, so all other simulation/fitting can be performed without installing graph-tool.


Running Example Scripts
------------------------------------------------------------
The scripts/ directory contains example scripts for performing the basic functions of this package. The section explains general usage of these scripts and exact commands to run them on the example data. To run these scripts as shown here without any changes, you must have populated the data/ directory by unpacking example_data.tar.gz

process_data.py processes real data to create ZData objects with estimated firing rates to fit. It is specific to the current dataset and will have to be changed for new data provided. From the scripts/ directory, run it as follows:

python3 process_data.py ../configs/raw_data_config.conf ../data/real/all_stim_data.mat

This will produce four files called 'fishX.fitdata' in the data/real/ directory.

example_simulation.py will run some example simulations. It is currently set up to generate a random excitatory matrix with 3 recurrently connected blocks and one readout block and simulate data for that matrix with 10 unperturbed trials and 5 random perturbations with 10 trials each. Examples of other simulation runs are included in the script but currently commented out. Run this script as follows:

General: python3 example_simulation.py config_file output_file (for some examples in this script, the last argument is unnecessary)
Specific: python3 example_simulation.py ../configs/example_sim_config.conf example_sim.fitdata

example_fit.py will run fits on some example simulations or real data. It is currently set up to run one fit on data provided by the users. Examples of other sets of fits corresponding to the other simulation runs in example_simulation.py are included in the script but currently commented out. Run this script as follows:

General: python3 example_file.py config_file data_file output_file (for some examples in this script, the last 2 arguments is unnecessary)
Specific: python3 example_fit.py ../configs/example_fit_config.conf example_sim.fitdata example_fit.fit

example_regularization_scan.py does an example run of several fits with different strengths of L2 regularization on a subset of the example simulation data (currently set up to run on the 2-stim data for the block diagonal + readout matrices). The command to run it is

python3 example_regularization_scan.py ../configs/example_fit_config.conf

The output files of all of these scripts will reside in the data/ directory because it is specified as the base directory in the configuration files. Note, I'm using relative paths here, and each script has a line at the beginning [sys.path.insert(1, '../src/')] that uses a relative path to point to the source directory. I did this so that all the examples will work out of the box in a Linux environment. If it's not working, replace these with absolute paths.


ZData Class - Configuration and Usage
------------------------------------------------------------
To use the ZData class, you first initialize a ZData object with a configuration file and a dtype that is either 'data' or 'sim'

data = ZData(configuration_file_name,dtype='sim')

Initializing with the dtype 'sim' will create a ZSim object with the ZData object to be used for simulating data. The weight matrix for the simulation can either be loaded from a file that is specified in the configuration file or generated randomly using

data.sim.generate_weight_matrix(args)

Refer to example_simulation.py for examples of each type of matrix that can be generated using this function. You can also set the weight matrix however you chose by doing

my_W = your_code_here()
data.sim.load_W(matrix=my_W)

You can also add tonic input to the cicuit with

data.sim.generate_tonic(mean,std) OR
data.sim.I_tonic = my_vector

Where my_vector is some user defined vector. The following code gives each cell a randomly chosen CIRF time constant, simulates the data (config file specifies how many stims and how many trials/stim), and does firing rate estimation from the simulated fluorescence data

data.sim.assign_cirf_taus()
data.sim.simulate(data)
data.estimate_firing_rates()

Finally, save the simulated data with

data.save(output_file_name)

Here is a list of all configurable parameters. The unit for all time parameters is seconds:

n - number of neurons in the simulated weight matrix

sim dt - time step to use for simulation

sample dt - time step to sample from the simulation

interpolated dt - time interval for sampled data to be interpolated to

sim tau - intrinsic cellular time constant for simulation

nstims - number of different stimulations to simulate

ntrials - a list of the number of trials to simulate for unperturbed saccades and each stimulation. The length of the list should be one greater than nstims

saccade period - a two item list specifying the onset time and offset time of the saccade

stim period - a two item list specifying the onset time and offset time of the stimulations

sim duration - total amount of time to simulate for each trial

saccade size - amplitude of saccadic input

stim size - amplitude of stimulation input

stim sign constraint - sets the sign of the stimulations. Can be a single number, which would apply to all neurons, or a list giving a different sign constraint for each neuron. Values should be either -1, 0, or 1. This constraint will apply to all stimulations in this simulation. If this configuration is not included, stimulations will be completely random.

synaptic nonlinearity - nonlinearity to apply to synaptic input. Default is linear. Currently available nonlinearities are saturated linear and relu

rate nonlinearity - nonlinearity to apply to firing rates. Default is linear. Currently available nonlinearities are saturated linear and relu

neuronal noise - amplitude of neuronal noise

measurements noise - amplitude of measurement noise

measurement - specifies what quantity is measured. Can be either 'fluor' or 'rate'

rate estimation - sets method for estimating firing rates. Can be either 'multiexp' to fit a sum of exponentials, '2dpen' to deconvolve with a penalty on the second derivative of the firing rate, or 'none' if firing rate is the measured quantity

average trials - True/False, sets whether to average trials together within each stimulus condition

cirf tau deconv - sets the CIRF time constant to use when estimating firing rates

multiexp sac taus - list of time constants to set the available exponentials for the saccadic response in the multiexp fit

multiexp stim taus - list of time constants to set the available exponentials for the stimulation response in the multiexp fit

multiexp coef pen - if set greater than 0, applies an L1 penalty on the coefficients of the exponentials to encourage the multiexp fit to use the fewest number necessary to fit the fluorescence

2dpen fixation - sets the magnitude of the second derivative penalty for time periods when the circuit is receiving no input

2dpen input - sets the magnitude of the second derivative penalty for time periods when the circuit is receiving saccadic or stimulation input

verbosity - used for debugging. Higher numbers mean print more information. Not likely to be very useful out of the box.

base dir - sets the base directory to use when looking for data files


ZFit Class - Configuration and Usage
------------------------------------------------------------
To use the ZFit class, instantiate an object with a configuration file, load a ZData object either from a file or from memory, run the fit() method, and save the output to a file.

fit = ZFit(configuration_file_name)
fit.load(input_data_file_name) OR fit.load(data=my_zdata_obj)
fit.fit()
fit.save(output_file_name)

That's all there is to it. All of the setup for the fit happens in the configuration file. Here is a list of all configurable parameters:

fit method - can be either 'lsq' to just run a least-squares fit on the next-time-step cost function or 'clma' to run constrained Levenberg-Marquardt with geodesic acceleration on the full time series cost function.

max iter - sets the maximum number of iterations for clma fit

min iter - sets the minimum number of iterations for clma fit

allow autapse - True/False toggle for allowing self connections

allow tonic - True/False toggle for including tonic input in the fit

allow inhibition - True/False toggle for allowing some connections to be inhibitory

obey dale - True/False toggle for when inhibition is allowed that specifies whether to enforce Dale's Law

max inhibition - for a fit with inhibitory connections allowed and Dale's Law enforced, sets the maximum fraction of cells that are allowed to be inhibitory

include stims - True False toggle for whether to include perturbed trials in the fit

regularization - adds a regularization on the weights to the fit. Should be a two-item list consisting of a string to set the type of regularization followed by a number to set the strength. Available regularizations are 'L1', 'L2', and 'NUC'.

prior - string that determines how the regularization is applied to each weight. Default is 'uniform', which will put the same penalty on every weight. Other options have the form 'coordinate-shape' where coordinate is either 'tau', 'rc', or 'ml', and shape is either 'local', 'distal', 'feedforward', or 'feedback'. For instance, 'tau-local' will more heavily penalize connections between cells when the difference in their effective time constants is large, and 'rc-feedforward' will more heavily penalize connections when the presynaptic cell is located closer to the caudal end than the postsynaptic cell. Finally, this string could point to a file containing a weight matrix. In that case, the regularization will be applied uniformly to deviations of the fit matrix from this prior matrix. Note, the priors will not work with a nuclear norm regularization since it is a regularization on eigenvalues, not elements of the weight matrix.

max weight - sets the maximum absolute value allowed for any individual element of the weight matrix

max tonic - sets the maximum absolute value allowed for any individual tonic input

max rate - sets the maximum value allowed for the initial firing rate

min rate - sets the minimum value allowed for the initial firing rate

synaptic nonlinearity - nonlinearity to apply to synaptic input. Default is linear. Currently available nonlinearities are saturated linear and relu

rate nonlinearity - nonlinearity to apply to firing rates. Default is linear. Currently available nonlinearities are saturated linear and relu

tau - intrinsic cellular time constant to use in the fit

start - sets the start time for the fit relative to stimulation offset. Default is 0.5, which starts the fit 0.5 seconds after the time when the stimulation would be turned off.

observed fraction - sets the fraction of cells in the network that are actually observed. These cells are chosen at random, and all information about the remaining cells is discarded.

observed cells - takes a list of indices and sets exactly which cells are observed. This will override the observed fraction configuration

whiten - True/False toggle for whether to whiten the firing rate data before fitting. Will be automatically set back to False if there are nonlinearities in the system

r2method - sets the method to use for calculating an R^2 value for the fit. Default is 'avg', which will calculate the R^2 separately for each cell and report the average of those R^2 values. This is recommended since it makes the R^2 insensitive to any scaling of the firing rates. Setting this to anything other than 'avg' will cause the R^2 to be calculated for all cells together.

verbosity - used for debugging. Higher numbers mean print more information. Not likely to be very useful out of the box.

base dir - sets the base directory to use when looking for data files


Configuration File Syntax
------------------------------------------------------------
All lines in a configuration file should have the form

command = value

I've tried to make the interpreter as forgiving as possible. For instance,

multiexp stim taus = [0.02,2.0,8.0,20]
AND
multiexpSTIMtaus=(0.02 ; 2.0 ; 8.0 ; 20)

will do the same thing. However, after removing white space and making them lowercase, the interpreter is matching the commands to exact strings, so

multiexp stim tau = [0.02,2.0,8.0,20]

would do nothing. To check that your configuration commands are working as intended, include the line

verbosity = x

where x is a number >= 5 at the top of your configuration file (it needs to be at the top so that this gets set before other commands are processed).


Using Jupyter Notebooks
------------------------------------------------------------
To use Jupyter notebooks, change to the notebooks/ directory (not strictly necessary, but nice to keep all teh notebooks in there) and run 

jupyter notebook

This should launch a tab in your browser that shows the contents of the current directory. Click on any notebook to open it. If this tab fails to lauch, look at the output in your console for a URL starting with 'http://localhost:8888' and copy it into your browser. Instructions for using PlotNotebook are included in the notebook.
