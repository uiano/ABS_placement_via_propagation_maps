# aerial_base_stations
Simulations for the paper "Aerial Base Station Placement via Propagation Radio Maps" by Daniel Romero, Pham Q. Viet, and Raju Shrestha.

After cloning the repository, do the following:

```
cd gsim
git submodule init
git submodule update
cd ..
bash gsim/install.sh

cd common
python grid_utilities_setup.py build
python grid_utilities_setup.py install
cd ..
```
You may need to install a compiler and some Python packages.

To run the simulations, type

```
python run_experiment.py M
```

where M is 1003 for Fig. 3, 1004 for Fig. 4, etc. 

The code of the experiments can be found in experiments/jpaper_experiments.py. 

More information on the simulation environment [here](https://github.com/fachu000/GSim-Python).

