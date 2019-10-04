'''
The following code was used in part for my masters thesis analyzing the disease Eastern Equine Encephalitis, publication pending.  These files were used to constructed smoothing splines based upon sampled field data, and then to utilize these splines in a SIR Model to simulate infection over the course of the season.  The code can be grouped and described as such.

Spline models
base.py - This file constructs a base spline class to be used by the following files.
host_spline.py - This file allows for the construction of a spline for the sampled host species populations.
bloodmeal_spline.py - This file allows for the construction of a spline for the sampled bloodmeals.

Distribution Types for Spline Models
multinomial.py  - Sets up a multinomial distribution to be used in the bloodmeal spline.
normal.py  -  Sets up a normal distribution for testing purposes.
poisson.py   -  Sets up a poisson distribution to be used in the host spline.

Mosquito Population Type Constructors
mos_constant.py  - This sets the population of the mosquito population to a constant.
mos_curve.py   - This allows for a non-constant callable mosquito population.

Disease Model
seasonal_spline_ode.py  - This file constructs the system of ordinary differential equations we used to simulate the spread
                          of disease over the course of a season.

Helper Functions
util.py            
execute.py

TO DO - add test files and sample cases.
...

