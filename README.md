# Bachelor's thesis, Magnus Edstrand

Github corresponding to the bachelor's thesis "Simulating from characteristic functions: Learning distributions from their characteristic functions using the theory of Maximum Mean Discrepancy" by Magnus Edstrand

To run an evaluation of a model, for example, an evaluation of the 2-dimensional moderately-tailed 6,000-epoch model, run:

python evaluate.py saved_models/2dim/2d_moderate_6000e.pth

Note that, in order to change whether the loss-graph plots every epoch or every 40 (used resp. to plot graphs for 200epochs and 6,000epochs), one has to comment out the line they don't want to use in plot_loss() in evaluate.py. There are comments within on this.

To run a new training on a generalized hyperbolic distribution, the main_ghd.py file can be used. Update the parameters as you wish, and make sure \alpha^2 > \beta^\top \Lambda \beta is fulfilled. The training may take quite a while, depending on the used GPU and the number of iterations.

Of course, the original code provided by Florian Brück, which has only been modified such as to be able to run GHD-distributions, is found at https://github.com/florianbrueck/charfctgen
