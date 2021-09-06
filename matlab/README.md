# State-space deep Gaussian processes in Matlab

This folder contains the Matlab scripts that used in Zhao et al. (2020). In this folder, you may find
the following runnable files that you can start with:

1. `./composite_sine_EKFS.m`. Use extended Kalman filters and smoothers to solve SS-DGP regression problems with data generated from a composite sinusoidal signal.

2. `./composite_sine_GFS.m`. Use (cubature) Gaussian filters and smoothers to solve SS-DGP regression problems with data generated from a composite sinusoidal signal.

3. `./rectangle_SSMAP.m`. Use a state-space MAP method to solve SS-DGP regression problems with data generated from a rectangular signal.

# Installation

No installation is required. Just git clone or download the codes to your local, and run the scripts in `./`.

# How to use?

The implementation here, to certain extent, allows you to generate SS-DGP regression models automatically with desired DGP structures. But this also in turn makes the computation slower. In the following, we show you how to perform SS-DGP regression on a bootstrap DGP model of the form

```math
\begin{align*}
    f &\sim GP\big(0, C_f(t, t'; \ell(t))\big),\\
    u^2_1 &\sim GP\big(0, C_{\mathrm{Mat.}}(t, t')\big),\\
    \ell(t) &= g(u^2_1(t)),
\end{align*}
where $g(u) = \exp(u)$.
```

```matlab
% First, let's construct the SS-DGP regression model above
% Transformation function be g(u) = exp(u)
g = "exp";

% Construct the first DGP node f
f = dgp.DGPNode('f', [], g, 'f', @gp.matern32_ns, [1 1.6]);

% Construct the second DGP node u21 which parametrises the length scale of f.
u21 = dgp.DGPNode(f, 'l', g, 'u21', @gp.matern12, [0.2 1.16]); 

% Now make a DGP instance
my_dgp = dgp.DGP(f);

% Integration and error checks (not really compile).
my_dgp.compile()

% Make an SS-DGP instance based on the DGP instance
% You can also try use "local", "EM", or "TME-order"
ss = ssdgp.SSDGP(my_dgp, "TME-2");

% Load data, here T is a vector of times, y is a vector of measurements, and R is a matrix 
% of measurement covariance.
my_dgp.load_data(T, y, R);

% Perform EKFS regression.
query_times = ... % Times for interpolation if you have any.
[posterior_times, posterior, ~] = filters.SGP_CKFS(ss, query_times, 1);

% The returned posterior is a cell containing
% {filtering_mean, filtering_cov, smooting_mean, smoothing_cov}
posterior
```

You may find a bit confused with the line `u21 = dgp.DGPNode(f, 'l', g, 'u21', @gp.matern12, [0.2 1.16]);`. This means that `u21` is a DGP node named `'u21'` that is a parent of the node `f`, and that `u21` parametrises the length scale (i.e., `'l'`) of `f`. Moreover, the covaraince function of `u21` is Matern 1/2 with parameters `l=0.2` and `sigma=1.16`. The parameters are transformed by the function `g`. Okay, let's go a bit deeper, can you tell me what does the following code block do? 

```matlab
hyper_para = [1, 1];
hyper_para2 = [0.01 0.9];

f = dgp.DGPNode('f', [], g, 'f', @gp.matern32_ns, hyper_para);

u21 = dgp.DGPNode(f, 'l', g, 'u21', @gp.matern12_ns, hyper_para2); 
u22 = dgp.DGPNode(f, 'sigma', g, 'u22', @gp.matern12_ns, hyper_para2);
u31 = dgp.DGPNode(u21, 'l', g, 'u31', @gp.matern12, hyper_para2);
u32 = dgp.DGPNode(u21, 'sigma', g, 'u32', @gp.matern12, hyper_para2);
u33 = dgp.DGPNode(u22, 'l', g, 'u33', @gp.matern12, hyper_para2);
u34 = dgp.DGPNode(u22, 'sigma', g, 'u34', @gp.matern12, hyper_para2);
```

The code block above says `f` is parametrised by nodes `u21` and `u22` in its lengh scale and magnitude parameters, respectively. The length scale and magnitudes of `u21` is parametrised by `u31` and `u32`, respectively. The length scale and magnitudes of `u22` is parametrised by `u33` and `u34`, respectively. 

Note that the names of DGP nodes must be unique. Also, `f` must always come first and be named `'f'`.

# Notes

To be honest, I don't like this Matlab implementation, and I think these codes are crap. The story is that I tried to make the code automatic for generating DGP models, but it turns the implementation to be quite messy and took me a ton of time to test.

Nonetheless, the codes are safe to use, as they are tested. If you use Python, I recommend the reader to check the Python implementation instead.

# Citation

See, `README.md` in the parent folder.

# License

GNU General Public License v3 or later.
