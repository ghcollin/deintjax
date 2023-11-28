Some routines for performing double exponential integration with JAX. 

These routines have been tested against incomplete gamma functions and exponential integrals; 
but, as with all numerical integration routines, you should test convergence yourself before deploying them in your software.

The following strategies have been implemented:

* 1 +/- Tanh-Sinh integration. (Just a more numerically stable Tanh-Sinh integration.)
* Exp-Sinh integration.
* Exp-Exp integration. Note that Exp-Exp integration is highly uneven in the distribution of the abscissa, and should only be employed where Exp-Sinh integration is inappropriate.

Notably, Sinh-Sinh has not been implemented yet. Feel free to submit a contribution if you extend these routines to cover it.

High level interface
====================

A high level interface is provided for both half-infinite interval and closed interval integrals.

The ``deint_halfinfinite_scale`` function performs integration in the interval ``(a, inf)``. 

.. code:: python

    deint_halfinfinite_scale(f, a, s)

It takes three required parameters:

* ``f``: The function to integrate. Should be a function ``R -> R`` defined on the domain ``(a, inf)``.
* ``a``: The lower bound of the integration.
* ``s``: The scale of the integral. The abscissa will be centered around this value. You can default this to 1, or choose another value to get better performance.

The ``deint_interval`` function performs integration in the interval ``(a, b)``.

.. code:: python

    deint_interval(f, a, b)

It takes three required parameters:

* ``f``: The function to integrate. Should be a function ``R -> R`` defined on the domain ``(a, b)``.
* ``a``: The lower bound of the integration.
* ``b``: The upper bound of the integration.

It defaults to Exp-Sinh integration, but you can provide the flag ``strategy='exp_exp'`` to switch to Exp-Exp integration.

For both of these functions, the most numerical precision is achieved near the ``a`` boundary, so place the most difficult part of your integral on that side of the integration interval.

Low level interface
===================

If you need high precision on both sides of the interval, there is a lower-level interface:

.. code:: python

    deint(f_lower, f_upper)

This takes two main parameters of which at least one must be not None:

* ``f_lower``: A function, ``R -> R`` that is called to evaluate the integral for only the lower abscissa.
* ``f_upper``. The same, but for only the upper abscissa.

The domains of these functions depends on the choice of integration strategy.

* 1 +/- Tanh-Sinh: ``f_lower`` has domain ``(-1, 0)`` with abscissa clustered around zero. ``f_upper`` has domain ``(0, 1)`` with abscissa clustered around zero.
* Exp-Sinh and Exp-Exp: ``f_lower`` has domain ``(0, 1)`` with abscissa clustered around zero. ``f_upper`` has domain ``(1, inf)``.

Optional arguments
==================

All functions (lower and higher level) take the following optional arguments:

* ``f_mid``: The value of the integrand at the central abscissa, by default this is evaluated from ``f_lower`` is available, otherwise from ``f_upper``.
* ``tol``: Tolerance for convergence condition. By default, this is ten times the precision of the dtype of ``f_mid`` or 1e-9, whichever is larger.
* ``max_samples``: The maximum number of abscissa to evaluate before terminating. Defaults to 4096.
* ``min_samples``: The minimum number of abscissa to evaluate before the convergence condition has engage. Defaults to 16.
* ``strategy``: Can be one of ``'one_pm_tanh_sinh'``, ``'exp_sinh'`` or ``'exp_exp'``. Defaults to ``'one_pm_tanh_sinh'``.
* ``lamb``: The lambda multiplier for Sinh, defaults to pi/2.
* ``t_max``: The largest t used to calculate the abscissa values, defaults to being derived from ``smallest_divisor``.
* ``smallest_divisor``: By default, ``t_max`` is chosen such that the abscissa are not denormals. 
  However, if you have a function like ``f = lambda x: x/10`` then the ``x/10`` could potentially take a near-denormal ``x`` into a denormal value. 
  This could impact performance, so if you can provide this value of 10 using ``smallest_divisor=10`` then it can be taken into account when calculating ``t_max``. Defaults to 2.
* ``log_valued``: If true, the integral will be summed in log-space. Defaults to false.
* ``dtype``: The dtype for the abscissa. Defaults to whatever JAX defaults to.
* ``debug_print``: When true, prints out some diagnostics that can useful for measuring performance (such as number of samples taken, etc). Defaults to false.

Examples
========

Some examples of calculating exponential integrals and lower incomplete gamma functions in JAX using these double exponential integral routines are provided in the examples folder.