"""
Copyright (c) 2023 G. H. Collin (ghcollin)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import jax
import jax.numpy as jnp

def ln_two_cosh(x):
    return jnp.logaddexp(0.0, 2*x) - x # = ln(1 + exp(2x)) - x = ln(1 + exp(2x)) + ln(exp(-x)) = ln(exp(-x) + exp(x))

def one_pm_tanh_sinh(x, exp_x, lamb, ln_weight):
    two_sinh_x = exp_x - 1/exp_x # = 2 sinh(x)
    two_cosh_x = exp_x + 1/exp_x # = 2 cosh(x)
    lamb_over_2ln2 = lamb/jnp.log(4) # = pi/(2 ln(2))
    exp_lam_sinh_x = jnp.exp2(lamb_over_2ln2 * two_sinh_x) # = exp(lamb sinh(x)) = exp2(lamb/ln2 sinh(x))
    #two_sinh_lam_sinh_x = exp_lam_sinh_x - 1/exp_lam_sinh_x
    two_cosh_lam_sinh_x = exp_lam_sinh_x + 1/exp_lam_sinh_x
    absic = 2/(jnp.square(exp_lam_sinh_x)+1) # = 1 - tanh(lamb sinh(x))
    if ln_weight:
        ln_two_cosh_lam_sinh_x = ln_two_cosh((lamb/2)*two_sinh_x)
        weight = jnp.log(2*lamb) + jnp.log(two_cosh_x) - 2*ln_two_cosh_lam_sinh_x #= ln(lamb cosh(x)/(cosh(lamb sinh(x))))
    else:
        weight = 2*lamb*two_cosh_x/jnp.square(two_cosh_lam_sinh_x) # = lamb cosh(x)/(cosh(lamb sinh(x)))^2
    return (-absic, weight), (absic, weight)

def exp_sinh(x, exp_x, lamb, ln_weight):
    two_sinh_x = exp_x - 1/exp_x # = 2 sinh(x)
    two_cosh_x = exp_x + 1/exp_x # = 2 cosh(x)
    lamb_over_2 = lamb/2
    lamb_over_2ln2 = lamb_over_2/jnp.log(2) # = pi/(4 ln(2))
    exp_lam_sinh_x = jnp.exp2(lamb_over_2ln2 * two_sinh_x) # = exp(lamb sinh(x)) = exp2(lamb/ln2 sinh(x))
    lower_absic = 1/exp_lam_sinh_x 
    upper_absic = exp_lam_sinh_x
    if ln_weight:
        ln_lamb_cosh_x = jnp.log(lamb_over_2) + jnp.log(two_cosh_x)
        lower_weight = ln_lamb_cosh_x - lamb_over_2 * two_sinh_x # = ln(lamb cosh(-x) exp(lamb sinh(-x)))
        upper_weight = ln_lamb_cosh_x + lamb_over_2 * two_sinh_x # = ln(lamb cosh(x) exp(lamb sinh(x)))
    else:
        lamb_cosh_x = lamb_over_2 * two_cosh_x
        lower_weight = lamb_cosh_x/exp_lam_sinh_x # = exp(lamb sinh(-x)) lamb cosh(-x) = exp(-lamb sinh(x)) lamb cosh(x)
        upper_weight = lamb_cosh_x * exp_lam_sinh_x # = exp(lamb sinh(x)) lamb cosh(x)
    return (lower_absic, lower_weight), (upper_absic, upper_weight)

def exp_exp(x, exp_x, lamb, ln_weight):
    ln_lower_absic = -x + (1 - exp_x)
    lower_absic = jnp.exp(ln_lower_absic)
    ln_upper_absic = x + (1 - 1/exp_x)
    upper_absic = jnp.exp(ln_upper_absic)
    if ln_weight:
        lower_weight = ln_lower_absic + jnp.log(1 + exp_x)
        upper_weight = ln_upper_absic + jnp.log(1 + 1/exp_x)
    else:
        lower_weight = lower_absic * (1 + exp_x)
        upper_weight = upper_absic * (1 + 1/exp_x)
    return (lower_absic, lower_weight), (upper_absic, upper_weight)

def jax_do_while(**init):
    def proj(d):
        return d['next'], d['cond']
    def wrapper(fun):
        result, _ = jax.lax.while_loop(lambda c: c[1], lambda c: proj(fun(**(c[0]))), (init, True))
        return lambda *keys: [ result[k] for k in keys ] if len(keys) > 1 else result[keys[0]]
    return wrapper

def deint(f_lower, f_upper, f_mid=None, tol=None, max_samples=4096, strategy='one_pm_tanh_sinh', 
          min_samples=16, lamb=jnp.pi/2, smallest_divisor=None, t_max=None, log_valued=False, dtype=None, debug_print=False):
    assert (f_lower is not None or f_upper is not None)
    q0_from = 0 if f_lower is not None else 1

    if log_valued:
        val_zero, val_one = jnp.array(-jnp.inf, dtype=dtype), jnp.array(0.0, dtype=dtype)
        add = jnp.logaddexp
        mul = lambda a, b: a + b
        div = lambda a, b: a - b
        abs_sub = lambda a, b: jnp.log1p(-jnp.exp(-jnp.abs(a - b))) + jnp.maximum(a, b)
        abs = lambda a: a
        from_literal = lambda a: jnp.log(a)
    else:
        val_zero, val_one = jnp.array(0.0, dtype=dtype), jnp.array(1.0, dtype=dtype)
        add = lambda a, b: a + b
        mul = lambda a, b: a * b
        div = lambda a, b: a / b
        abs = lambda a: jnp.fabs(a)
        abs_sub = lambda a, b: abs(a - b)
        from_literal = lambda a: a
    f_lower = f_lower if f_lower is not None else lambda x: val_zero
    f_upper = f_upper if f_upper is not None else lambda x: val_zero

    strats = dict(
        one_pm_tanh_sinh = (one_pm_tanh_sinh, lambda x: jnp.arcsinh(-jnp.log(x)/(2*lamb))),
        exp_sinh = (exp_sinh, lambda x: jnp.arcsinh(-jnp.log(x)/(lamb))),
        exp_exp = (exp_exp, lambda x: jnp.log(-jnp.log(x))),
    )

    strat_fn, t_max_fn = strats[strategy]

    def eval_fw(level, current, t, exp_t):
        (lower_absic, lower_weight), (upper_absic, upper_weight) = strat_fn(t, exp_t, lamb=lamb, ln_weight=log_valued)
        #jax.debug.print("\t\t{} \t{} \t{} \t{}", lower_absic, lower_weight, upper_absic, upper_weight)
        lower = jax.lax.cond(lower_absic != 0, lambda: mul(lower_weight, f_lower(lower_absic)), lambda: val_zero)
        upper = jax.lax.cond(upper_absic != 0, lambda: mul(upper_weight, f_upper(upper_absic)), lambda: val_zero)
        return jax.lax.cond(jnp.isfinite(lower), lambda: lower, lambda: val_zero), jax.lax.cond(jnp.isfinite(upper), lambda: upper, lambda: val_zero)

    q0 = f_mid if f_mid is not None else eval_fw(2, 0, t=0.0, exp_t=1.0)[q0_from]

    tol = tol if tol is not None else jnp.maximum(jnp.array(jnp.finfo(q0.dtype).eps, dtype=dtype) * 10, jnp.array(1e-9, dtype=dtype))
    smallest_divisor = smallest_divisor if smallest_divisor is not None else 2
    smallest_arg = jnp.array(jnp.minimum(smallest_divisor*jnp.finfo(q0.dtype).tiny, 1e-10), dtype=dtype)

    # from https://arxiv.org/pdf/2007.15057.pdf
    t_max = t_max if t_max is not None else t_max_fn(smallest_arg)

    if debug_print:
        jax.debug.print("Starting with tol = {}, q0 = {}, t_max = {}", tol, q0, t_max)
        jax.debug.print("Level\t| rel. diff \t| abs. diff\t| integral - q0")

    @jax_do_while(level=2, q=val_zero)
    def outer_loop(level, q):
        t0 = t_max/level
        t_step = 2*t0
        exp_t0 = jnp.exp(t0)
        exp_t_step = jnp.square(exp_t0)
        @jax_do_while(current=1, t=t0, exp_t=exp_t0, val=val_zero)
        def inner_loop(current, t, exp_t, val):
            vl, vu = eval_fw(level, current, t, exp_t)
            return dict(
                cond = current + 2 < level,
                next = dict(current=current + 2, t=t+t_step, exp_t=exp_t*exp_t_step, val=add(val, add(vl, vu))) # val = val + vl + vu
            )
        vals = inner_loop('val')
        abs_diff = div(abs_sub(add(q0, q), vals), from_literal(2.0)) # = (q0 + q - vals)/2  = q - (vals + q)/2
        summ = add(q, vals) # = q + vals
        if debug_print:
            rel_diff = div(abs_diff, abs(add(q0, summ))) # = |diff|/|q0 + summ|
            jax.debug.print("{} \t| {} \t| {} \t| {}", level, rel_diff, abs_diff, mul(from_literal(t_max/level), summ))
        return dict(
            cond = jnp.logical_or(jnp.logical_and(abs_diff > mul(from_literal(tol), abs(add(q0, summ))), level < max_samples), level < min_samples),
            next = dict(level=2*level, q=summ)
        )
    q, level = outer_loop('q', 'level')
    return mul(from_literal(2*t_max/level), add(q0, q)) # = 2*t_max * (q0+q)/(level)

def deint_interval(f, a, b, smallest_divisor=2, log_valued=False, **kw_args):
    h = (b-a)/2
    xform_lower, xform_upper = lambda x: b + h*x, lambda x: a + h*x
    smallest_divisor = jnp.maximum(1.0, 1/h) * smallest_divisor
    result = deint(lambda x: f(xform_lower(x)), lambda x: f(xform_upper(x)), smallest_divisor=smallest_divisor, strategy='one_pm_tanh_sinh', log_valued=log_valued, **kw_args)
    return jnp.log(h) + result if log_valued else h*result

def deint_halfinfinite_scale(f, a, s, smallest_divisor=2, log_valued=False, strategy='exp_sinh', **kw_args):
    xform = lambda x: a + (s - a)*x
    weight = (s - a)
    smallest_divisor = jnp.maximum(1.0, 1/weight) * smallest_divisor
    result = deint(lambda x: f(xform(x)), lambda x: f(xform(x)), smallest_divisor=smallest_divisor, strategy=strategy, **kw_args)
    return jnp.log(weight) + result if log_valued else weight*result