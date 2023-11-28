"""
Copyright (c) 2023 G. H. Collin (ghcollin)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

def lambertWm1lowerbound_minus(y, z): # this is an lower bound on W_-1(-e^(-(y-z))) - z for y-z > 1
    return - jnp.sqrt(2*(y-z-1)) - y

def de_nexpintv_scale_point_minus_one(v, x):
    pos_v = jax.lax.cond(v > 0, lambda: v, lambda: -v) # this 'pos_v' value is only used when v > 0, so we ensure it's always positive even in the v < 0 branch to avoid NaNs in the following function
    pos_v_decay_length_minus_one = lambda q: v*jnp.logaddexp(-x/pos_v, q/pos_v + jnp.log(x/pos_v))/x
    neg_v = jax.lax.cond(v < 0, lambda: -v, lambda: v) # this 'neg_v' value is only used when v < 0, so we ensure it's always positive even in the v > 0 branch to avoid NaNs in the following function
    neg_v_decay_length_minus_one = lambda q: v*lambertWm1lowerbound_minus(q/neg_v - jnp.log(x/neg_v), -x/neg_v)/x
    e_folds = 1.0
    scale_point = jax.lax.cond(-v/x > 2, 
        lambda: -v/x - 1, 
        lambda: jax.lax.cond(v > 0, 
            lambda: pos_v_decay_length_minus_one(e_folds), 
            lambda: jax.lax.cond(v < 0, 
                lambda: neg_v_decay_length_minus_one(e_folds), 
                lambda: e_folds/x) # v == 0 case, just use the exponential decay length for one e-fold
        )
    )
    return scale_point

@jax.custom_jvp
def de_nexpintv_normed(v, x, ln_norm):
    scale_point_minus_one = de_nexpintv_scale_point_minus_one(v, x)
    result = deint_halfinfinite_scale(          lambda tm1: jnp.exp(-x*tm1 - (v)*jnp.log1p(tm1) - (ln_norm+x)), 0.0, scale_point_minus_one, strategy='exp_exp')
    return result

"""
Calculates the generalised exponential integral E_v(x).
Returns the primal value and a logarithmic normalisation. Such that for
p, n = de_nexpintv_norm(v, x)
We have E_v(x) = p*exp(n)
"""
def de_nexpintv_norm(v, x):
    ln_norm = de_nexpint_norm_value(v, x)
    return de_nexpintv_normed(v, x, ln_norm), ln_norm

def de_dv_nexpintv_normed(v, x, ln_norm):
    scale_point_minus_one = de_nexpintv_scale_point_minus_one(v-1, x) # we approximate the log(t) as t, so this is approximately a exponential integral with v-1
    result = deint_halfinfinite_scale(          lambda tm1: -jnp.log1p(tm1) * jnp.exp(-x*(tm1) - (v)*jnp.log1p(tm1) - (ln_norm+x)), 0.0, scale_point_minus_one, strategy='exp_exp')
    return result

@de_nexpintv_normed.defjvp
def de_nexptinv_normed_jvp(primals, tangents):
    v, x, ln_norm = primals
    dv, dx, dln_norm = tangents
    primal_value = de_nexpintv_normed(v, x, ln_norm)
    dfdx = -de_nexpintv_normed(v-1, x, ln_norm)
    dfdv = de_dv_nexpintv_normed(v, x, ln_norm)
    dfdln_norm = -primal_value
    return primal_value, dfdv * dv + dfdx * dx + dfdln_norm * dln_norm

"""
Calculates the difference: E_v(x) - E_v(0)
This is used for very small values of x, where cancellation might appear. 
"""
@jax.custom_jvp
def de_nexpintv_minus_expintv0(v, x):
    result = deint_interval(        lambda s: jnp.expm1(-x/s) * jnp.power(s, v-2), 0.0, 1.0)
    return (result)

@de_nexpintv_minus_expintv0.defjvp
def de_nexpintv_minus_expintv0_jvp(primals, tangents):
    v, x = primals
    dv, dx = tangents
    primal_value = de_nexpintv_minus_expintv0(v, x)
    dfdv = deint_interval(        lambda s: jnp.log(s) * jnp.expm1(-x/s) * jnp.power(s, v-2), 0.0, 1.0)
    df_ln_norm = de_nexpint_norm_value(v-1, x)
    dfdx = -de_nexpintv_normed(v-1, x, ln_norm=df_ln_norm) * jnp.exp(df_ln_norm)
    return primal_value, dfdv * dv + dfdx * dx

def de_lower_incomplete_gamma_norm_value(a, x):
    t_peak = jnp.clip((a-1)/x, a_min=0.0, a_max=1.0)

    ln_integrand = lambda t: -x*t + jax.scipy.special.xlogy(a - 1, t) #(a - 1) * jnp.log(t)

    ln_norm = ln_integrand(t_peak)
    ln_post = a * jnp.log(x)
    return ln_norm, ln_post

@jax.custom_jvp
def de_lower_expintv_normed(a, x, ln_norm):
    result = deint_interval(        lambda t: jnp.exp(-x*t + (a - 1) * jnp.log(t) - ln_norm), 0.0, 1.0)
    return (result) 

"""
Calcualtes the lower incomplete gamma function.
Returns the main value and a logarithmic normalisation value.
"""
def de_lower_incomplete_gamma_norm(a, x):
    ln_norm, ln_post = de_lower_incomplete_gamma_norm_value(a, x)
    return de_lower_expintv_normed(a, x, ln_norm), ln_norm + ln_post

@de_lower_expintv_normed.defjvp
def de_lower_expintv_normed_jvp(primals, tangents):
    a, x, ln_norm = primals
    da, dx, dln_norm = tangents
    primal_value = de_lower_expintv_normed(a, x, ln_norm)
    dfda = deint_interval(        lambda t: jnp.log(t) * jnp.exp( -x*t + (a - 1) * jnp.log(t) - ln_norm), 0.0, 1.0)
    dfdx = -de_lower_expintv_normed(a+1, x, ln_norm)
    dfdln_norm = -primal_value
    return primal_value, dfda * da + dfdx * dx + dfdln_norm * dln_norm