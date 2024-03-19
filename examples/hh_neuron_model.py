import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import brainpy.core as bc


class HHWithEuler(bc.Dynamics):

  def __init__(self, size, keep_size: bool = False, ENa=50., gNa=120., EK=-77.,
               gK=36., EL=-54.387, gL=0.03, V_th=20., C=1.0):
    # initialization
    super().__init__(size=size, keep_size=keep_size, )

    # parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.C = C
    self.V_th = V_th

  # m channel
  # m_alpha = lambda self, V: 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
  m_alpha = lambda self, V: 1. / bc.math.exprel(-(V + 40) / 10)
  m_beta = lambda self, V: 4.0 * jnp.exp(-(V + 65) / 18)
  m_inf = lambda self, V: self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))
  dm = lambda self, m, t, V: self.m_alpha(V) * (1 - m) - self.m_beta(V) * m

  # h channel
  h_alpha = lambda self, V: 0.07 * jnp.exp(-(V + 65) / 20.)
  h_beta = lambda self, V: 1 / (1 + jnp.exp(-(V + 35) / 10))
  h_inf = lambda self, V: self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))
  dh = lambda self, h, t, V: self.h_alpha(V) * (1 - h) - self.h_beta(V) * h

  # n channel
  # n_alpha = lambda self, V: 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
  n_alpha = lambda self, V: 0.1 / bc.math.exprel(-(V + 55) / 10)
  n_beta = lambda self, V: 0.125 * jnp.exp(-(V + 65) / 80)
  n_inf = lambda self, V: self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))
  dn = lambda self, n, t, V: self.n_alpha(V) * (1 - n) - self.n_beta(V) * n

  def init_state(self, batch_size=None):
    self.V = bc.State(jnp.ones(self.varshape, bc.environ.dftype()))
    self.m = bc.State(self.m_inf(self.V.value))
    self.h = bc.State(self.h_inf(self.V.value))
    self.n = bc.State(self.n_inf(self.V.value))
    self.spike = bc.State(jnp.zeros(self.varshape, bool))

  def dV(self, V, t, m, h, n, I):
    I = self.sum_current_inputs(V, init=I)
    I_Na = (self.gNa * m * m * m * h) * (V - self.ENa)
    n2 = n * n
    I_K = (self.gK * n2 * n2) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + I) / self.C
    return dVdt

  def update(self, x=None):
    t = bc.share.get('t')
    x = 0. if x is None else x
    dt = bc.environ.get_dt()
    V = self.V.value + self.dV(self.V.value, t, self.m.value, self.h.value, self.n.value, x) * dt
    m = self.m.value + self.dm(self.m.value, t, self.V.value) * dt
    h = self.h.value + self.dh(self.h.value, t, self.V.value) * dt
    n = self.n.value + self.dn(self.n.value, t, self.V.value) * dt
    V += self.sum_delta_inputs()
    self.spike.value = jnp.logical_and(self.V.value < self.V_th, V >= self.V_th)
    self.V.value = V
    self.m.value = m
    self.h.value = h
    self.n.value = n
    return self.spike.value

  def update_return_info(self):
    return jax.ShapeDtypeStruct(self.varshape, bc.environ.dftype())

  def update_return(self):
    return self.spike.value


bc.environ.set(dt=0.01)
hh = HHWithEuler(10)
hh.init_state()
states = hh.states()


def run(st_vals, x):
  i, inp = x
  bc.share.set(i=i, t=i * bc.environ.get_dt())
  states.assign_values(st_vals)
  hh(inp)
  return states.collect_values(), hh.V.value


n = 100000
indices = jnp.arange(n)
st_vals, vs = jax.lax.scan(run, states.collect_values(), (indices, bc.random.uniform(01., 10., n)))
states.assign_values(st_vals)

plt.plot(indices * bc.environ.get_dt(), vs)
plt.show()
