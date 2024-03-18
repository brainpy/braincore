import brainpy.core as bc
import brainpy as bp
import brainpy.math as bm


class HHLTC(bc.Dynamics):

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


    # integral
    self.integral = odeint(method=method, f=self.derivative)

  # m channel
  # m_alpha = lambda self, V: 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
  m_alpha = lambda self, V: 1. / bm.exprel(-(V + 40) / 10)
  m_beta = lambda self, V: 4.0 * bm.exp(-(V + 65) / 18)
  m_inf = lambda self, V: self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))
  dm = lambda self, m, t, V: self.m_alpha(V) * (1 - m) - self.m_beta(V) * m

  # h channel
  h_alpha = lambda self, V: 0.07 * bm.exp(-(V + 65) / 20.)
  h_beta = lambda self, V: 1 / (1 + bm.exp(-(V + 35) / 10))
  h_inf = lambda self, V: self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))
  dh = lambda self, h, t, V: self.h_alpha(V) * (1 - h) - self.h_beta(V) * h

  # n channel
  # n_alpha = lambda self, V: 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
  n_alpha = lambda self, V: 0.1 / bm.exprel(-(V + 55) / 10)
  n_beta = lambda self, V: 0.125 * bm.exp(-(V + 65) / 80)
  n_inf = lambda self, V: self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))
  dn = lambda self, n, t, V: self.n_alpha(V) * (1 - n) - self.n_beta(V) * n

  def reset_state(self, batch_size=None, **kwargs):
    self.V = self.init_variable(self._V_initializer, batch_size)
    if self._m_initializer is None:
      self.m = bm.Variable(self.m_inf(self.V.value), batch_axis=self.V.batch_axis)
    else:
      self.m = self.init_variable(self._m_initializer, batch_size)
    if self._h_initializer is None:
      self.h = bm.Variable(self.h_inf(self.V.value), batch_axis=self.V.batch_axis)
    else:
      self.h = self.init_variable(self._h_initializer, batch_size)
    if self._n_initializer is None:
      self.n = bm.Variable(self.n_inf(self.V.value), batch_axis=self.V.batch_axis)
    else:
      self.n = self.init_variable(self._n_initializer, batch_size)
    self.spike = self.init_variable(partial(bm.zeros, dtype=bool), batch_size)

  def dV(self, V, t, m, h, n, I):
    I = self.sum_current_inputs(V, init=I)
    I_Na = (self.gNa * m * m * m * h) * (V - self.ENa)
    n2 = n * n
    I_K = (self.gK * n2 * n2) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + I) / self.C
    return dVdt

  @property
  def derivative(self):
    return JointEq(self.dV, self.dm, self.dh, self.dn)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    x = 0. if x is None else x

    V, m, h, n = self.integral(self.V.value, self.m.value, self.h.value, self.n.value, t, x, dt)
    V += self.sum_delta_inputs()
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.V.value = V
    self.m.value = m
    self.h.value = h
    self.n.value = n
    return self.spike.value

  def return_info(self):
    return self.spike
