# -*- coding: utf-8 -*-


import functools
from typing import Callable

from braincore import environ
from brainpy import math as bm
from brainpy._src.integrators import constants, utils
from brainpy._src.integrators.base import Integrator
from brainpy._src.integrators.constants import DT
from brainpy.errors import DiffEqError, CodeError

from ._joint_eq import JointEq

__all__ = [
  'register_ode_integrator',
  'get_supported_methods',
]

name2method = {
}


def odeint(
    f=None,
    method=None,
    var_type=None,
    dt=None,
    name=None,
    **kwargs
):
  """Numerical integration for ODEs.

  Examples
  --------

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> a=0.7;  b=0.8;  tau=12.5;  Vth=1.9
    >>> V = 0;  w = 0  # initial values
    >>>
    >>> @bp.odeint(method='rk4', dt=0.04)
    >>> def integral(V, w, t, Iext):
    >>>   dw = (V + a - b * w) / tau
    >>>   dV = V - V * V * V / 3 - w + Iext
    >>>   return dV, dw
    >>>
    >>> hist_V = []
    >>> for t in bp.math.arange(0, 100, integral.dt):
    >>>     V, w = integral(V, w, t, 0.5)
    >>>     hist_V.append(V)
    >>> plt.plot(bp.math.arange(0, 100, integral.dt), hist_V)
    >>> plt.show()


  Parameters
  ----------
  f : callable, function
    The derivative function.
  method : str
    The shortcut name of the numerical integrator.
  var_type: str
    The type of the variable defined in the equation.
  dt: float
    The numerical integration precision.
  name: str
    The integrator node.

  Returns
  -------
  integral : ODEIntegrator
      The numerical solver of `f`.
  """
  if method not in name2method:
    raise ValueError(f'Unknown ODE numerical method "{method}". Currently '
                     f'BrainPy only support: {list(name2method.keys())}')

  if f is None:
    return lambda f: name2method[method](f,
                                         var_type=var_type,
                                         dt=dt,
                                         name=name,
                                         **kwargs)
  else:
    return name2method[method](f,
                               var_type=var_type,
                               dt=dt,
                               name=name,
                               **kwargs)


def register_ode_integrator(name, integrator):
  """Register a new ODE integrator.

  Parameters
  ----------
  name: ste
  integrator: type
  """
  if name in name2method:
    raise ValueError(f'"{name}" has been registered in ODE integrators.')
  if not issubclass(integrator, ODEIntegrator):
    raise ValueError(f'"integrator" must be an instance of {ODEIntegrator.__name__}')
  name2method[name] = integrator


def get_supported_methods():
  """Get all supported numerical methods for DDEs."""
  return list(name2method.keys())


def f_names(f):
  func_name = constants.unique_name('ode')
  if f.__name__.isidentifier():
    func_name += '_' + f.__name__
  return func_name


class ODEIntegrator(Integrator):
  """Numerical Integrator for Ordinary Differential Equations (ODEs).

  Parameters
  ----------
  f : callable
    The derivative function.
  var_type: str
    The type for each variable.
  dt: float, int
    The numerical precision.
  name: str
    The integrator name.
  """

  def __init__(
      self,
      f: Callable,
      var_type: str = None,
      dt: float = None,
      name: str = None,
      show_code: bool = False,
  ):

    dt = environ.get_dt() if dt is None else dt
    parses = utils.get_args(f)
    variables = parses[0]  # variable names, (before 't')
    parameters = parses[1]  # parameter names, (after 't')
    arguments = parses[2]  # function arguments

    for p in tuple(variables) + tuple(parameters):
      if p == DT:
        raise CodeError(f'{DT} is a system keyword denotes the '
                        f'precision of numerical integration. '
                        f'It cannot be used as a variable or parameter, '
                        f'please change an another name.')

    # super initialization
    super(ODEIntegrator, self).__init__(name=name,
                                        variables=variables,
                                        parameters=parameters,
                                        arguments=arguments,
                                        dt=dt)

    # others
    self.show_code = show_code
    self.var_type = var_type  # variable type

    # derivative function
    self.derivative = {constants.F: f}
    self.f = f

    # code scope
    self.code_scope = {constants.F: f}

    # code lines
    self.func_name = f_names(f)
    self.code_lines = [f'def {self.func_name}({", ".join(self.arguments)}):']

  def __call__(self, *args, **kwargs):
    assert self.integral is not None, 'Please build the integrator first.'

    # check arguments
    for i, arg in enumerate(args):
      kwargs[self.arg_names[i]] = arg

    # integral
    new_vars = self.integral(**kwargs)
    return new_vars


class ExponentialEuler(ODEIntegrator):
  """Exponential Euler method using automatic differentiation.

  Parameters
  ----------
  f : function, joint_eq.JointEq
    The derivative function.
  var_type : optional, str
    The variable type.
  dt : optional, float
    The default numerical integration step.
  name : optional, str
    The integrator name.
  """

  def __init__(
      self,
      f,
      var_type=None,
      dt=None,
      name=None,
      show_code=False,
  ):
    super().__init__(f=f,
                     var_type=var_type,
                     dt=dt,
                     name=name,
                     show_code=show_code)

    # if var_type == C.SYSTEM_VAR:
    #   raise NotImplementedError(f'{self.__class__.__name__} does not support {C.SYSTEM_VAR}, '
    #                             f'because the auto-differentiation ')

    # build the integrator
    self.code_lines = []
    self.code_scope = {}
    self.integral = self.build()

  def build(self):
    parses = self._build_integrator(self.f)
    all_vps = self.variables + self.parameters

    @functools.wraps(self.f)
    def integral_func(*args, **kwargs):
      # format arguments
      params_in = bm.Collector()
      for i, arg in enumerate(args):
        params_in[all_vps[i]] = arg
      params_in.update(kwargs)
      if constants.DT not in params_in:
        params_in[constants.DT] = self.dt

      # call integrals
      results = []
      for i, parse in enumerate(parses):
        f_integral, vars_, pars_ = parse
        vps = vars_ + pars_ + [constants.DT]
        r = f_integral(params_in[vps[0]], **{arg: params_in[arg] for arg in vps[1:] if arg in params_in})
        results.append(r)
      return results if len(self.variables) > 1 else results[0]

    return integral_func

  def _build_integrator(self, eq):
    if isinstance(eq, JointEq):
      results = []
      for sub_eq in eq.eqs:
        results.extend(self._build_integrator(sub_eq))
      return results
    else:
      vars, pars, _ = utils.get_args(eq)

      # checking
      if len(vars) != 1:
        raise DiffEqError(constants.multi_vars_msg.format(cls=self.__class__.__name__,
                                                          vars=str(vars),
                                                          eq=str(eq)))

      # gradient function
      value_and_grad = bm.vector_grad(eq, argnums=0, return_value=True)

      # integration function
      def integral(*args, **kwargs):
        assert len(args) > 0
        dt = kwargs.pop(constants.DT, self.dt)
        linear, derivative = value_and_grad(*args, **kwargs)
        phi = bm.exprel(dt * linear)
        return args[0] + dt * phi * derivative

      return [(integral, vars, pars), ]


register_ode_integrator('exp_euler', ExponentialEuler)
