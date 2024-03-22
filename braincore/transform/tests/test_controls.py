import unittest

import jax

import braincore as bc


class TestIfElse(unittest.TestCase):
  def test1(self):
    def f(a):
      return bc.ifelse(conditions=[a < 0, a < 2, a < 5, a < 10],
                       branches=[lambda: 1,
                                 lambda: 2,
                                 lambda: 3,
                                 lambda: 4,
                                 lambda: 5])

    self.assertTrue(f(3) == 3)
    self.assertTrue(f(1) == 2)
    self.assertTrue(f(-1) == 1)

  def test2(self):
    def f(a):
      return bc.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
                       branches=[1, 2, 3, 4, 5])

    self.assertTrue(f(3) == 3)
    self.assertTrue(f(1) == 4)
    self.assertTrue(f(-1) == 5)

  def test_vmap(self):
    def f(operands):
      f = lambda a: bc.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
                              branches=[lambda _: 1,
                                        lambda _: 2,
                                        lambda _: 3,
                                        lambda _: 4,
                                        lambda _: 5, ],
                              operands=a,
                              show_code=True)
      return jax.vmap(f)(operands)

    r = f(bc.random.randint(-20, 20, 200))
    self.assertTrue(r.size == 200)

  def test_vmap2(self):
    def f2():
      f = lambda a: bc.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
                              branches=[1, 2, 3, 4, lambda _: 5],
                              operands=a,
                              show_code=True)
      return jax.vmap(f)(bc.random.randint(-20, 20, 200))

    self.assertTrue(f2().size == 200)

  def test_grad1(self):
    def F2(x):
      return bc.ifelse(conditions=(x >= 10,),
                       branches=[lambda x: x,
                                 lambda x: x ** 2, ],
                       operands=x)

    self.assertTrue(jax.grad(F2)(9.0) == 18.)
    self.assertTrue(jax.grad(F2)(11.0) == 1.)

  def test_grad2(self):
    def F3(x):
      return bc.ifelse(conditions=(x >= 10, x >= 0),
                       branches=[lambda x: x,
                                 lambda x: x ** 2,
                                 lambda x: x ** 4, ],
                       operands=x)

    self.assertTrue(jax.grad(F3)(9.0) == 18.)
    self.assertTrue(jax.grad(F3)(11.0) == 1.)
