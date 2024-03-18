import brainpy.core as bc
import unittest


class TestMixin(unittest.TestCase):
  def test_mixin(self):
    self.assertTrue(bc.mixin.Mixin)
    self.assertTrue(bc.mixin.ParamDesc)
    self.assertTrue(bc.mixin.ParamDescriber)
    self.assertTrue(bc.mixin.AllOfTypes)
    self.assertTrue(bc.mixin.OneOfTypes)
    self.assertTrue(bc.mixin.Mode)
    self.assertTrue(bc.mixin.Batching)
    self.assertTrue(bc.mixin.Training)


class TestMode(unittest.TestCase):
  def test_JointMode(self):
    a = bc.mixin.JointMode(bc.mixin.Batching(), bc.mixin.Training())
    self.assertTrue(a.is_a(bc.mixin.AllOfTypes[bc.mixin.Batching, bc.mixin.Training]))
    self.assertTrue(a.has(bc.mixin.Batching))
    self.assertTrue(a.has(bc.mixin.Training))
    b = bc.mixin.JointMode(bc.mixin.Batching())
    self.assertTrue(b.is_a(bc.mixin.AllOfTypes[bc.mixin.Batching]))
    self.assertTrue(b.is_a(bc.mixin.Batching))
    self.assertTrue(b.has(bc.mixin.Batching))

  def test_Training(self):
    a = bc.mixin.Training()
    self.assertTrue(a.is_a(bc.mixin.Training))
    self.assertTrue(a.is_a(bc.mixin.AllOfTypes[bc.mixin.Training]))
    self.assertTrue(a.has(bc.mixin.Training))
    self.assertTrue(a.has(bc.mixin.AllOfTypes[bc.mixin.Training]))
    self.assertFalse(a.is_a(bc.mixin.Batching))
    self.assertFalse(a.has(bc.mixin.Batching))

  def test_Batching(self):
    a = bc.mixin.Batching()
    self.assertTrue(a.is_a(bc.mixin.Batching))
    self.assertTrue(a.is_a(bc.mixin.AllOfTypes[bc.mixin.Batching]))
    self.assertTrue(a.has(bc.mixin.Batching))
    self.assertTrue(a.has(bc.mixin.AllOfTypes[bc.mixin.Batching]))

    self.assertFalse(a.is_a(bc.mixin.Training))
    self.assertFalse(a.has(bc.mixin.Training))

  def test_Mode(self):
    a = bc.mixin.Mode()
    self.assertTrue(a.is_a(bc.mixin.Mode))
    self.assertTrue(a.is_a(bc.mixin.AllOfTypes[bc.mixin.Mode]))
    self.assertTrue(a.has(bc.mixin.Mode))
    self.assertTrue(a.has(bc.mixin.AllOfTypes[bc.mixin.Mode]))

    self.assertFalse(a.is_a(bc.mixin.Training))
    self.assertFalse(a.has(bc.mixin.Training))
    self.assertFalse(a.is_a(bc.mixin.Batching))
    self.assertFalse(a.has(bc.mixin.Batching))


