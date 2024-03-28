import importlib.util
import types

# Lazy import of braincore modules
class LazyImport(types.ModuleType):
  def __init__(self, module_name):
    self.module_name = module_name
    self.module = None

  def _load_module(self):
    if self.module is None:
      self.module = importlib.import_module(self.module_name)

  # load when attribute is accessed
  def __getattr__(self, item):
    self._load_module()
    return getattr(self.module, item)

  # process special attributes
  def __setattr__(self, key, value):
    if key in {"module", "module_name"}:
      super().__setattr__(key, value)
    else:
      self._load_module()
      setattr(self.module, key, value)

  def __dir__(self):
    self._load_module()
    return dir(self.module)
