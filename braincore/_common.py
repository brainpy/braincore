def set_module_as(module: str):
  def wrapper(fun: callable):
    fun.__module__ = module
    return fun

  return wrapper
