def warp_module(module: str):
  def wrapper(fun: callable):
    fun.__module__ = module
    return fun

  return wrapper
