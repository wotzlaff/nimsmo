from nython import nythonize
from os.path import expanduser


def build(setup_kwargs):
    """Called by poetry, the args are added to the kwargs for setup."""
    ext_modules = nythonize(
        expanduser("~") + "/.choosenim/toolchains/nim-1.6.8/lib/nimbase.h",
        [{"name": "nimsmo", "path": "src/nimsmo/nimsmo.nim"}],
    )
    ext_modules[0].sources.append('./src/math.c')
    setup_kwargs.update(dict(ext_modules=ext_modules))
