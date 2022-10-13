from nython import nythonize
from os.path import expanduser


def build(setup_kwargs):
    """Called by poetry, the args are added to the kwargs for setup."""
    nimbase = expanduser("~") + "/.choosenim/toolchains/nim-1.6.8/lib/nimbase.h"
    setup_kwargs.update(
        {
            "ext_modules": nythonize(
                nimbase, [{"name": "nimsmo", "path": "src/nimsmo/nimsmo.nim"}]
            ),
        }
    )
