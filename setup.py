from importlib import import_module

from setuptools import setup, find_packages

MAIN_MODULE = "agent"

# Find the agent package that contains the main module
packages = find_packages(".")
agent_package = "econ_dispatch"

# Find the version number from the main module
agent_module = agent_package + "." + MAIN_MODULE
_temp = import_module(agent_module)
__version__ = _temp.__version__

# Setup
setup(
    name=agent_package + "agent",
    version=__version__,
    install_requires=["volttron", "numpy", "pandas", "pulp", "cvxopt", "scipy", "scikit-learn"],
    packages=packages,
    entry_points={
        "setuptools.installation": [
            "eggsecutable = " + agent_module + ":main",
        ]
    },
)
