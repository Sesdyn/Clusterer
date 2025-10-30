"""
Python model 'arms race.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.statefuls import Integ, Smooth
from pysd import Component

__pysd_version__ = "3.14.3"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 100,
    "time_step": lambda: 0.0625,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    """
    Current time of the model.
    """
    return __data["time"]()


@component.add(
    name="FINAL TIME", units="Month", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(
    name="INITIAL TIME", units="Month", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="SAVEPER",
    units="Month",
    limits=(0.0, np.nan),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time_step": 1},
)
def saveper():
    """
    The frequency with which output is stored.
    """
    return __data["time"].saveper()


@component.add(
    name="TIME STEP",
    units="Month",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def time_step():
    """
    The time step for the simulation.
    """
    return __data["time"].time_step()


#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(
    name="arms expenditure A perception of B",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_arms_expenditure_a_perception_of_b": 1},
    other_deps={
        "_smooth_arms_expenditure_a_perception_of_b": {
            "initial": {"arms_expenditure_a": 1},
            "step": {"arms_expenditure_a": 1, "perception_delay_time_of_b": 1},
        }
    },
)
def arms_expenditure_a_perception_of_b():
    return _smooth_arms_expenditure_a_perception_of_b()


_smooth_arms_expenditure_a_perception_of_b = Smooth(
    lambda: arms_expenditure_a(),
    lambda: perception_delay_time_of_b(),
    lambda: arms_expenditure_a(),
    lambda: 3,
    "_smooth_arms_expenditure_a_perception_of_b",
)


@component.add(
    name="arms expenditure B perception of A",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_arms_expenditure_b_perception_of_a": 1},
    other_deps={
        "_smooth_arms_expenditure_b_perception_of_a": {
            "initial": {"arms_expenditure_b": 1},
            "step": {"arms_expenditure_b": 1, "perception_delay_time_of_a": 1},
        }
    },
)
def arms_expenditure_b_perception_of_a():
    return _smooth_arms_expenditure_b_perception_of_a()


_smooth_arms_expenditure_b_perception_of_a = Smooth(
    lambda: arms_expenditure_b(),
    lambda: perception_delay_time_of_a(),
    lambda: arms_expenditure_b(),
    lambda: 3,
    "_smooth_arms_expenditure_b_perception_of_a",
)


@component.add(
    name="arms expenditure A",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_arms_expenditure_a": 1},
    other_deps={
        "_integ_arms_expenditure_a": {
            "initial": {"initial_arms_expenditure_a": 1},
            "step": {"inflow_a": 1, "outflow_a": 1},
        }
    },
)
def arms_expenditure_a():
    return _integ_arms_expenditure_a()


_integ_arms_expenditure_a = Integ(
    lambda: inflow_a() - outflow_a(),
    lambda: initial_arms_expenditure_a(),
    "_integ_arms_expenditure_a",
)


@component.add(
    name="arms expenditure B",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_arms_expenditure_b": 1},
    other_deps={
        "_integ_arms_expenditure_b": {
            "initial": {"initial_arms_expenditure_b": 1},
            "step": {"inflow_b": 1, "outflow_b": 1},
        }
    },
)
def arms_expenditure_b():
    return _integ_arms_expenditure_b()


_integ_arms_expenditure_b = Integ(
    lambda: inflow_b() - outflow_b(),
    lambda: initial_arms_expenditure_b(),
    "_integ_arms_expenditure_b",
)


@component.add(
    name="fear responsiveness A", comp_type="Constant", comp_subtype="Normal"
)
def fear_responsiveness_a():
    return 1


@component.add(
    name="fear responsiveness B", comp_type="Constant", comp_subtype="Normal"
)
def fear_responsiveness_b():
    return 1


@component.add(name="grievance A", comp_type="Constant", comp_subtype="Normal")
def grievance_a():
    return 1


@component.add(name="grievance B", comp_type="Constant", comp_subtype="Normal")
def grievance_b():
    return 1


@component.add(
    name="inflow A",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "arms_expenditure_b_perception_of_a": 1,
        "fear_responsiveness_a": 1,
        "grievance_a": 1,
    },
)
def inflow_a():
    return (
        arms_expenditure_b_perception_of_a() * fear_responsiveness_a() + grievance_a()
    )


@component.add(
    name="inflow B",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "arms_expenditure_a_perception_of_b": 1,
        "fear_responsiveness_b": 1,
        "grievance_b": 1,
    },
)
def inflow_b():
    return (
        arms_expenditure_a_perception_of_b() * fear_responsiveness_b() + grievance_b()
    )


@component.add(
    name="initial arms expenditure A", comp_type="Constant", comp_subtype="Normal"
)
def initial_arms_expenditure_a():
    return 100


@component.add(
    name="initial arms expenditure B", comp_type="Constant", comp_subtype="Normal"
)
def initial_arms_expenditure_b():
    return 100


@component.add(
    name="outflow A",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"restraint_a": 1, "arms_expenditure_a": 1},
)
def outflow_a():
    return restraint_a() * arms_expenditure_a()


@component.add(
    name="outflow B",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"restraint_b": 1, "arms_expenditure_b": 1},
)
def outflow_b():
    return restraint_b() * arms_expenditure_b()


@component.add(
    name="perception delay time of A", comp_type="Constant", comp_subtype="Normal"
)
def perception_delay_time_of_a():
    return 1


@component.add(
    name="perception delay time of B", comp_type="Constant", comp_subtype="Normal"
)
def perception_delay_time_of_b():
    return 1


@component.add(name="restraint A", comp_type="Constant", comp_subtype="Normal")
def restraint_a():
    return 1


@component.add(name="restraint B", comp_type="Constant", comp_subtype="Normal")
def restraint_b():
    return 1
