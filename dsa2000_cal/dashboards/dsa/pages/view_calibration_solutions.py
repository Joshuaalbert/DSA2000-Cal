import asyncio

import ray
import streamlit as st

from dsa2000_fm.actors.calibration_solution_cache import CalibrationSolutionCache, \
    CalibrationSolution


# We will enable viewing certain things during a backtest:
# 1. Plot equity curve for a given account_id, dropdown select account_id. Show transfers underneath.
# 2. For a given timestamp, strategy horizon, martingale horizon, and simulation count, plot the optimal strategy tables per loop (drop down select loop), and the simulated future PNL.
#

@st.cache_resource
def get_calibration_cache() -> CalibrationSolutionCache:
    ray.init(address='auto', ignore_reinit_error=True)
    return CalibrationSolutionCache()


@st.cache_data
def get_calibration_solution_snapshot(sol_int_time_idx: int, sol_int_freq_idx: int) -> CalibrationSolution:
    return asyncio.run(get_calibration_cache().get_calibration_solution_snapshot(sol_int_time_idx, sol_int_freq_idx))


def view_calibration_solutions():
    sol_int_time_idx = st.number_input('Solution Interval Time Index', value=0, step=1,
                                       help='Solution Interval Time Index.')
    sol_int_freq_idx = st.number_input('Solution Interval Frequency Index', value=0, step=1,
                                       help='Solution Interval Frequency Index.')
    if st.button('View Calibration Solution'):
        calibration_solution = get_calibration_solution_snapshot(sol_int_time_idx, sol_int_freq_idx)
        st.write(calibration_solution)


def run():
    view_calibration_solutions()


run()
