# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import subprocess
from typing import List

from .genscenario import (
    gen_bubbles,
    gen_friction_map,
    gen_group_laps,
    gen_map,
    gen_missions,
    gen_scenario,
    gen_social_agent_missions,
    gen_traffic,
    gen_traffic_histories,
)


# PYTHONHASHSEED must be "random", unset (default `None`), or an integer in range [0; 4294967295]
_hashseed = os.getenv("PYTHONHASHSEED")
if _hashseed is None:
    _hashseed = 42
    # We replace the seed if it does not exist to make subprocesses predictable
    os.environ["PYTHONHASHSEED"] = f"{_hashseed}"
elif _hashseed == "random":
    import logging

    logging.warning(
        "PYTHONHASHSEED is 'random'. Simulation and generation may be unpredictable."
    )


def build_scenario(scenario: List[str]):
    """Build the given scenarios.

    Args:
        scenario (List[str]): Scenarios to build.
    """
    build_scenario = " ".join(["scl scenario build-all --clean"] + scenario)
    subprocess.call(build_scenario, shell=True)
