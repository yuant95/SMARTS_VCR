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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import psutil

from smarts.core.agent_buffer import AgentBuffer
from smarts.core.buffer_agent import BufferAgent
from smarts.core.local_agent import LocalAgent


class LocalAgentBuffer(AgentBuffer):
    """A buffer that manages social agents."""

    def __init__(self):
        num_cpus = max(
            2, psutil.cpu_count(logical=False) or (psutil.cpu_count() - 1)
        )
        self._act_executor = ThreadPoolExecutor(num_cpus)

    def destroy(self):
        self._act_executor.shutdown(wait=True)

    def acquire_agent(
        self, retries: int = 3, timeout: Optional[float] = None
    ) -> BufferAgent:
        localAgent = LocalAgent(self._act_executor)
        return localAgent
