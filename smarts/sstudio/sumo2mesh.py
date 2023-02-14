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
import argparse

from smarts.core.sumo_road_network import SumoRoadNetwork
from smarts.sstudio.types import MapSpec


def generate_glb_from_sumo_file(sumo_net_file: str, out_glb_dir: str):
    """Creates a geometry file from a sumo map file."""
    map_spec = MapSpec(sumo_net_file)
    road_network = SumoRoadNetwork.from_spec(map_spec)
    road_network.to_glb(out_glb_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "sumo2mesh.py",
        description="Utility to export sumo road networks to mesh files.",
    )
    parser.add_argument("net", help="sumo net file (*.net.xml)", type=str)
    parser.add_argument("output_path", help="where to write the mesh file", type=str)
    args = parser.parse_args()

    generate_glb_from_sumo_file(args.net, args.output_path)
