# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import math

import pytest
from smarts.core.opendrive_road_network import OpenDriveRoadNetwork
from smarts.core.scenario import Scenario
from smarts.core.sumo_road_network import SumoRoadNetwork


@pytest.fixture
def sumo_scenario():
    return Scenario(scenario_root="scenarios/intersections/4lane")


@pytest.fixture
def opendrive_scenario():
    return Scenario(scenario_root="scenarios/opendrive")


def test_sumo_map(sumo_scenario):
    road_map = sumo_scenario.road_map
    assert isinstance(road_map, SumoRoadNetwork)

    point = (125.20, 139.0, 0)
    lane = road_map.nearest_lane(point)
    assert lane.lane_id == "edge-north-NS_0"
    assert lane.road.road_id == "edge-north-NS"
    assert lane.index == 0
    assert lane.road.point_on_road(point)

    right_lane, direction = lane.lane_to_right
    assert not right_lane

    left_lane, direction = lane.lane_to_left
    assert left_lane
    assert direction
    assert left_lane.lane_id == "edge-north-NS_1"
    assert left_lane.index == 1

    lefter_lane, direction = left_lane.lane_to_left
    assert not lefter_lane

    on_roads = lane.road.oncoming_roads_at_point(point)
    assert on_roads
    assert len(on_roads) == 1
    assert on_roads[0].road_id == "edge-north-SN"

    reflinept = lane.to_lane_coord(point)
    assert reflinept.s == 1.0
    assert reflinept.t == 0.0

    offset = reflinept.s
    assert lane.width_at_offset(offset) == 3.2
    assert lane.curvature_radius_at_offset(offset) == math.inf

    on_lanes = lane.oncoming_lanes_at_offset(offset)
    assert not on_lanes
    on_lanes = left_lane.oncoming_lanes_at_offset(offset)
    assert len(on_lanes) == 1
    assert on_lanes[0].lane_id == "edge-north-SN_1"

    in_lanes = lane.incoming_lanes
    assert not in_lanes

    out_lanes = lane.outgoing_lanes
    assert out_lanes
    assert len(out_lanes) == 2
    assert out_lanes[0].lane_id == "edge-west-EW_0"
    assert out_lanes[1].lane_id == "edge-south-NS_0"

    foes = lane.foes
    assert foes
    assert len(foes) == 6
    foe_set = set(f.lane_id for f in foes)
    assert "edge-west-WE_0" in foe_set
    assert "edge-east-EW_0" in foe_set
    assert ":junction-intersection_0_0" in foe_set
    assert ":junction-intersection_1_0" in foe_set
    assert ":junction-intersection_5_0" in foe_set
    assert ":junction-intersection_12_0" in foe_set

    r1 = road_map.road_by_id("edge-north-NS")
    assert r1
    r2 = road_map.road_by_id("edge-east-WE")
    assert r2

    routes = road_map.generate_routes(r1, r2)
    assert routes
    assert len(routes[0].roads) == 4

    route = routes[0]
    db = route.distance_between(point, (198, 65.20, 0))
    assert db == 134.01


def test_opendrive_map():
    road_map = OpenDriveRoadNetwork.from_file(
        "/home/saul/code/SMARTS/scenarios/opendrive/map.xodr"
    )
    # road_map = opendrive_scenario.road_map
    assert isinstance(road_map, OpenDriveRoadNetwork)

    r1 = road_map.road_by_id("0")
    assert r1
    assert r1.is_junction == False
    assert r1.length == 103
    assert len(r1.lanes) == 8

    l1 = road_map.lane_by_id("0_0_1")
    assert l1
    assert l1.road.road_id == "0"
    assert l1.index == 1
    assert len(l1.lanes_in_same_direction) == 3

    right_lane, direction = l1.lane_to_right
    assert right_lane
    assert direction
    assert right_lane.lane_id == "0_0_2"
    assert right_lane.index == 2

    # left_lane, direction = l1.lane_to_left
    # assert not left_lane

    # further_right_lane, direction = right_lane.lane_to_right
    # assert further_right_lane
    # assert direction
    # assert further_right_lane.lane_id == "0_0_3"
    # assert further_right_lane.index == 3

    # l1_in_lanes = l1.incoming_lanes
    # assert not l1_in_lanes

    # l1_out_lanes = l1.outgoing_lanes
    # assert l1_out_lanes
    # assert len(l1_out_lanes) == 3
    # assert l1_out_lanes[0].lane_id == "3_0_-1"
    # assert l1_out_lanes[1].lane_id == "8_0_-1"
    # assert l1_out_lanes[2].lane_id == "15_0_-1"

    # l2 = road_map.lane_by_id("0_0_-1")
    # assert l2
    # assert l2.road.road_id == "0"
    # assert l2.index == -1
    # l2_in_lanes = l2.incoming_lanes
    # assert l2_in_lanes
    # assert len(l2_in_lanes) == 3
    # assert l2_in_lanes[0].lane_id == "5_0_-1"
    # assert l2_in_lanes[1].lane_id == "7_0_-1"
    # assert l2_in_lanes[2].lane_id == "9_0_-1"

    # l2_out_lanes = l2.outgoing_lanes
    # assert not l2_out_lanes

    # l3 = road_map.lane_by_id("9_0_-1")
    # foes = l3.foes
    # assert foes
    # assert len(foes) == 2
    # foe_set = set(f.lane_id for f in foes)
    # assert "7_0_-1" in foe_set
    # assert "5_0_-1" in foe_set
