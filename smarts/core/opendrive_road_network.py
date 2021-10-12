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
import logging

from lxml import etree
from cached_property import cached_property
from typing import List, Sequence, Tuple
from opendrive2lanelet.opendriveparser.elements.opendrive import OpenDrive
from opendrive2lanelet.opendriveparser.elements.road import Road as RoadElement
from opendrive2lanelet.opendriveparser.elements.roadLanes import Lane as LaneElement
from opendrive2lanelet.opendriveparser.parser import parse_opendrive

from smarts.core.road_map import RoadMap


class OpenDriveRoadNetwork(RoadMap):
    def __init__(self, network: OpenDrive, xodr_file: str):
        self._log = logging.getLogger(self.__class__.__name__)
        self._network = network
        self._xodr_file = xodr_file
        self._lanes = {}
        self._roads = {}
        self._lanepoints = None
        self._junctions = network.junctions
        self._junction_connections = {}
        self._precompute_junction_connections()

    @classmethod
    def from_file(
        cls,
        xodr_file,
    ):
        with open(xodr_file, "r") as f:
            network = parse_opendrive(etree.parse(f).getroot())

        return cls(network, xodr_file)

    def _precompute_junction_connections(self):
        for road in self._network.roads:
            if road.junction:
                for lane in road.lanes.lane_sections[0].allLanes:
                    lane_id = str(road.id) + "_" + str(0) + "_" + str(lane.id)
                    if lane_id not in self._junction_connections:
                        self._junction_connections[lane_id] = [[], []]
                    if lane.link.predecessorId:
                        road_predecessor = road.link.predecessor
                        road_elem = self._network.getRoad(road_predecessor.element_id)
                        last_ls_index = road_elem.lanes.getLastLaneSectionIdx()
                        pred_lane_id = (
                            str(road_predecessor.element_id)
                            + "_"
                            + str(last_ls_index)
                            + "_"
                            + str(lane.link.predecessorId)
                        )
                        if pred_lane_id not in self._junction_connections:
                            self._junction_connections[pred_lane_id] = [[], [lane_id]]
                        else:
                            self._junction_connections[pred_lane_id][1].append(lane_id)

                        self._junction_connections[lane_id][0].append(pred_lane_id)

                    if lane.link.successorId:
                        road_successor = road.link.successor
                        succ_lane_id = (
                            str(road_successor.element_id)
                            + "_"
                            + str(0)
                            + "_"
                            + str(lane.link.successorId)
                        )
                        if succ_lane_id not in self._junction_connections:
                            self._junction_connections[succ_lane_id] = [[lane_id], []]
                        else:
                            self._junction_connections[succ_lane_id][0].append(lane_id)

                        self._junction_connections[lane_id][1].append(succ_lane_id)

    @property
    def source(self) -> str:
        """ This is the .xodr file of the OpenDRIVE map. """
        return self._xodr_file

    @property
    def junction_connections(self):
        return self._junction_connections

    def get_junction(self, junction_id):
        return self._network.getJunction(junction_id)

    class Road(RoadMap.Road):
        def __init__(self, road_id: str, road_elem: RoadElement, road_map):
            self._road_id = road_id
            self._road_elem = road_elem
            self._predecessor_elem = self._road_elem.link.predecessor
            self._successor_elem = self._road_elem.link.successor
            self._lane_sections = self._road_elem.lanes.lane_sections
            self._map = road_map

        @property
        def predecessor(self):
            return self._predecessor_elem

        @property
        def successor(self):
            return self._successor_elem

        @property
        def lane_sections(self):
            return self._lane_sections

        @cached_property
        def is_junction(self) -> bool:
            if self._road_elem.junction:
                return True
            return False

        @cached_property
        def length(self) -> float:
            return self._road_elem._length

        @property
        def road_id(self) -> str:
            return self._road_id

        def _find_junction_connections(self, junction):
            junction_conns = []
            for connection in junction.connections:
                if connection.incomingRoad == self.road_id:
                    junction_conns.append(
                        self._map.road_by_id(str(connection.connectingRoad))
                    )
            return junction_conns

        @cached_property
        def lanes(self) -> List[RoadMap.Lane]:
            lanes = []
            for i in range(len(self._lane_sections)):
                for od_lane in self._lane_sections[i].allLanes:
                    lane_id = self.road_id + "_" + str(i) + "_" + str(od_lane.id)
                    lanes.append(self._map.lane_by_id(lane_id))
            return lanes

        @cached_property
        def left_lanes(self) -> List[RoadMap.Lane]:
            left_lanes = []
            for i in range(len(self._lane_sections)):
                for od_lane in self._lane_sections[i].leftLanes:
                    lane_id = self.road_id + "_" + str(i) + "_" + str(od_lane.id)
                    left_lanes.append(self._map.lane_by_id(lane_id))
            return left_lanes

        @cached_property
        def right_lanes(self) -> List[RoadMap.Lane]:
            right_lanes = []
            for i in range(len(self._lane_sections)):
                for od_lane in self._lane_sections[i].rightLanes:
                    lane_id = self.road_id + "_" + str(i) + "_" + str(od_lane.id)
                    right_lanes.append(self._map.lane_by_id(lane_id))
            return right_lanes

        @cached_property
        def centre_lanes(self) -> List[RoadMap.Lane]:
            centre_lanes = []
            for i in range(len(self._lane_sections)):
                for od_lane in self._lane_sections[i].centreLanes:
                    lane_id = self.road_id + "_" + str(i) + "_" + str(od_lane.id)
                    centre_lanes.append(self._map.lane_by_id(lane_id))
            return centre_lanes

        def lane_at_index(self, index: int) -> RoadMap.Lane:
            return self.lanes[index]

        @cached_property
        def incoming_roads(self) -> List[RoadMap.Road]:
            in_roads = []
            if self._predecessor_elem:
                if self._predecessor_elem.elementType == "road":
                    in_roads.append(
                        self._map.road_by_id(str(self._predecessor_elem.element_id))
                    )
                elif self._predecessor_elem.elementType == "junction":
                    junction = self._map.get_junction(self._predecessor_elem.element_id)
                    in_roads.extend(self._find_junction_connections(junction))

            return in_roads

        @cached_property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            og_roads = []
            if self._successor_elem:
                if self._successor_elem.elementType == "road":
                    og_roads.append(
                        self._map.road_by_id(str(self._successor_elem.element_id))
                    )
                elif self._successor_elem.elementType == "junction":
                    junction = self._map.get_junction(self._successor_elem.element_id)
                    og_roads.extend(self._find_junction_connections(junction))
            return og_roads

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        if road:
            return road
        road_elem = self._network.getRoad(int(road_id))
        if not road_elem:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown road_id '{road_id}'"
            )
            return None
        road = OpenDriveRoadNetwork.Road(road_id, road_elem, self)
        self._roads[road_id] = road
        return road

    class Lane(RoadMap.Lane):
        def __init__(self, lane_id: str, lane_elem: LaneElement, road_map):
            self._lane_id = lane_id
            self._map = road_map
            self._road = road_map.road_by_id(str(lane_elem.parentRoad.id))
            assert self._road
            self._lane_elem = lane_elem
            self._curr_lane_section = self._lane_elem.lane_section
            self.type = self._lane_elem.type

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @property
        def in_junction(self) -> bool:
            return self._road.is_junction

        @cached_property
        def index(self) -> int:
            return self._lane_elem.id

        @cached_property
        def lanes_in_same_direction(self) -> List[RoadMap.Lane]:
            if not self.in_junction:
                # When not in an intersection, all Opendrive Lanes for a Road with same index sign
                # go in same direction.
                same_direction_lanes = []
                lane_section_id = self.lane_id.split("_")[1]
                if self.index > 0:
                    for l in self._curr_lane_section.allLanes:
                        if l.id > 0 and l.id != self.index:
                            lane_id = (
                                self.road.road_id
                                + "_"
                                + lane_section_id
                                + "_"
                                + str(l.id)
                            )
                            same_direction_lanes.append(self._map.lane_by_id(lane_id))
                elif self.index < 0:
                    for l in self._curr_lane_section.allLanes:
                        if l.id < 0 and l.id != self.index:
                            lane_id = (
                                self.road.road_id
                                + "_"
                                + lane_section_id
                                + "_"
                                + str(l.id)
                            )
                            same_direction_lanes.append(self._map.lane_by_id(lane_id))
                return same_direction_lanes
            result = []
            in_roads = set(il.road for il in self.incoming_lanes)
            out_roads = set(il.road for il in self.outgoing_lanes)
            for lane in self.road.lanes:
                if self == lane:
                    continue
                other_in_roads = set(il.road for il in lane.incoming_lanes)
                if in_roads & other_in_roads:
                    other_out_roads = set(il.road for il in self.outgoing_lanes)
                    if out_roads & other_out_roads:
                        result.append(lane)
            return result

        @cached_property
        def lane_to_left(self) -> Tuple[RoadMap.Lane, bool]:
            if self.index == 0:
                return None, True
            result = None
            if self.index > 0:
                for other in self.lanes_in_same_direction:
                    if self.index - other.index == 1:
                        result = other
                        break
            elif self.index < 0:
                for other in self.lanes_in_same_direction:
                    if self.index - other.index == -1:
                        result = other
                        break
            return result, True

        @cached_property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
            if self.index == 0:
                return None, True
            result = None
            if self.index > 0:
                for other in self.lanes_in_same_direction:
                    if self.index - other.index == -1:
                        result = other
                        break
            elif self.index < 0:
                for other in self.lanes_in_same_direction:
                    if self.index - other.index == 1:
                        result = other
                        break
            return result, True

        @cached_property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            il = []
            if self.lane_id in self._map.junction_connections:
                for pred_lane_id in self._map.junction_connections[self.lane_id][0]:
                    il.append(self._map.lane_by_id(pred_lane_id))

            if self.in_junction:
                return il

            lane_link = self._lane_elem.link
            if lane_link.predecessorId:
                ls_index = self._curr_lane_section.idx
                if ls_index == 0:
                    road_predecessor = self._road.predecessor
                    if road_predecessor and road_predecessor.elementType == "road":
                        pred_road = self._map.road_by_id(
                            str(road_predecessor.element_id)
                        )
                        if len(pred_road.lane_sections) > 1:
                            last_ls_index = len(pred_road.lane_sections) - 1
                        else:
                            last_ls_index = 0
                        pred_lane_id = (
                            str(road_predecessor.element_id)
                            + "_"
                            + str(last_ls_index)
                            + "_"
                            + str(lane_link.predecessorId)
                        )
                        il.append(self._map.lane_by_id(pred_lane_id))

                else:
                    pred_lane_id = (
                        self._road.road_id
                        + "_"
                        + str(ls_index - 1)
                        + "_"
                        + str(lane_link.predecessorId)
                    )
                    il.append(self._map.lane_by_id(pred_lane_id))

            return il

        @cached_property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            ol = []
            if self.lane_id in self._map.junction_connections:
                for succ_lane_id in self._map.junction_connections[self.lane_id][1]:
                    ol.append(self._map.lane_by_id(succ_lane_id))

            if self.in_junction:
                return ol

            lane_link = self._lane_elem.link
            if lane_link.successorId:
                ls_index = self._curr_lane_section.idx
                if ls_index == len(self._road.lane_sections) - 1:
                    road_successor = self._road.successor
                    if road_successor:
                        if road_successor.elementType == "road":
                            succ_lane_id = (
                                str(road_successor.element_id)
                                + "_"
                                + str(0)
                                + "_"
                                + str(lane_link.successorId)
                            )
                            ol.append(self._map.lane_by_id(succ_lane_id))
                else:
                    succ_lane_id = (
                        self._road.road_id
                        + "_"
                        + str(ls_index + 1)
                        + "_"
                        + str(lane_link.successorId)
                    )
                    ol.append(self._map.lane_by_id(succ_lane_id))

            return ol

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        lane = self._lanes.get(lane_id)
        if lane:
            return lane
        lane_elem = None
        split_lst = lane_id.split("_")
        road_id, lane_section_index, od_lane_id = (
            split_lst[0],
            split_lst[1],
            split_lst[2],
        )
        road = self.road_by_id(road_id)
        if not road:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown road_id '{road_id}'"
            )
            return None
        road_elem = self._network.getRoad(int(road_id))

        lane_section = road_elem.lanes.lane_sections[int(lane_section_index)]
        for od_lane in lane_section.allLanes:
            if od_lane.id == int(od_lane_id):
                lane_elem = od_lane
                break
        if not lane_elem:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown lane_id '{lane_id}'"
            )
            return None
        lane = OpenDriveRoadNetwork.Lane(lane_id, lane_elem, self)
        self._lanes[lane_id] = lane
        return lane
