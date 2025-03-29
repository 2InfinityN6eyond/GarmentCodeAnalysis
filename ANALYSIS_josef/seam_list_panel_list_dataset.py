import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from constants import PYGARMENT_ROOT, DATASET_ROOT, PROJECT_ROOT

sys.path.append(PYGARMENT_ROOT)
import pygarment as pyg




import os 
import sys

# sys.path.append(os.path.dirname(os.getcwd()))
from constants import PYGARMENT_ROOT, DATASET_ROOT

# sys.path.append(PYGARMENT_ROOT)
import pygarment as pyg

import math
import pickle
import numpy as np
import svgpathtools as svgpath
from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset
import random
import trimesh

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


@dataclass
class Stitch :
    panel_0 : str
    edge_0 : int
    panel_1 : str
    edge_1 : int
    stitch_direction : bool
    

class StitchDict:
    def __init__(self, raw_stitch_dict: Dict[int, List[Dict]]):
        """
        Initialize the StitchManager from a raw stitch dictionary.        
        raw_stitch_dict: keys are stitch ids and values are lists of two dictionaries
        with 'panel' and 'edge' keys.
        """
        self._stitches: Dict[int, Stitch] = {}
        for stitch_id, stitch_pair in raw_stitch_dict.items():
            # Each stitch_pair contains two entries representing the two sides of the stitch.
            side0 = stitch_pair[0]
            side1 = stitch_pair[1]
            # Optionally, you could compute stitch_direction from panel names or edges.
            stitch_direction = False  # You can update this logic as needed.
            self._stitches[stitch_id] = Stitch(
                panel_0=side0['panel'],
                edge_0=side0['edge'],
                panel_1=side1['panel'],
                edge_1=side1['edge'],
                stitch_direction=stitch_direction
            )
    
    def reindex(
        self,
        mapping: Dict[int, int],
    ) -> None:
        """
        Re-index the internal dictionary using a provided mapping.
        For example, mapping could be ann_to_img_seam_idx_map or img_to_ann_seam_idx_map.
        """
        new_stitches = {}
        for old_key, stitch in self._stitches.items():
            new_key = mapping.get(old_key, old_key)
            new_stitches[new_key] = stitch
        self._stitches = new_stitches
    
    def __getitem__(self, key: int) -> Stitch:
        return self._stitches[key]
    
    def __setitem__(self, key: int, value: Stitch) -> None:
        self._stitches[key] = value
    
    def items(self):
        return self._stitches.items()

    def __len__(self):
        return len(self._stitches)
    
    def __repr__(self):
        return repr(self._stitches)


class SVGPanel:
    def __init__(self, svg_path: List[svgpath.Path]):
        self.svg_path = svg_path
        
    @property
    def edge_len_list(self) :
        return [edge.length() for edge in self.svg_path]
    
    @property
    def normalized_edge_stt(self, edge_idx : int) :
        return sum(self.edge_len_list[:edge_idx]) / sum(self.edge_len_list)
        
    @property
    def normalized_edge_end(self, edge_idx : int) :
        return sum(self.edge_len_list[:edge_idx+1]) / sum(self.edge_len_list)
    
    def translate(self, dx: float, dy: float) -> None:
        """
        Translate the panel by dx along x-axis and dy along y-axis.
        """
        # In the complex plane, translation by (dx, dy) means adding (dx + dy*j)
        delta = dx + dy * 1j
        self.svg_path = svgpath.Path(*[path.translated(delta) for path in self.svg_path])


    def scale(self, factor: float, pivot: np.ndarray = np.array([0, 0])) -> None:
        """
        Scale the panel by the given factor relative to the pivot.
        By default, the pivot is 0 (the origin), but you can specify a different pivot point.
        """
        pivot_complex = pivot[0] + pivot[1] * 1j
        self.svg_path = svgpath.Path(*[
            (path.translated(-pivot_complex)).scaled(factor).translated(pivot_complex) for path in self.svg_path
        ])
        
    def set_scale_to(self, size : float, use_vert_bbox : bool = False) -> float :
        if use_vert_bbox :
            x1, y1, x2, y2 = self.vert_bbox()
        else :
            x1, y1, x2, y2 = self.bbox()
        width = x2 - x1
        height = y2 - y1

        scale_factor = size / max(width, height)
        self.scale(scale_factor)
        return scale_factor
    
    def set_start_position_at(self, x : float, y : float) -> None :
        start_x = self.svg_path[0].start.real
        start_y = self.svg_path[0].start.imag
        self.translate(x - start_x, y - start_y)
        
    def rotate_clockwise(self, angle_degrees: float, pivot: np.ndarray = np.array([0, 0])) -> None:
        """
        Rotate the entire path by a given angle in degrees around a pivot point.
        
        Args:
            angle_degrees: The angle to rotate the path, in degrees.
            pivot: The point around which to rotate the path. Default is the origin (0+0j).
        """
        pivot_complex = pivot[0] + pivot[1] * 1j
        self.svg_path = svgpath.Path(*[
            path.rotated(angle_degrees, pivot_complex) for path in self.svg_path
        ])
        
    def mirror_horizontal(self) -> None:
        """
        Mirror the panel horizontally (flip the x-axis).
        """
        # self.svg_path = [svgpath.Path(*[self._mirror_horizontal_segment(seg) for seg in path]) for path in self.svg_path]
        self.svg_path = svgpath.Path(*[self._mirror_horizontal_segment(seg) for seg in self.svg_path])        

    def mirror_vertical(self) -> None:
        """
        Mirror the panel vertically (flip the y-axis).
        """
        # self.svg_path = [svgpath.Path(*[self._mirror_vertical_segment(seg) for seg in path]) for path in self.svg_path]
        self.svg_path = svgpath.Path(*[self._mirror_vertical_segment(seg) for seg in self.svg_path])

    def _mirror_horizontal_segment(self, segment):
        if isinstance(segment, svgpath.Arc):
            # Mirror the start and end points horizontally
            start = complex(-segment.start.real, segment.start.imag)
            end = complex(-segment.end.real, segment.end.imag)
            # Reverse the sweep flag
            sweep = not segment.sweep
            return svgpath.Arc(start=start, radius=segment.radius, rotation=segment.rotation,
                       large_arc=segment.large_arc, sweep=sweep, end=end)
        else:
            return segment.scaled(-1, 1)

    def _mirror_vertical_segment(self, segment):
        if isinstance(segment, svgpath.Arc):
            # Mirror the start and end points vertically
            start = complex(segment.start.real, -segment.start.imag)
            end = complex(segment.end.real, -segment.end.imag)
            # Reverse the sweep flag
            sweep = not segment.sweep
            return svgpath.Arc(start=start, radius=segment.radius, rotation=segment.rotation,
                       large_arc=segment.large_arc, sweep=sweep, end=end)
        else:
            return segment.scaled(1, -1)

    def draw(
        self,
        ax          : plt.Axes = None,
        panel_name  : str = None,
        N_SAMPLE_PER_EDGE : int = 80,
        stitch_list : List[int] = None,
        edge_color_list : List[Tuple[float, float, float]] = None
    ) -> None:
        if stitch_list is not None :
            assert len(stitch_list) == len(self.svg_path), "stitch_list must be the same length as the number of edges"
        if ax is None :
            ax = plt.gca()
        if panel_name is not None :
            ax.set_title(panel_name)
        if edge_color_list is None :
            edge_color_list = plt.cm.rainbow(np.linspace(0, 1, len(self.svg_path)))
         
        path_start_pos = np.array([self.svg_path[0].start.real, self.svg_path[0].start.imag])   
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ax.annotate(
                "0",
                path_start_pos + np.array([dx, dy]),
                color = "white", fontweight = "bold",
                fontsize = 9
            )
        ax.annotate(
            "0",
            path_start_pos,
            color = "black", # fontweight = "bold",
            fontsize = 9
        )
        for edge_idx, (edge, edge_color) in enumerate(zip(self.svg_path, edge_color_list)) :
            ax.add_patch(
                FancyArrowPatch(
                    [edge.start.real, edge.start.imag],
                    [edge.end.real, edge.end.imag],
                    arrowstyle='-|>',
                    mutation_scale=15,
                    color = "black",
                    linewidth = 0.5
                )
            )
            ax.scatter(
                list(map(lambda t : edge.point(t).real, np.linspace(0, 1, N_SAMPLE_PER_EDGE))),
                list(map(lambda t : edge.point(t).imag, np.linspace(0, 1, N_SAMPLE_PER_EDGE))),
                s = 0.5, color = edge_color
            )
            if stitch_list is not None :
                # for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                #     ax.annotate(
                #         f"{stitch_list[edge_idx]}",
                #         [
                #             (edge.start.real + edge.end.real) / 2 + dx,
                #             (edge.start.imag + edge.end.imag) / 2 + dy
                #         ],
                #         color = "white", # fontweight = "bold",
                #         fontsize = 11, fontweight = "bold"
                #     )
                ax.annotate(
                    f"{stitch_list[edge_idx]}",
                    [
                        (edge.start.real + edge.end.real) / 2,
                        (edge.start.imag + edge.end.imag) / 2
                    ],
                    color = edge_color,
                    fontsize = 11, fontweight = "bold"
                )
        ax.invert_yaxis()
        ax.axis("equal")
        
    def is_clockwise(self, n_sample_per_edge : int = 10) -> bool:
        """
        Determine if the path is clockwise.
        """
        total = 0
        for segment in self.svg_path:
            t_list = np.linspace(0, 1, n_sample_per_edge)
            sampled_points = np.array(list(map(
                lambda t : [segment.point(t).real, segment.point(t).imag],
                t_list
            )))
            
            total += np.sum(
                (
                    sampled_points[1:, 0] - sampled_points[:-1, 0]
                ) * (
                    sampled_points[1:, 1] + sampled_points[:-1, 1]
                )
            )
            
            # start = segment.start
            # end = segment.end
            # total += (end.real - start.real) * (end.imag + start.imag)
        return total <= 0


    def reverse_path(self) -> None:
        """
        Reverse the order of the path segments and their directions.
        """
        reversed_segments = []
        for segment in reversed(self.svg_path):
            reversed_segment = segment.reversed()
            reversed_segments.append(reversed_segment)
        self.svg_path = svgpath.Path(*reversed_segments)
        
    def set_start(self, idx: int) -> None :
        """
        Set the start point of the path to the point at index idx.
        """
        self.svg_path = svgpath.Path(*(self.svg_path[idx:] + self.svg_path[:idx]))

    def min_distance_to_point(self, point: complex) -> float:
        """
        Calculate the minimum distance from any point on the panel to a given point.
        
        Args:
            point: A complex number representing the point in the complex plane.
            
        Returns:
            float: The minimum distance from any point on the panel to the given point.
        """
        min_distance = float('inf')
        for path in self.svg_path:
            for segment in path:
                distance = segment.distance(point)
                if distance < min_distance:
                    min_distance = distance
        return min_distance
    
    def bbox(self) :
        """
        (xmin, ymin, xmax, ymax)
        """
        xmin, xmax, ymin, ymax = svgpath.Path(*self.svg_path).bbox()
        return xmin, ymin, xmax, ymax
    
    def vert_bbox(self) :
        vert_list = []
        for edge in self.svg_path :
            vert_list.append([edge.start.real, edge.start.imag])
        vert_arr = np.array(vert_list)
        # print(vert_arr)
        xmin = np.min(vert_arr[:, 0])
        xmax = np.max(vert_arr[:, 0])
        ymin = np.min(vert_arr[:, 1])
        ymax = np.max(vert_arr[:, 1])
        return xmin, ymin, xmax, ymax
    
    def get_center(self) :
        xmin, ymin, xmax, ymax = self.bbox()
        return np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    
    def __repr__(self):
        return f"Panel(svg_path={self.svg_path})"

    def approximate_quadratic_bezier_with_cubic_bezier(self, VIS : bool = False) :
        for i, edge in enumerate(self.svg_path):
            if isinstance(edge, svgpath.QuadraticBezier):
                # Convert quadratic to cubic using the standard formula:
                # CP1 = start + 2/3 * (control - start)
                # CP2 = end + 2/3 * (control - end)
                start = edge.start
                end = edge.end
                control = edge.control
                
                cp1 = start + (control - start) * (2/3)
                cp2 = end + (control - end) * (2/3)
                
                self.svg_path[i] = svgpath.CubicBezier(start, cp1, cp2, end)
                
                if VIS :
                    t_list = np.linspace(0, 1, 100)
                    plt.figure(figsize=(10, 10))
                    plt.title("quadratic => cubic bezier")
                    plt.plot(
                        list(map(lambda t : edge.point(t).real, t_list)),
                        list(map(lambda t : edge.point(t).imag, t_list)),
                        color = "black"
                    )
                    plt.plot(
                        list(map(lambda t : self.svg_path[i].point(t).real, t_list)),
                        list(map(lambda t : self.svg_path[i].point(t).imag, t_list)),
                        color = "red"
                    )
                    plt.axis("equal")
                    plt.show()
        return self
    
    def approximate_arc_with_cubic_bezier(self, VIS : bool = False) :
        for i, edge in enumerate(self.svg_path):
            if isinstance(edge, svgpath.Arc):
                x1 = edge.start.real
                y1 = edge.start.imag
                x2 = edge.end.real
                y2 = edge.end.imag
                rx = edge.radius.real
                ry = edge.radius.imag
                rotation = edge.rotation
                large_arc = edge.large_arc
                sweep = edge.sweep
                
                phi_rad = math.radians(rotation)
                cx = (x1 - x2) / 2.0
                cy = (y1 - y2) / 2.0
                
                            
                # x1', y1'
                x1p = math.cos(phi_rad)*cx + math.sin(phi_rad)*cy
                y1p = -math.sin(phi_rad)*cx + math.cos(phi_rad)*cy
                
                # Step 2: rx,ry 스케일링 체크 (여기서는 필요 없으리라 가정)
                lam = (x1p**2)/(rx**2) + (y1p**2)/(ry**2)
                if lam > 1:
                    scale = math.sqrt(lam)
                    rx *= scale
                    ry *= scale

                # Step 3: 계수 (SVG 사양에 따른)
                num = rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2
                den = rx**2 * y1p**2 + ry**2 * x1p**2
                # 부동소수점 오차로 음수 방지
                factor = math.sqrt(max(0, num/den)) if den != 0 else 0
                # SVG 사양에 따라, large_arc와 sweep 플래그가 같으면 부호 반전

                if large_arc == sweep:
                    factor = -factor
                    
                    
                cxp = factor * (rx * y1p / ry)
                cyp = factor * (-ry * x1p / rx)

                # Step 4: 원래 좌표계로 복원 (phi=0이면 회전 없이 중간점에 더함)
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0
                cx = math.cos(phi_rad)*cxp - math.sin(phi_rad)*cyp + mid_x
                cy = math.sin(phi_rad)*cxp + math.cos(phi_rad)*cyp + mid_y

                # Step 5: 시작각과 끝각 (원 중심 기준)
                theta_start = math.atan2(y1 - cy, x1 - cx)
                theta_end   = math.atan2(y2 - cy, x2 - cx)
                
                # Step 6: 아크의 진행각 delta_theta 결정 (SVG 사양)
                delta_theta = theta_end - theta_start
                
                if sweep:
                    if delta_theta < 0:
                        delta_theta += 2*math.pi
                else:
                    if delta_theta > 0:
                        delta_theta -= 2*math.pi
        
                P0 = (cx + rx * math.cos(theta_start), cy + ry * math.sin(theta_start))
                P3 = (cx + rx * math.cos(theta_end),   cy + ry * math.sin(theta_end))
        
                # Cubic Bézier 공식: k = (4/3)*tan(delta_theta/4)
                k = (4.0/3.0) * math.tan(delta_theta/4.0)
                
                # 컨트롤 포인트
                P1 = (P0[0] - k * rx * math.sin(theta_start),
                    P0[1] + k * ry * math.cos(theta_start))
                P2 = (P3[0] + k * rx * math.sin(theta_end),
                    P3[1] - k * ry * math.cos(theta_end))
                # print(P0, P1, P2, P3)
                
                self.svg_path[i] = svgpath.CubicBezier(
                    P0[0] + P0[1] * 1j, P1[0] + P1[1] * 1j, P2[0] + P2[1] * 1j, P3[0] + P3[1] * 1j
                )
                
                if VIS :
                    t_list = np.linspace(0, 1, 100)
                    plt.figure(figsize=(10, 10))
                    plt.title("arc => cubic bezier")
                    plt.plot(
                        list(map(lambda t : edge.point(t).real, t_list)),
                        list(map(lambda t : edge.point(t).imag, t_list)),
                        color = "black"
                    )
                    plt.plot(
                        list(map(lambda t : self.svg_path[i].point(t).real, t_list)),
                        list(map(lambda t : self.svg_path[i].point(t).imag, t_list)),
                        color = "red"
                    )
                    plt.axis("equal")
                    plt.show()

class SewingPattern :
    def __init__(self,
        panel_svg_path_dict : Dict[str, List[svgpath.Path]],
        stitch_dict : Dict[int, List[Dict]],
        panel_name_refine_map : Dict[str, str] = None,
    ) :
        self.panel_dict = {
            panel_name : SVGPanel(panel_svg_path[0])
            for panel_name, panel_svg_path in panel_svg_path_dict.items()
        }
        self.stitch_dict = StitchDict(stitch_dict)
        self.panel_name_refine_map = panel_name_refine_map
    
    def apply_panel_name_refine_map(
        self, panel_name_refine_map : Dict[str, str] = None
    ) :
        if panel_name_refine_map is None :
            panel_name_refine_map = self.panel_name_refine_map
        for panel_name in self.panel_name_list :
            self.panel_dict[panel_name_refine_map[panel_name]] = self.panel_dict[panel_name]
            del self.panel_dict[panel_name]
    @property
    def panel_name_list(self) :
        return list(self.panel_dict.keys())
    @property
    def panel_list(self) :
        return list(self.panel_dict.values())
    
    def set_panel_start(self, panel_name : str, start_idx : int) :
        panel_edge_count = len(self.panel_dict[panel_name].svg_path)
        panel_edge_idx_map = {
            idx : (idx - start_idx) % panel_edge_count
            for idx in range(panel_edge_count)
        }
        for stch_id, stitch in self.stitch_dict.items() :
            if stitch.panel_0 == panel_name :
                stitch.edge_0 = panel_edge_idx_map[stitch.edge_0]
            if stitch.panel_1 == panel_name :
                stitch.edge_1 = panel_edge_idx_map[stitch.edge_1]
        
        self.panel_dict[panel_name].set_start(start_idx)
        
    
    def reverse_panel_path(self, panel_name : str) :
        panel_edge_len = len(self.panel_dict[panel_name].svg_path)
        self.panel_dict[panel_name].reverse_path()
        for stch_id, stitch in self.stitch_dict.items() :
            if stitch.panel_0 == panel_name :
                stitch.edge_0 = panel_edge_len - 1 - stitch.edge_0
            if stitch.panel_1 == panel_name :
                stitch.edge_1 = panel_edge_len - 1 - stitch.edge_1
       
    def mirror_panel_horizontally(self, panel_name : str) :
        self.panel_dict[panel_name].mirror_horizontal()

    # def mirror_back_panel_horizontally(self) :
    #     for panel_name, panel in self.panel_dict.items() :
    #         if (
    #             panel_name in self.panel_name_refine_map
    #         ) and (
    #             "back" in self.panel_name_refine_map[panel_name]
    #         ) :
    #             panel.mirror_horizontal()
       
    # def unifiy_loop_direction(self, clockwise_only : bool = True) :
    #     for panel_name, panel in self.panel_dict.items() :
    #         if panel.is_clockwise() != clockwise_only :
    #             self.reverse_panel_path(panel_name)
       
    def draw(
        self,
        # ax : plt.Axes = None,
        FIGLEN : int = 5,
        N_SAMPLE_PER_EDGE : int = 80, 
        show=False
    ) :
        NROWS = int(np.ceil(len(self.panel_dict) ** 0.5))
        NCOLS = int(np.ceil(len(self.panel_dict) / NROWS))
        plt.figure(figsize=(FIGLEN*NCOLS, FIGLEN*NROWS))
        
        color_list = plt.cm.rainbow(np.linspace(0, 1, len(self.stitch_dict)))
        for panel_idx, (panel_name, panel) in enumerate(self.panel_dict.items()) :
            stitch_idx_list = []
            edge_color_list = []
            for edge_idx, edge in enumerate(panel.svg_path) :
                stitch_idx = -1
                for stitch_id, stitch in self.stitch_dict.items() :
                    if (
                        stitch.panel_0 == panel_name and stitch.edge_0 == edge_idx
                    ) or (
                        stitch.panel_1 == panel_name and stitch.edge_1 == edge_idx
                    ):
                        stitch_idx = stitch_id
                        break
                stitch_idx_list.append(stitch_idx)
                edge_color_list.append(color_list[stitch_idx] if stitch_idx != -1 else "black")
            
            # print(len(panel.svg_path), len(edge_color_list), len(stitch_idx_list))
            
            
            ax = plt.subplot(NROWS, NCOLS, panel_idx + 1)
            panel.draw(
                ax, panel_name, N_SAMPLE_PER_EDGE,
                stitch_idx_list, edge_color_list
            )
        if show :
            plt.show()
            
    def get_panel_stch_idx_list(self, panel_name : str) :
        panel_stch_idx_list = []
        for edge_idx in range(len(self.panel_dict[panel_name].svg_path)) :
            stitch_idx = -1
            for stch_idx, stitch in self.stitch_dict.items() :
                if (
                    stitch.panel_0 == panel_name and stitch.edge_0 == edge_idx
                ) or (
                    stitch.panel_1 == panel_name and stitch.edge_1 == edge_idx
                ):
                    stitch_idx = stch_idx
                    break
            panel_stch_idx_list.append(stitch_idx)
        return panel_stch_idx_list
    
    
@dataclass
class ParameterizedSeamLine :
    stch_idx : int = None
    
    whole_stch_vert_idx_arr : np.ndarray = None
    whole_stch_vert_vis_mask : np.ndarray = None
    # whole_stch_vert_projected_pos_arr : np.ndarray = None
    
    segment_vert_idx_arr_list : List[np.ndarray] = None
    segment_vert_pos_arr_list : List[np.ndarray] = None
    segment_edge_len_arr_list : List[np.ndarray] = None
    segment_t_arr_list : List[np.ndarray] = None
    segment_u_arr_list : List[np.ndarray] = None
    segment_v_arr_list : List[np.ndarray] = None
    
    def translate(self, dx : float, dy : float) :
        for segment_vert_pos_arr in self.segment_vert_pos_arr_list :
            segment_vert_pos_arr[:, 0] += dx
            segment_vert_pos_arr[:, 1] += dy

    def reverse_order(self) :
        self.whole_stch_vert_idx_arr = self.whole_stch_vert_idx_arr[::-1]
        self.whole_stch_vert_vis_mask = self.whole_stch_vert_vis_mask[::-1]
        self.segment_vert_idx_arr_list = [
            segment_vert_idx_arr[::-1] for segment_vert_idx_arr in self.segment_vert_idx_arr_list
        ]
        self.segment_vert_pos_arr_list = [
            segment_vert_pos_arr[::-1] for segment_vert_pos_arr in self.segment_vert_pos_arr_list
        ]
        self.segment_edge_len_arr_list = [
            segment_edge_len_arr[::-1] for segment_edge_len_arr in self.segment_edge_len_arr_list
        ]
        self.segment_t_arr_list = [
            1 - segment_t_arr[::-1] for segment_t_arr in self.segment_t_arr_list
        ]
        self.segment_u_arr_list = [
            1 - segment_u_arr[::-1] for segment_u_arr in self.segment_u_arr_list
        ]
        self.segment_v_arr_list = [
            -segment_v_arr[::-1] for segment_v_arr in self.segment_v_arr_list
        ]
    
    def reorder_segments(
        self, order_f
    ) :
        for segment_idx in range(len(self.segment_vert_pos_arr_list)) :
            if not order_f(self.segment_vert_pos_arr_list[segment_idx]) :
                self.segment_vert_pos_arr_list[segment_idx] = self.segment_vert_pos_arr_list[segment_idx][::-1]
                self.segment_edge_len_arr_list[segment_idx] = self.segment_edge_len_arr_list[segment_idx][::-1]
                self.segment_t_arr_list[segment_idx] = 1 - self.segment_t_arr_list[segment_idx][::-1]
                self.segment_u_arr_list[segment_idx] = 1 - self.segment_u_arr_list[segment_idx][::-1]
                self.segment_v_arr_list[segment_idx] = -self.segment_v_arr_list[segment_idx][::-1]

@dataclass
class ParameterizedEdgeLine :
    pass
    
class SingleViewLabel :
    """
    class for label of single image
    """    
    def __init__(self,
        img,
        img_foreground_mask : np.ndarray,
        vert_visibility_mask : np.ndarray,
        vert_projected_pos_arr : np.ndarray,
        fltrd_vis_seam_line_dict : Dict[int, Dict[str, ParameterizedSeamLine]],
    ) :
        self.img = img.convert("RGB")
        self.img_foreground_mask = img_foreground_mask
        self.vert_visibility_mask = vert_visibility_mask
        self.vert_projected_pos_arr = vert_projected_pos_arr
        
        self.seam_line_dict = {}
        for seam_line_idx, seam_line in fltrd_vis_seam_line_dict.items() :
            self.seam_line_dict[seam_line_idx] = ParameterizedSeamLine(
                stch_idx = seam_line_idx,
                whole_stch_vert_idx_arr = seam_line["raw_idx_arr"],
                whole_stch_vert_vis_mask = seam_line["raw_vis_mask"],
                segment_vert_idx_arr_list = seam_line["segment_idx_arr_list"],
                segment_vert_pos_arr_list = seam_line["segment_pos_arr_list"],
                segment_edge_len_arr_list = seam_line["segment_edge_len_arr_list"],
                segment_t_arr_list = seam_line["segment_t_arr_list"],
                segment_u_arr_list = seam_line["segment_u_arr_list"],
                segment_v_arr_list = seam_line["segment_v_arr_list"],
            )

    def crop(self, crop_l : int, crop_t : int, crop_r : int, crop_b : int) :
        """
        crop_l, crop_t, crop_r, crop_b : int
        """
        self.img = self.img.crop((crop_l, crop_t, crop_r, crop_b))
        self.img_foreground_mask = self.img_foreground_mask[crop_t:crop_b, crop_l:crop_r].copy()
        self.vert_projected_pos_arr[:, 0] -= crop_l
        self.vert_projected_pos_arr[:, 1] -= crop_t
        
        for seam_line_idx, seam_line_info in self.seam_line_dict.items() :
            seam_line_info.translate(-crop_l, -crop_t)

    def pad(
        self, pad_l : int, pad_t : int, pad_r : int, pad_b : int) :
        """
        pad the image and the foreground mask
        """
        self.img = Image.fromarray(np.pad(
            np.array(self.img),
            pad_width=((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
            mode="constant",
            constant_values=255
        ))
        self.img_foreground_mask = np.pad(
            self.img_foreground_mask,
            pad_width=((pad_t, pad_b), (pad_l, pad_r)),
            mode="constant",
            constant_values=0
        )
        
        for seam_line_idx, seam_line_info in self.seam_line_dict.items() : 
            seam_line_info.translate(pad_l, pad_t)
    
    def translate(self, dx : float, dy : float) :
        for seam_line_idx, seam_line_info in self.seam_line_dict.items() :
            seam_line_info.translate(dx, dy)

class UnconstrainedFewViewLabel :
    def __init__(self,
        sewing_pattern : SewingPattern,
        vert_visibility_mask_list : List[np.ndarray],
    ) :
        self.sewing_pattern = sewing_pattern
        self.img = None
        self.seam_line_dict_list = []
        self.vert_visibility_mask_list = vert_visibility_mask_list
        
    def order_seam(self) :
        pass

    def mirror_back_panel_horizontally(self) :
        for panel_name, panel in self.sewing_pattern.panel_dict.items() :
            if (
                panel_name in self.sewing_pattern.panel_name_refine_map
            ) and (
                "back" in self.sewing_pattern.panel_name_refine_map[panel_name]
            ) :
                panel.mirror_horizontal()
    
    def unify_loop_direction(self, clockwise_only : bool = True) :
        for panel_name, panel in self.sewing_pattern.panel_dict.items() :
            if panel.is_clockwise() != clockwise_only :
                self.sewing_pattern.reverse_panel_path(panel_name)

    # def normalize_coord(self, width : int, height : int, resize_img : bool = False) :
    #     if resize_img :
    #         self.img = self.img.resize((width, height))
            


def get_poc_dataset_view_name_list() :
    return ["front", "back", "left", "right"]


def get_bbox_from_mask(img_mask) :
    """
    (x1, y1, x2, y2)
    """
    fg_ycoord, fg_xcoord = np.where(img_mask > 0)
    fg_ycoord_min = np.min(fg_ycoord)
    fg_ycoord_max = np.max(fg_ycoord)
    fg_xcoord_min = np.min(fg_xcoord)
    fg_xcoord_max = np.max(fg_xcoord)
    return (fg_xcoord_min, fg_ycoord_min, fg_xcoord_max, fg_ycoord_max)

def read_poc_files(
    garment_path: str,
    return_data_list = [
        "rendered_image_dict",
        "panel_svg_path_dict",
        "stitch_dict",
        "panel_vertex_mask_dict"
        "vertex_visibility_mask_dict",
        "projected_vertex_pose_dict",
        "fltrd_vis_seam_line_dict",
        "box_mesh",
    ]
) :
    garment_id = os.path.basename(garment_path)

    SPEC_FILE_PATH = os.path.join(garment_path, f"{garment_id}_specification.json")
    pattern = pyg.pattern.wrappers.VisPattern(SPEC_FILE_PATH)

    if "rendered_image_dict" in return_data_list :
        rendered_image_dict = {}
        for side in ["front", "back", "left", "right"] :
            rendered_image_dict[side] = Image.open(os.path.join(garment_path, f"rendered_{side}.png"))
    
    if "panel_svg_path_dict" in return_data_list :
        # Get Garment Blueprint
        panel_svg_path_dict = {
            panel_name : pattern._draw_a_panel(
                panel_name, apply_transform=False, fill=True
            )
            for panel_name in pattern.panel_order()
        }
    if "stitch_dict" in return_data_list :
        stitch_dict = {
            i : v for i, v in enumerate(pattern.pattern['stitches'])
        }
    if "panel_vertex_mask_dict" in return_data_list :
        with open(os.path.join(garment_path, f"panel_vertex_mask_dict.pkl"), "rb") as f :
            panel_vertex_mask_dict = pickle.load(f)            
    if "vertex_visibility_mask_dict" in return_data_list :
        with open(os.path.join(garment_path, f"{garment_id}_vertex_visibility_mask.pkl"), "rb") as f :
            vertex_visibility_mask_dict = pickle.load(f)            
    if "projected_vertex_pose_dict" in return_data_list :
        with open(os.path.join(garment_path, f"{garment_id}_projected_vertex_pose.pkl"), "rb") as f :
            projected_vertex_pose_dict = pickle.load(f)
    if "fltrd_vis_seam_line_dict" in return_data_list :
        with open(os.path.join(garment_path, f"{garment_id}_fltrd_vis_seam_line_dict.pkl"), "rb") as f :
            fltrd_vis_seam_line_dict = pickle.load(f)
    if "box_mesh" in return_data_list :
        box_mesh = trimesh.load_mesh(
            os.path.join(garment_path, f"{garment_id}_boxmesh.ply"),
            process=False
        )
    return [values for key, values in locals().items() if key in return_data_list]


def read_poc_datapoint(
    garment_path, 
    view_name_list = ["front", "back", "left", "right"],
    panel_name_refine_map = None,
    return_data_list = [
        "rendered_image_dict",
        "panel_svg_path_dict",
        "stitch_dict",
        "panel_vertex_mask_dict",
        "vertex_visibility_mask_dict",
        "projected_vertex_pose_dict",
        "fltrd_vis_seam_line_dict",
        # "box_mesh",
    ]
) :
    (
        rendered_image_dict,
        panel_svg_path_dict,
        stitch_dict,
        panel_vertex_mask_dict,
        vertex_visibility_mask_dict,
        projected_vertex_pose_dict,
        fltrd_vis_seam_line_dict,
        # box_mesh,
    ) = read_poc_files(
        garment_path,
        # os.path.join(DATASET_ROOT, garment_path),
        return_data_list,
    )
    view_label_dict = {}
    for side in view_name_list :
        view_label_dict[side] = SingleViewLabel(
            img = rendered_image_dict[side],
            img_foreground_mask = np.array(rendered_image_dict[side].getchannel("A")),
            vert_visibility_mask = vertex_visibility_mask_dict[side],
            vert_projected_pos_arr = projected_vertex_pose_dict[side],
            fltrd_vis_seam_line_dict = fltrd_vis_seam_line_dict[side],
        )
    
    return view_label_dict, SewingPattern(
        panel_svg_path_dict, stitch_dict, panel_name_refine_map
    ), (
        # box_mesh,
        panel_vertex_mask_dict
    )
    
def combine_images_and_labels(
    view_label_list: List[SingleViewLabel],
    sewing_pattern : SewingPattern,
) :
    """
    Can handle max 6 views
    """
    N_VIEWS = len(view_label_list)
    assert N_VIEWS <= 6, "Can handle max 6 views"
    
    combined_label = UnconstrainedFewViewLabel(
        sewing_pattern,
        vert_visibility_mask_list = list(map(
            lambda view_label : view_label.vert_visibility_mask,
            view_label_list
        ))
    )
    
    if len(view_label_list) == 1 :
        IMG_W, IMG_H = view_label_list[0].img.size
        img_bbox_x1, img_bbox_y1, img_bbox_x2, img_bbox_y2 = get_bbox_from_mask(view_label_list[0].img_foreground_mask)
        crop_t = random.randint(0, img_bbox_y1)
        crop_b = random.randint(img_bbox_y2, IMG_H - 1)
        crop_l = random.randint(0, img_bbox_x1)
        crop_r = random.randint(img_bbox_x2, IMG_W - 1)
        view_label_list[0].crop(crop_l, crop_t, crop_r, crop_b)
        combined_label.img = view_label_list[0].img
        combined_label.seam_line_dict_list.append(view_label_list[0].seam_line_dict)
        return combined_label
        
    bb_w_list = []
    bb_h_list = []
    for view_label in view_label_list :
        x1, y1, x2, y2 = get_bbox_from_mask(view_label.img_foreground_mask)
        # view_label.crop(x1, y1, x2, y2)
        bb_w_list.append(x2 - x1)
        bb_h_list.append(y2 - y1)
    if len(view_label_list) in [2, 3] :
        h = np.max(bb_h_list)
        x_pos = 0
        for (view_label, bb_w) in zip(view_label_list, bb_w_list) :
            x1, y1, x2, y2 = get_bbox_from_mask(view_label.img_foreground_mask)
            view_label.crop(x1, y1, x2, y2)
            view_label.pad(0, 0, 0, h - (y2 - y1))
            view_label.translate(x_pos, 0)
            combined_label.seam_line_dict_list.append(
                view_label.seam_line_dict
            )
            x_pos += bb_w
            
        combined_label.img = Image.fromarray(np.hstack([
            np.array(view_label.img) for view_label in view_label_list
        ]))
    elif len(view_label_list) == 4 :
        row1_h = max(bb_h_list[0], bb_h_list[1])
        row1_w = bb_w_list[0] + bb_w_list[1]
        row2_h = max(bb_h_list[2], bb_h_list[3])
        row2_w = bb_w_list[2] + bb_w_list[3]
        w = max(row1_w, row2_w)
        
        x_pos = 0
        for (view_label, bb_w, x_pad) in zip(
            view_label_list[:2], bb_w_list[:2], [0, w - row1_w]
        ) :
            x1, y1, x2, y2 = get_bbox_from_mask(view_label.img_foreground_mask)
            view_label.crop(x1, y1, x2, y2)
            view_label.pad(0, 0, x_pad, row1_h - (y2 - y1))
            view_label.translate(x_pos, 0)
            combined_label.seam_line_dict_list.append(view_label.seam_line_dict)
            x_pos += bb_w
        x_pos = 0
        for (view_label, bb_w, x_pad) in zip(
            view_label_list[2:], bb_w_list[2:], [0, w - row2_w]
        ) :
            x1, y1, x2, y2 = get_bbox_from_mask(view_label.img_foreground_mask)
            view_label.crop(x1, y1, x2, y2)
            view_label.pad(0, 0, x_pad, row2_h - (y2 - y1))
            view_label.translate(x_pos, row1_h)
            combined_label.seam_line_dict_list.append(view_label.seam_line_dict)
            x_pos += bb_w
        combined_label.img = Image.fromarray(np.vstack([
            np.hstack([
                np.array(view_label_list[0].img),
                np.array(view_label_list[1].img),
            ]),
            np.hstack([
                np.array(view_label_list[2].img),
                np.array(view_label_list[3].img),
            ]),
        ]))
    return combined_label
        
def labels_to_token_sequence(
    combined_label,
    sewing_pattern,
) :
    pass










def seam_segment_order_f(
    segment_vert_pos_arr,
    seam_stt_to_end_h_to_w_ratio_thresh = 3
) :
    stt2end_bb_w, stt2end_bb_h = np.abs(segment_vert_pos_arr[-1] - segment_vert_pos_arr[0])
    if stt2end_bb_h / stt2end_bb_w > seam_stt_to_end_h_to_w_ratio_thresh : 
        if segment_vert_pos_arr[0, 1] > segment_vert_pos_arr[-1, 1] :
            return False
        else :
            return True
    else :
        if segment_vert_pos_arr[0, 0] > segment_vert_pos_arr[-1, 0] :
            return False
        else :
            return True
        
        
class SeamListPanelListDataset(Dataset) :
    def __init__(
        self,
        garment_path_list,
        seed = 42,
        enforce_images_n = None,
        
        visible_panel_vert_count_thresh = 80,
        
        seam_stt_to_end_h_to_w_ratio_thresh = 3,
        n_tuv_samples = 30,
        
        seam_start_token = "<dcap>",
        seam_end_token   = "</dcap>",
        side_start_token = "<grounding>",
        side_end_token   = "</grounding>",
        panel_list_start_token = "<proposal>",
        panel_list_end_token   = "</proposal>",
        panel_start_token     = "<poly>",
        panel_end_token       = "</poly>",
        
        return_hidden_state_mask        = False,
        return_tuv_arr_list             = False,
        return_annotation_dict          = False,
        return_img_annotation_list      = False,
        return_panel_annotation_dict    = False,
        return_panel_svg_path_dict      = False,
        return_stitch_dict              = False,
        return_ann_to_img_seam_idx_map  = True,
        return_img_to_ann_seam_idx_map  = True,
    ) :
        
        self.garment_path_list = garment_path_list
        self.enforce_images_n = enforce_images_n
        
        self.visible_panel_vert_count_thresh = visible_panel_vert_count_thresh
        self.seam_stt_to_end_h_to_w_ratio_thresh = seam_stt_to_end_h_to_w_ratio_thresh
    
        self.seam_start_token = seam_start_token
        self.seam_end_token = seam_end_token
        self.side_start_token = side_start_token
        self.side_end_token = side_end_token
        self.panel_list_start_token = panel_list_start_token
        self.panel_list_end_token = panel_list_end_token
        self.panel_start_token = panel_start_token
        self.panel_end_token = panel_end_token
    
        
    
        if seed is not None :
            random.seed(seed)
            np.random.seed(seed)
        
        self.scenario_dict = {
            0 : "Locate a endpoint of a seam line, identify the seam line, and predict a sewing pattern of a garment",
            1 : "Draw a seam line of a garment and identify the seam line"
        }
        
        self.panel_name_refine_map = {
            'left_collar_back'      : " collar left back",
            'left_collar_front'     : " collar left front",
            'right_collar_back'     : " collar right back",
            'right_collar_front'    : " collar right front",
            'left_hood'             : " hood left",
            'right_hood'            : " hood right",
            'pant_b_l'              : " pant left back",
            'pant_b_r'              : " pant right back",
            'pant_f_l'              : " pant left front",
            'pant_f_r'              : " pant right front",
            'pant_l_cuff_b'         : " pant cuff left back",
            'pant_l_cuff_f'         : " pant cuff left front",
            'pant_l_cuff_skirt_b'   : " pant cuff skirt left back",
            'pant_l_cuff_skirt_f'   : " pant cuff skirt left front",
            'pant_r_cuff_b'         : " pant cuff right back",
            'pant_r_cuff_f'         : " pant cuff right front",
            'pant_r_cuff_skirt_b'   : " pant cuff skirt right back",
            'pant_r_cuff_skirt_f'   : " pant cuff skirt right front",
            'left_btorso'           : " torso left back",
            'left_ftorso'           : " torso left front",
            'right_btorso'          : " torso right back",
            'right_ftorso'          : " torso right front",
            'left_sleeve_b'         : " sleeve left back",
            'left_sleeve_f'         : " sleeve left front",
            'skirt_back'            : " skirt back",
            'skirt_front'           : " skirt front",
            'right_sleeve_b'        : " sleeve right back",
            'right_sleeve_f'        : " sleeve right front",
            'sl_left_cuff_b'        : " sleeve cuff left back",
            'sl_left_cuff_f'        : " sleeve cuff left front",
            'sl_left_cuff_skirt_b'  : " sleeve cuff skirt left back",
            'sl_left_cuff_skirt_f'  : " sleeve cuff skirt left front",
            'sl_right_cuff_b'       : " sleeve cuff right back",
            'sl_right_cuff_f'       : " sleeve cuff right front",
            'sl_right_cuff_skirt_b' : " sleeve cuff skirt right back",
            'sl_right_cuff_skirt_f' : " sleeve cuff skirt right front",
            'wb_back'               : " waist band back",
            'wb_front'              : " waist band front",
        }
        self.side_list = get_poc_dataset_view_name_list()
        
    def __len__(self) :
        return len(self.garment_path_list)

    def __getitem__(self, idx) :
        view_label_dict, sewing_pattern, (panel_vert_mask_dict) = read_poc_datapoint(
            self.garment_path_list[idx], self.side_list, self.panel_name_refine_map
        )
        
        n_side    = random.randint(1, 4) if self.enforce_images_n is None else self.enforce_images_n
        sampled_side_list = random.sample(self.side_list, n_side)
        
        combined_label = combine_images_and_labels(
            list(map(lambda x: view_label_dict[x], sampled_side_list)),
            sewing_pattern,
        )
        combined_label.mirror_back_panel_horizontally()
        combined_label.unify_loop_direction()
                
        combined_label.sorted_seam_line_dict_list = []
        for seam_line_dict in combined_label.seam_line_dict_list :
            for seam_line_idx, seam_line in seam_line_dict.items() :
                seam_line.reorder_segments(seam_segment_order_f)
                
            # print(list(combined_label.seam_line_dict_list[0].values())[0].segment_vert_pos_arr_list)
            combined_label.sorted_seam_line_dict_list.append(dict(sorted(
                seam_line_dict.items(),
                key=lambda x : (
                    x[1].segment_vert_pos_arr_list[0][0, 1],
                    x[1].segment_vert_pos_arr_list[0][:, 1].mean()
                )
            )))


        # # ===========================
        # plt.figure(figsize=(20, 20))
        # plt.imshow(combined_label.img)
        # for side_idx, seam_line_dict in enumerate(combined_label.sorted_seam_line_dict_list) :
        #     for seam_idx, seam_dict in seam_line_dict.items() :
        #         for segment_vert_pos_arr in seam_dict.segment_vert_pos_arr_list :
        #             plt.plot(segment_vert_pos_arr[:, 0], segment_vert_pos_arr[:, 1], "r-")
        #             plt.annotate(
        #                 str(seam_idx),
        #                 (segment_vert_pos_arr[0, 0], segment_vert_pos_arr[0, 1]),
        #                 color="green",
        #                 fontsize=15,
        #                 fontweight="bold"
        #             )
        # plt.show()
        # # ===========================


        visible_panel_dict = {}
        for panel_name, panel_vert_mask in panel_vert_mask_dict.items() :
            visible_panel_dict[panel_name] = {"visible_view_count" : 0, "visible_vert_count" : 0}
            for view_idx, vert_visibility_mask in enumerate(combined_label.vert_visibility_mask_list) :
                visible_vert_count = np.sum(vert_visibility_mask & panel_vert_mask)
                if visible_vert_count > self.visible_panel_vert_count_thresh :
                    visible_panel_dict[panel_name]["visible_view_count"] += 1
                    visible_panel_dict[panel_name]["visible_vert_count"] += visible_vert_count

        sorted_visible_panel_dict = dict(sorted(
            visible_panel_dict.items(),
            key=lambda item: (item[1]["visible_view_count"], item[1]["visible_vert_count"]),
            reverse=True
        ))
        
        
        detected_seam_idx_list = []
        for side_idx, seam_line_dict in enumerate(combined_label.sorted_seam_line_dict_list) :
            for seam_idx, seam_dict in seam_line_dict.items() :
                if seam_idx not in detected_seam_idx_list :
                    detected_seam_idx_list.append(seam_idx)
                    
        # # ===========================
        # print(detected_seam_idx_list)
        # # ===========================
                    
        # decide starting vertex of each panel
        # if one of stitching edge is detected as seam, start most first detected edge
        # else, start edge with edge where connecting egde in other panel has occured first
        for panel_idx, (panel_name, panel_info) in enumerate(sorted_visible_panel_dict.items()) :
            edge_stitch_idx_list = combined_label.sewing_pattern.get_panel_stch_idx_list(panel_name)
            start_edge_idx = None
            for detected_seam_idx in detected_seam_idx_list :
                if detected_seam_idx in edge_stitch_idx_list :
                    start_edge_idx = edge_stitch_idx_list.index(detected_seam_idx)
                    break
            if start_edge_idx is None :
                for prev_panel_name, prev_panel_info in sorted_visible_panel_dict.items() :
                    if prev_panel_name == panel_name :
                        break
                    prev_edge_stitch_idx_list = combined_label.sewing_pattern.get_panel_stch_idx_list(prev_panel_name)
                    for prev_edge_stitch_idx in prev_edge_stitch_idx_list :
                        if prev_edge_stitch_idx in edge_stitch_idx_list :
                            start_edge_idx = edge_stitch_idx_list.index(prev_edge_stitch_idx)
                            break
            if start_edge_idx is None : # This should not happen
                start_edge_idx = 0
                
            combined_label.sewing_pattern.set_panel_start(
                panel_name,
                start_edge_idx
            )
            scale_factor = combined_label.sewing_pattern.panel_dict[panel_name].set_scale_to(
                500, use_vert_bbox = True
            )
            combined_label.sewing_pattern.panel_dict[panel_name].set_start_position_at(500, 500)
            panel_info["scale_factor"] = scale_factor
            
        # print(sorted_visible_panel_dict.keys())
        # print(combined_label.sorted_seam_line_dict_list)
        # print(combined_label.sewing_pattern.stitch_dict)

        # combined_label.sewing_pattern.draw()
        # plt.show()

        ann_to_img_seam_idx_map = {-1 : -1}
        img_to_ann_saem_idx_map = {-1 : -1}

        for sorted_seam_dict in combined_label.sorted_seam_line_dict_list :
            for seam_idx, seam_dict in sorted_seam_dict.items() :
                if seam_idx not in ann_to_img_seam_idx_map :
                    ann_to_img_seam_idx_map[seam_idx] = len(ann_to_img_seam_idx_map)
                    img_to_ann_saem_idx_map[len(img_to_ann_saem_idx_map)] = seam_idx

        for panel_name, panel_info in sorted_visible_panel_dict.items() :
            panel_stch_idx_list = combined_label.sewing_pattern.get_panel_stch_idx_list(panel_name)
            for stitch_idx in panel_stch_idx_list :
                if stitch_idx not in ann_to_img_seam_idx_map :
                    ann_to_img_seam_idx_map[stitch_idx] = len(ann_to_img_seam_idx_map)
                    img_to_ann_saem_idx_map[len(img_to_ann_saem_idx_map)] = stitch_idx
        
        ann_to_img_seam_idx_map[-1] = 0
        img_to_ann_saem_idx_map[0]  = -1

        token_primitive_list = [self.seam_start_token]
        for per_side_seam_dict in combined_label.sorted_seam_line_dict_list :
            token_primitive_list.append(self.side_start_token)
            for seam_idx, seam_dict in per_side_seam_dict.items() :
                for segment_vert_pos_arr in seam_dict.segment_vert_pos_arr_list :
                    
                    token_primitive_list.append(str(ann_to_img_seam_idx_map[seam_idx]))
                    
                    x1 = min(
                        int(segment_vert_pos_arr[0, 0]  * 1000 / combined_label.img.width ),
                        999
                    )
                    y1 = min(
                        int(segment_vert_pos_arr[0, 1]  * 1000 / combined_label.img.height),
                        999
                    )
                    x2 = min(
                        int(segment_vert_pos_arr[-1, 0] * 1000 / combined_label.img.width ),
                        999
                    )
                    y2 = min(
                        int(segment_vert_pos_arr[-1, 1] * 1000 / combined_label.img.height),
                        999
                    )
                    
                    token_primitive_list.extend([f"<loc_{x1}>", f"<loc_{y1}>", f"<loc_{x2}>", f"<loc_{y2}>"])
        
            token_primitive_list.append(self.side_end_token)
        token_primitive_list.append(self.seam_end_token)
        token_primitive_list.append(self.panel_list_start_token)
        
        for panel_name, panel_info in sorted_visible_panel_dict.items() :
            token_primitive_list.append(self.panel_start_token)
            
            panel_stch_idx_list = combined_label.sewing_pattern.get_panel_stch_idx_list(panel_name)
            for edge_idx, edge in enumerate(combined_label.sewing_pattern.panel_dict[panel_name].svg_path) :
                
                stch_idx = panel_stch_idx_list[edge_idx]
                token_primitive_list.extend([
                    f"<loc_{min(int(edge.start.real), 999)}>",
                    f"<loc_{min(int(edge.start.imag), 999)}>",
                    str(ann_to_img_seam_idx_map[stch_idx])
                ])
            token_primitive_list.extend([
                f"<loc_{min(int(edge.end.real), 999)}>",
                f"<loc_{min(int(edge.end.imag), 999)}>",
                f"<loc_{min(int(panel_info['scale_factor']), 999)}>",
                # self.panel_end_token
            ])
        # token_primitive_list.append(self.panel_list_end_token)
                
        return (
            self.scenario_dict[0],
            combined_label.img,
            token_primitive_list,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ann_to_img_seam_idx_map,
            img_to_ann_saem_idx_map,
        )
        
        
    def parse_token_primitive_list(self, img, token_primitive_list) :
        if isinstance(img, Image.Image) :
            width, height = img.size
        else :
            width, height = img.shape[1], img.shape[0]
            
        
        
        side_stt_token_idx_list = []
        side_end_token_idx_list = []
        for token_idx, token in enumerate(token_primitive_list) :
            if token == self.side_start_token :
                side_stt_token_idx_list.append(token_idx)
            elif token == self.side_end_token :
                side_end_token_idx_list.append(token_idx)
        
        panel_stt_idx_list = []
        panel_stt_token_list = []
        for token_idx, token in enumerate(token_primitive_list) :
            if token == self.panel_start_token :
                panel_stt_idx_list.append(token_idx)
                panel_stt_token_list.append(token)
            if token == "</s>" :
                panel_stt_idx_list.append(token_idx)
                panel_stt_token_list.append(token)
        
        if panel_stt_token_list[-1] != "</s>" :
            panel_stt_idx_list.append(len(token_primitive_list))
            panel_stt_token_list.append("</s>")
        
        
        seam_data_list = []
        for side_stt_token_idx, side_end_token_idx in zip(side_stt_token_idx_list, side_end_token_idx_list) :
            side_seam_token_list = token_primitive_list[side_stt_token_idx + 1 : side_end_token_idx]
            
            if len(side_seam_token_list) % 5 != 0 :
                print("SEAM data is not multiple of 5")
            
            for stch_token_idx in range(0, len(side_seam_token_list), 5) :
                try :    
                    stch_idx = int(side_seam_token_list[stch_token_idx])
                    x1 = int(side_seam_token_list[stch_token_idx + 1].replace("<loc_", "").replace(">", "")) / 1000 * width
                    y1 = int(side_seam_token_list[stch_token_idx + 2].replace("<loc_", "").replace(">", "")) / 1000 * height
                    x2 = int(side_seam_token_list[stch_token_idx + 3].replace("<loc_", "").replace(">", "")) / 1000 * width
                    y2 = int(side_seam_token_list[stch_token_idx + 4].replace("<loc_", "").replace(">", "")) / 1000 * height
                    
                    seam_data_list.append((stch_idx, x1, y1, x2, y2))
                except Exception as e :
                    print(e)
                    print(side_seam_token_list[stch_token_idx:stch_token_idx+5])
                    raise e
                
        panel_data_list = []
        for panel_stt_idx, panel_end_idx in zip(
            panel_stt_idx_list[:-1],
            panel_stt_idx_list[1:]
        ) :
            panel_token_list = token_primitive_list[panel_stt_idx + 1 : panel_end_idx]
            
            if len(panel_token_list) % 3 != 0 :
                print("PANEL data is not multiple of 3")
            
            vertex_list = []
            edge_stch_idx_list = []
            try :
                for edge_token_idx in range(0, len(panel_token_list)-3, 3) :
                    vert_x1 = int(panel_token_list[edge_token_idx].replace("<loc_", "").replace(">", ""))
                    vert_y1 = int(panel_token_list[edge_token_idx + 1].replace("<loc_", "").replace(">", ""))
                    stch_idx = int(panel_token_list[edge_token_idx + 2])
                    vertex_list.append((vert_x1, vert_y1))
                    edge_stch_idx_list.append(stch_idx)
                
                vert_x1 = int(panel_token_list[-3].replace("<loc_", "").replace(">", ""))
                vert_y1 = int(panel_token_list[-2].replace("<loc_", "").replace(">", ""))
                scale_factor = int(panel_token_list[-1].replace("<loc_", "").replace(">", ""))
                vertex_list.append((vert_x1, vert_y1))
                    
                panel_data_list.append((vertex_list, edge_stch_idx_list, scale_factor))
            
            except Exception as e :
                print(e)
                print(panel_token_list)
                raise e
                
        return seam_data_list, panel_data_list

    def plot_parsed_data(self, img, seam_data_list, panel_data_list) :
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        seam_count = 0
        for seam_data in seam_data_list :
            stch_idx, x1, y1, x2, y2 = seam_data
            
            plt.arrow(
                x1, y1,
                x2 - x1, y2 - y1,
                head_width=10,
                head_length=10,
                color="green"
            )
            
            
            for dx, dy in [(-1, 1), (1, 1), (-1, -1), (1, -1)] :
                plt.annotate(
                    f"{stch_idx}",
                    (
                        (x1 + x2) / 2 + dx,
                        (y1 + y2) / 2 + dy
                    ),
                    fontweight="bold",
                    color="white"
                )
            plt.annotate(
                f"{stch_idx}",
                (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2
                ),
                fontsize=10,
                fontweight="bold"
            )
            
            
            # for dx, dy in [(-1, 1), (1, 1), (-1, -1), (1, -1)] :
            #     plt.annotate(
            #         f"{seam_count}",
            #         (x1 + dx, y1 + dy),
            #         fontsize=10,
            #         fontweight="bold",
            #         color="white"
            #     )
            # plt.annotate(
            #     f"{seam_count}",
            #     (x1, y1),
            #     fontsize=10,
            #     fontweight="bold",
            #     color = "green"
            # )
            # seam_count += 1
            
        plt.show()
        
        NROWs = int(np.ceil(np.sqrt(len(panel_data_list))))
        NCOLs = int(np.ceil(len(panel_data_list) / NROWs))
        FIGLEN = 4
        fig_idx = 0
        plt.figure(figsize=(NROWs*FIGLEN, NCOLs*FIGLEN))
        for panel_data in panel_data_list :
            vertex_list, edge_stch_idx_list, scale_factor = panel_data
            fig_idx += 1
            
            # print(vertex_list)
            # print(edge_stch_idx_list)
            # print(scale_factor)
            ax = plt.subplot(NROWs, NCOLs, fig_idx)
            plt.title(f"scale factor : {scale_factor}")
            edge_count = 0
            for edge_stt, edge_end, stch_idx in zip(
                vertex_list[:-1],
                vertex_list[1:],
                edge_stch_idx_list
            ) :
                ax.arrow(
                    edge_stt[0], edge_stt[1],
                    edge_end[0] - edge_stt[0], edge_end[1] - edge_stt[1],
                    head_width=10,
                    head_length=10,
                    color="red"
                )
                
                for dx, dy in [(-1, 1), (1, 1), (-1, -1), (1, -1)] :
                    ax.annotate(
                        f"{stch_idx}",
                        (
                            (edge_stt[0] + edge_end[0]) / 2 + dx,
                            (edge_stt[1] + edge_end[1]) / 2 + dy
                        ),
                        fontsize=10,
                        fontweight="bold",
                        color="white"
                    )
                ax.annotate(
                    f"{stch_idx}",
                    (
                        (edge_stt[0] + edge_end[0]) / 2,
                        (edge_stt[1] + edge_end[1]) / 2
                    ),
                    fontsize=10,
                    fontweight="bold"
                )
                
                
                for dx, dy in [(-1, 1), (1, 1), (-1, -1), (1, -1)] :
                    ax.annotate(
                        f"{edge_count}",
                        (edge_stt[0] + dx, edge_stt[1] + dy),
                        color="white",
                        fontsize=8,
                        fontweight="bold"
                    )
                
                ax.annotate(
                    f"{edge_count}",
                    (edge_stt[0], edge_stt[1]),
                    fontsize=8,
                    fontweight="bold",
                    color = "green"
                )
                edge_count += 1
            ax.invert_yaxis()
            ax.axis("equal")
