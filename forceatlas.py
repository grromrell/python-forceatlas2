import os
import sys
import time
import ctypes
import random
import subprocess
import numpy as np

# Try loading native C engine; compile if needed
_C_LIB = None

def _get_c_lib():
    global _C_LIB
    if _C_LIB is not None:
        return _C_LIB
    
    dir_path = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(dir_path, "_fa2.so")
    c_path = os.path.join(dir_path, "_fa2.c")

    if not os.path.exists(so_path) and os.path.exists(c_path):
        compiler = "clang" if subprocess.run(["which", "clang"], capture_output=True).returncode == 0 else "gcc"
        try:
            subprocess.run(
                [compiler, "-O3", "-shared", "-fPIC", "-pthread", "-ffast-math", c_path, "-o", so_path],
                check=True, capture_output=True
            )
        except Exception:
            pass

    if os.path.exists(so_path):
        try:
            lib = ctypes.CDLL(so_path)
            # Define function signatures
            lib.c_repulsion_pairwise.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int
            ]
            lib.c_repulsion_pairwise.restype = None

            lib.c_repulsion_barnes_hut.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int
            ]
            lib.c_repulsion_barnes_hut.restype = None

            lib.c_gravity.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int
            ]
            lib.c_gravity.restype = None

            lib.c_attraction.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            lib.c_attraction.restype = None

            lib.c_step.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_int
            ]
            lib.c_step.restype = None

            _C_LIB = lib
        except Exception:
            _C_LIB = None

    return _C_LIB


class ForceAtlas2:
    
    def __init__(self, graph, iterations, pos=None, sizes=None, directed=False, barnes_hut_theta=1.2, 
                 edge_weight_influence=0, gravity=0, jitter_tolerance=1, scaling_ratio=2, adjust_sizes=False, 
                 barnes_hut_optimize=False, lin_log_mode=False, outbound_attraction_distribution=False, 
                 strong_gravity_mode=False, num_threads=None):

        """
        Class to facilitate force atlas iterations. Run_algo is the main 
        function that will perform the calculations.

        Inputs
        ------
        graph : np.array or array-like
            A two-dimensional square matrix of size [num_nodes, num_nodes] 
            containing either flags for connections (1 if an edge exists, 0 
            if not), or edge weights

        iterations : int
            Number of iterations of the algorithm to perform

        pos : np.array or array-like
             A one dimensional array containing a tuple or list of initial
             x and y coordinates

        sizes : np.array or array-like
            A one dimensional array containing node sizes, required for 
            adjust_sizes mode

        directed : bool
            Whether the supplied graph is directed or undirected

        barnes_hut_theta : float (default=1.2)
            The distance modifier parameter for barnes hut optimization, 
            if barnes_hut_optimize is False then this is ignored

        edge_weight_influence : float (default=0)
            The weighting factor on edges:
                - 0: No weighting applied
                - 1: Passed weights in graph variable used
                - Other: Passed weight ** (other) used
        
        gravity : float (default=0)
            A factor that controls how much attractive force any two nodes
            have on each other. The higher the value the smaller the graph

        jitter_tolerance : float (default=1)
            How reactive the graph is to small changes, the higher the value
            the less likely small changes are to affect the graph. Larger
            graphs may need a higher jitter tolerance

        scaling_ratio : float (default=2)
            How "spread out" the graph is. The higher the number the larger
            the graph

        adjust_sizes : bool
            Whether the use anti-collision mode. If True, must also pass
            sizes. Anti-collision mode prevents node overlap based on their 
            size
        
        barnes_hut_optimize : bool
            Whether to use regional optimization. For small graphs (n<500)
            this will slow down the calculation. As the graph gets larger
            the more speed gains to capture from using this mode.

        lin_log_mode : bool
            Whether to use log based distance. When active, neighborhoods
            tend to be tighter, leaving more white space between neighborhoods

        outbound_attraction_distribution : bool
            Whether to use the dissuade hubs option. When active, nodes with
            high indegree are more central while nodes with high outdegree 
            are pushed to the periphery.

        strong_gravity_mode : bool
            Whether to use the strong gravity mode. When active, nodes will
            have a much higher gravity, creating a much more compact graph.
        """
        self.iterations = iterations
        self.directed = directed
        self.barnes_hut_theta = float(barnes_hut_theta)
        self.barnes_hut_optimize = bool(barnes_hut_optimize)
        self.edge_weight_influence = edge_weight_influence
        self.gravity = float(gravity)
        self.jitter_tolerance = float(jitter_tolerance)
        self.scaling_ratio = float(scaling_ratio)
        self.adjust_sizes = bool(adjust_sizes)
        self.lin_log_mode = bool(lin_log_mode)
        self.outbound_attraction_distribution = bool(outbound_attraction_distribution)
        self.strong_gravity_mode = bool(strong_gravity_mode)

        if num_threads is None:
            self.num_threads = min(8, max(1, os.cpu_count() or 1))
        else:
            self.num_threads = int(num_threads)

        graph_arr = np.asarray(graph, dtype=np.float64)
        self.N = len(graph_arr)

        # Positions
        if pos is not None:
            self.pos = np.asarray(pos, dtype=np.float64).copy()
        else:
            self.pos = np.random.rand(self.N, 2).astype(np.float64)

        # Node sizes
        if sizes is not None:
            self.sizes = np.asarray(sizes, dtype=np.float64).copy()
        else:
            self.sizes = None

        # Degrees & masses
        if directed:
            in_deg = np.count_nonzero(graph_arr, axis=0)
            out_deg = np.count_nonzero(graph_arr, axis=1)
            self.mass = (1 + in_deg + out_deg).astype(np.float64)
            x_cor, y_cor = graph_arr.nonzero()
        else:
            deg = np.count_nonzero(graph_arr, axis=1)
            self.mass = (1 + deg).astype(np.float64)
            upper = np.triu(graph_arr)
            x_cor, y_cor = upper.nonzero()

        self.edges_src = x_cor.astype(np.int32)
        self.edges_dst = y_cor.astype(np.int32)

        raw_weights = graph_arr[x_cor, y_cor]
        if self.edge_weight_influence == 0:
            self.edges_weight = np.ones(len(x_cor), dtype=np.float64)
        elif self.edge_weight_influence == 1:
            self.edges_weight = raw_weights.astype(np.float64)
        else:
            self.edges_weight = (raw_weights ** self.edge_weight_influence).astype(np.float64)

        self.forces = np.zeros((self.N, 2), dtype=np.float64)
        self.old_forces = np.zeros((self.N, 2), dtype=np.float64)
        self.speed = 1.0
        self.speed_efficiency = 1.0

        if self.outbound_attraction_distribution:
            self.outbound_att_compensation = float(np.mean(self.mass))
        else:
            self.outbound_att_compensation = 1.0

        # Precompute upper triangle indices for vectorized NumPy pairwise repulsion
        self._i_idx, self._j_idx = np.triu_indices(self.N, k=1)
        self._c_lib = _get_c_lib()

    @property
    def nodes(self):
        # Backward-compatible dict view of nodes
        nodes = []
        for i in range(self.N):
            nodes.append({
                'id': i,
                'x': float(self.pos[i, 0]),
                'y': float(self.pos[i, 1]),
                'dx': float(self.forces[i, 0]),
                'dy': float(self.forces[i, 1]),
                'old_dx': float(self.old_forces[i, 0]),
                'old_dy': float(self.old_forces[i, 1]),
                'mass': float(self.mass[i]),
                'size': float(self.sizes[i]) if self.sizes is not None else None
            })
        return nodes

    @property
    def edges(self):
        # Backward-compatible dict view of edges
        edges = []
        for e in range(len(self.edges_src)):
            edges.append({
                'source': int(self.edges_src[e]),
                'target': int(self.edges_dst[e]),
                'weight': float(self.edges_weight[e])
            })
        return edges

    def run_algo(self):
        for i in range(self.iterations):
            self.go_algo()
        return [(float(self.pos[i, 0]), float(self.pos[i, 1])) for i in range(self.N)]

    def go_algo(self):
        if self._c_lib is not None:
            self._go_algo_c()
        else:
            self._go_algo_numpy()

    def _go_algo_c(self):
        lib = self._c_lib
        self.old_forces[:] = self.forces
        self.forces.fill(0.0)

        pos_p = self.pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        mass_p = self.mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        sizes_p = self.sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if (self.adjust_sizes and self.sizes is not None) else None
        forces_p = self.forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        old_forces_p = self.old_forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Repulsion
        if self.barnes_hut_optimize:
            lib.c_repulsion_barnes_hut(
                pos_p, mass_p, sizes_p, forces_p,
                self.N, ctypes.c_double(self.scaling_ratio),
                ctypes.c_double(self.barnes_hut_theta),
                int(self.adjust_sizes), self.num_threads
            )
        else:
            lib.c_repulsion_pairwise(
                pos_p, mass_p, sizes_p, forces_p,
                self.N, ctypes.c_double(self.scaling_ratio),
                int(self.adjust_sizes), self.num_threads
            )

        # Gravity
        lib.c_gravity(
            pos_p, mass_p, forces_p, self.N,
            ctypes.c_double(self.gravity), ctypes.c_double(self.scaling_ratio),
            int(self.strong_gravity_mode)
        )

        # Attraction
        if len(self.edges_src) > 0:
            edges_src_p = self.edges_src.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            edges_dst_p = self.edges_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            edges_w_p = self.edges_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            lib.c_attraction(
                pos_p, mass_p, sizes_p,
                edges_src_p, edges_dst_p, edges_w_p, forces_p,
                len(self.edges_src), ctypes.c_double(self.outbound_att_compensation),
                int(self.lin_log_mode), int(self.outbound_attraction_distribution),
                int(self.adjust_sizes)
            )

        # Step
        c_speed = ctypes.c_double(self.speed)
        c_speed_eff = ctypes.c_double(self.speed_efficiency)
        lib.c_step(
            pos_p, mass_p, sizes_p, old_forces_p, forces_p,
            self.N, ctypes.byref(c_speed), ctypes.byref(c_speed_eff),
            ctypes.c_double(self.jitter_tolerance), int(self.adjust_sizes)
        )
        self.speed = c_speed.value
        self.speed_efficiency = c_speed_eff.value

    def _go_algo_numpy(self):
        self.old_forces[:] = self.forces
        self.forces.fill(0.0)

        # 1. Repulsion (pairwise vectorized)
        i = self._i_idx
        j = self._j_idx
        delta = self.pos[i] - self.pos[j]
        dist = np.hypot(delta[:, 0], delta[:, 1])

        if self.adjust_sizes and self.sizes is not None:
            dist_collision = dist - self.sizes[i] - self.sizes[j]
            factor = np.zeros_like(dist)
            pos_mask = dist_collision > 0
            neg_mask = dist_collision < 0
            factor[pos_mask] = (self.scaling_ratio * self.mass[i][pos_mask] * self.mass[j][pos_mask]) / (dist_collision[pos_mask] ** 2)
            factor[neg_mask] = 100.0 * self.scaling_ratio * self.mass[i][neg_mask] * self.mass[j][neg_mask]
        else:
            factor = np.zeros_like(dist)
            pos_mask = dist > 0
            factor[pos_mask] = (self.scaling_ratio * self.mass[i][pos_mask] * self.mass[j][pos_mask]) / (dist[pos_mask] ** 2)

        rep_f = delta * factor[:, None]
        np.add.at(self.forces, i, rep_f)
        np.add.at(self.forces, j, -rep_f)

        # 2. Gravity
        dist_g = np.hypot(self.pos[:, 0], self.pos[:, 1])
        if self.strong_gravity_mode:
            factor_g = self.scaling_ratio * self.mass * (self.gravity / self.scaling_ratio)
        else:
            factor_g = np.zeros(self.N, dtype=np.float64)
            mask_g = dist_g > 0
            factor_g[mask_g] = (self.scaling_ratio * self.mass[mask_g] * (self.gravity / self.scaling_ratio)) / dist_g[mask_g]
        self.forces -= self.pos * factor_g[:, None]

        # 3. Attraction
        if len(self.edges_src) > 0:
            src = self.edges_src
            dst = self.edges_dst
            w = self.edges_weight
            delta_att = self.pos[src] - self.pos[dst]
            dist_att = np.hypot(delta_att[:, 0], delta_att[:, 1])

            if self.adjust_sizes and self.sizes is not None:
                dist_eff = dist_att - self.sizes[src] - self.sizes[dst]
            else:
                dist_eff = dist_att

            valid = dist_eff > 0
            factor_att = np.zeros_like(dist_eff)

            if self.lin_log_mode:
                if self.outbound_attraction_distribution:
                    factor_att[valid] = -self.outbound_att_compensation * w[valid] * np.log1p(dist_eff[valid]) / (dist_eff[valid] * self.mass[src[valid]])
                else:
                    factor_att[valid] = -self.outbound_att_compensation * w[valid] * np.log1p(dist_eff[valid]) / dist_eff[valid]
            else:
                if self.outbound_attraction_distribution:
                    if self.adjust_sizes and self.sizes is not None:
                        factor_att[valid] = -self.outbound_att_compensation * w[valid] / self.mass[src[valid]]
                    else:
                        factor_att = -self.outbound_att_compensation * w / self.mass[src]
                else:
                    if self.adjust_sizes and self.sizes is not None:
                        factor_att[valid] = -self.outbound_att_compensation * w[valid]
                    else:
                        factor_att = -self.outbound_att_compensation * w

            att_f = delta_att * factor_att[:, None]
            np.add.at(self.forces, src, att_f)
            np.add.at(self.forces, dst, -att_f)

        # 4. Auto adjust speed
        diff = self.old_forces - self.forces
        sum_f = self.old_forces + self.forces
        swinging = np.hypot(diff[:, 0], diff[:, 1])
        total_swinging = float(np.sum(self.mass * swinging))
        total_effective_traction = float(0.5 * np.sum(self.mass * np.hypot(sum_f[:, 0], sum_f[:, 1])))

        if total_swinging == 0 or total_effective_traction == 0:
            return

        opt_jt = 0.05 * (self.N ** 0.5)
        min_jt = opt_jt ** 0.5
        max_jt = 10.0
        jt = self.jitter_tolerance * max(min_jt, min(max_jt, (opt_jt * total_effective_traction) / (self.N ** 2)))

        min_speed_eff = 0.05
        if (total_swinging / total_effective_traction) > 2.0:
            if self.speed_efficiency > min_speed_eff:
                self.speed_efficiency *= 0.5
            jt = max(jt, self.jitter_tolerance)

        target_speed = (jt * self.speed_efficiency * total_effective_traction) / total_swinging

        if total_swinging > (jt * total_effective_traction):
            if self.speed_efficiency > min_speed_eff:
                self.speed_efficiency *= 0.7
        elif self.speed < 1000:
            self.speed_efficiency *= 1.3

        max_rise = 0.5
        self.speed += min(target_speed - self.speed, max_rise * self.speed)

        # 5. Apply forces
        node_swinging = self.mass * swinging
        if self.adjust_sizes and self.sizes is not None:
            factor_step = (0.1 * self.speed) / (1.0 + (self.speed * node_swinging) ** 0.5)
            df = np.hypot(self.forces[:, 0], self.forces[:, 1])
            mask_df = df > 0
            factor_step[mask_df] = np.minimum(factor_step[mask_df] * df[mask_df], 10.0) / df[mask_df]
            factor_step[~mask_df] = 0.0
        else:
            factor_step = self.speed / (1.0 + (self.speed * node_swinging) ** 0.5)

        self.pos += self.forces * factor_step[:, None]