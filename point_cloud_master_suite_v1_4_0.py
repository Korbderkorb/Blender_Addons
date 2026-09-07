bl_info = {
    "name": "Point Cloud Master Suite",
    "author": "Korbinian Enzinger",
    "version": (1, 3, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > PC Tools",
    "description": "Comprehensive toolkit for generating point clouds from video, clustering, color filtering, thin part detection, and boolean cutting against a closed mesh.",
    "warning": "",
    "category": "Mesh",
}

import os
import shutil
import tempfile
import time

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Color
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree

# --- 1. CORE LOGIC ---

def check_color_match(point_rgb, ref_hsv, tolerance, material_mode=False):
    c = Color((point_rgb[0], point_rgb[1], point_rgb[2]))
    h, s, v = c.hsv
    ref_h, ref_s, ref_v = ref_hsv
    h_diff = abs(h - ref_h)
    if h_diff > 0.5: h_diff = 1.0 - h_diff
    if h_diff > tolerance: return False
    if abs(s - ref_s) > tolerance: return False
    if not material_mode and abs(v - ref_v) > tolerance: return False
    return True

def run_dist_clustering(obj, radius, min_points):
    original_mode = obj.mode
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = obj.data
    num_verts = len(mesh.vertices)
    attr = mesh.attributes.get("cluster_id_dist") or mesh.attributes.new(name="cluster_id_dist", type='INT', domain='POINT')
    for val in attr.data: val.value = -1
    kd = KDTree(num_verts)
    for i, v in enumerate(mesh.vertices): kd.insert(v.co, i)
    kd.balance()
    visited = [False] * num_verts
    cluster_count = 0
    for i in range(num_verts):
        if visited[i]: continue
        q, cluster = [i], []
        visited[i] = True
        while q:
            idx = q.pop(0); cluster.append(idx)
            for _, n_idx, _ in kd.find_range(mesh.vertices[idx].co, radius):
                if not visited[n_idx]:
                    visited[n_idx] = True; q.append(n_idx)
        if len(cluster) >= min_points:
            for idx in cluster: attr.data[idx].value = cluster_count
            cluster_count += 1
    mesh.update()
    bpy.ops.object.mode_set(mode=original_mode)
    return cluster_count

def run_global_color_clustering(obj, tolerance, material_mode, attr_name, ref_color):
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = obj.data
    color_attr = mesh.attributes.get(attr_name)
    if not color_attr: return -1
    attr_out = mesh.attributes.get("cluster_id_color") or mesh.attributes.new(name="cluster_id_color", type='INT', domain='POINT')
    for val in attr_out.data: val.value = -1
    ref_hsv = Color((ref_color[0], ref_color[1], ref_color[2])).hsv
    count = 0
    for i, item in enumerate(color_attr.data):
        if check_color_match(item.color[:3], ref_hsv, tolerance, material_mode):
            attr_out.data[i].value = 0
            count += 1
    mesh.update()
    return count

# --- 1b. BOOLEAN CORE LOGIC ---
#
# Classifies every point of a point cloud as inside or outside a closed mesh.
#
# The inside test is a crossing-parity ray cast: a ray fired from the point
# crosses a closed surface an odd number of times if and only if the point
# started inside it. A single ray is fragile, because one that clips an edge or
# a vertex, or slips through a small hole in the container, miscounts by one and
# flips the answer. That shows up as isolated stray points surviving far from
# the container. Several directions therefore vote on each point.

# Deliberately not axis aligned, and mutually well spread, so a face that is
# degenerate with respect to one direction is not degenerate for the others.
BOOL_RAY_DIRS = [Vector(v).normalized() for v in (
    (0.5773502692, 0.4082482905, 0.7071067812),
    (-0.8014, 0.2673, 0.5345),
    (0.3015, -0.9045, 0.3015),
    (0.2182, 0.4364, -0.8729),
    (-0.4082, -0.5774, -0.7071),
)]

# Values per element, the RNA property holding them, and the numpy dtype that
# foreach_get / foreach_set expect. The dtype has to match the underlying RNA
# type or Blender raises "couldn't access the py sequence".
BOOL_ATTR_LAYOUT = {
    'FLOAT':        (1, 'value', np.float64),
    'INT':          (1, 'value', np.int32),
    'INT8':         (1, 'value', np.int8),
    'BOOLEAN':      (1, 'value', np.bool_),
    'FLOAT2':       (2, 'vector', np.float64),
    'FLOAT_VECTOR': (3, 'vector', np.float64),
    'FLOAT_COLOR':  (4, 'color', np.float64),
    'BYTE_COLOR':   (4, 'color', np.float64),
    'QUATERNION':   (4, 'value', np.float64),
    'INT32_2D':     (2, 'value', np.int32),
    'FLOAT4X4':     (16, 'value', np.float64),
}


def build_container_bvh(obj, depsgraph):
    """Build a BVH tree of the container object in WORLD space.

    BVHTree.FromObject() returns a tree in object-local space, which is an easy
    thing to get wrong. Building from a bmesh that has already been transformed
    keeps every coordinate in one space.

    Returns (bvh, bbox_min, bbox_max, open_edge_count).
    """
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()

    try:
        if len(eval_mesh.polygons) == 0:
            raise RuntimeError(f"'{obj.name}' has no faces after evaluation")
        bm = bmesh.new()
        bm.from_mesh(eval_mesh)
    finally:
        eval_obj.to_mesh_clear()

    bm.transform(obj.matrix_world)

    coords = np.empty((len(bm.verts), 3), dtype=np.float64)
    for i, v in enumerate(bm.verts):
        coords[i] = v.co

    open_edges = sum(1 for e in bm.edges if len(e.link_faces) != 2)

    bvh = BVHTree.FromBMesh(bm)
    bm.free()

    return bvh, coords.min(axis=0), coords.max(axis=0), open_edges


def cast_parity(bvh, point, direction, eps, max_hits=1000):
    """Count surface crossings along one direction.

    An odd number of crossings means the point is inside. Returns None if the
    ray never escaped, in which case the caller should ignore this vote.
    """
    hits = 0
    origin = point.copy()

    while hits <= max_hits:
        location = bvh.ray_cast(origin, direction)[0]
        if location is None:
            return (hits % 2) == 1
        hits += 1
        origin = location + direction * eps

    return None


def point_is_inside(bvh, point, eps, samples):
    """Majority vote of `samples` parity tests.

    Returns (inside, unanimous). Voting stops as soon as one side has an
    unbeatable lead, so an unambiguous point costs samples // 2 + 1 rays.
    """
    needed = samples // 2 + 1
    yes = no = 0

    for i in range(samples):
        vote = cast_parity(bvh, point, BOOL_RAY_DIRS[i], eps)
        if vote is None:
            continue
        if vote:
            yes += 1
        else:
            no += 1
        if yes >= needed or no >= needed:
            break

    return yes > no, not (yes and no)


def filter_points_only_mesh(mesh, keep_mask):
    """Fast path for a mesh with vertices but no edges or faces.

    Rebuilds the vertex array with numpy and carries every point-domain
    attribute (vertex colours, intensity, cluster ids, ...) across.
    """
    keep_idx = np.nonzero(keep_mask)[0]
    new_count = len(keep_idx)
    dropped = []

    # Read every point attribute before the geometry is cleared.
    saved = []
    for attr in mesh.attributes:
        if attr.domain != 'POINT' or attr.name == 'position':
            continue
        if attr.name.startswith('.'):
            continue  # internal, Blender recreates these
        layout = BOOL_ATTR_LAYOUT.get(attr.data_type)
        if layout is None:
            dropped.append(attr.name)
            continue
        size, prop, dtype = layout
        buf = np.empty(len(attr.data) * size, dtype=dtype)
        attr.data.foreach_get(prop, buf)
        saved.append((attr.name, attr.data_type, prop,
                      buf.reshape(-1, size)[keep_idx]))

    positions = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get('co', positions)
    positions = positions.reshape(-1, 3)[keep_idx]

    active_color = mesh.color_attributes.active_color_name
    render_color = mesh.color_attributes.default_color_name

    mesh.clear_geometry()
    mesh.vertices.add(new_count)
    mesh.vertices.foreach_set('co', positions.ravel())

    for name, data_type, prop, buf in saved:
        attr = mesh.attributes.get(name)
        if attr is None:
            attr = mesh.attributes.new(name=name, type=data_type,
                                       domain='POINT')
        attr.data.foreach_set(prop, buf.ravel())

    if active_color and active_color in mesh.color_attributes:
        mesh.color_attributes.active_color_name = active_color
    if render_color and render_color in mesh.color_attributes:
        mesh.color_attributes.default_color_name = render_color

    mesh.update()
    return dropped


def filter_mesh_with_topology(mesh, keep_mask):
    """General path: delete the unwanted vertices with bmesh.

    Slower and heavier on memory than the points-only path, but it takes care
    of the edges and faces that reference the removed vertices.
    """
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    doomed = [v for i, v in enumerate(bm.verts) if not keep_mask[i]]
    bmesh.ops.delete(bm, geom=doomed, context='VERTS')

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return []


def tag_redraw_sidebar(context):
    """Repaint the N-panel so the progress line updates while work runs."""
    screen = getattr(context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'UI':
                    region.tag_redraw()


class BooleanJob:
    """Chunked inside/outside classification of a point cloud.

    Kept separate from the operators so one implementation drives both the
    modal path, which reports live progress in the panel, and the blocking
    path used when there is no window, such as a script in background mode.

    Raises ValueError with a user facing message if the setup is not usable.
    """

    def __init__(self, context, operation):
        scene = context.scene
        self.operation = operation
        self.container = scene.pc_bool_container
        self.cloud = scene.pc_bool_cloud
        self.dropped = []

        if self.container is None:
            raise ValueError("Pick a closed mesh in the Boolean Logic panel")
        if self.cloud is None:
            raise ValueError("Pick a point cloud in the Boolean Logic panel")
        if self.cloud == self.container:
            raise ValueError("The closed mesh and the point cloud must be "
                             "two different objects")
        if self.cloud.type != 'MESH':
            raise ValueError(f"'{self.cloud.name}' is not a mesh")
        if self.container.type != 'MESH' or len(self.container.data.polygons) == 0:
            raise ValueError(f"'{self.container.name}' has no faces, so it "
                             f"cannot act as a closed container")

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        self.mesh = self.cloud.data
        self.total = len(self.mesh.vertices)
        if self.total == 0:
            raise ValueError(f"'{self.cloud.name}' has no vertices")

        depsgraph = context.evaluated_depsgraph_get()
        try:
            self.bvh, bbox_min, bbox_max, self.open_edges = \
                build_container_bvh(self.container, depsgraph)
        except RuntimeError as exc:
            raise ValueError(str(exc))

        # Point cloud vertices, in world space, in one numpy pass.
        coords = np.empty(self.total * 3, dtype=np.float64)
        self.mesh.vertices.foreach_get('co', coords)
        coords = coords.reshape(-1, 3)
        matrix = np.array(self.cloud.matrix_world, dtype=np.float64)
        self.world = coords @ matrix[:3, :3].T + matrix[:3, 3]

        diagonal = float(np.linalg.norm(bbox_max - bbox_min))
        self.eps = max(diagonal * 1e-6, 1e-7)

        # Anything outside the container bounding box is outside the
        # container, so it never needs a BVH query at all.
        in_box = np.all((self.world >= bbox_min - self.eps) &
                        (self.world <= bbox_max + self.eps), axis=1)
        self.candidates = np.nonzero(in_box)[0]

        self.samples = int(scene.pc_bool_samples)
        self.mask = np.zeros(self.total, dtype=bool)
        self.cursor = 0
        self.contested = 0
        self.rate = 0.0
        self.start = time.time()

    @property
    def progress(self):
        if len(self.candidates) == 0:
            return 1.0
        return self.cursor / len(self.candidates)

    def step(self, budget=0.08):
        """Classify for up to `budget` seconds. Returns True when finished."""
        cands = self.candidates
        n = len(cands)
        deadline = time.time() + budget

        while self.cursor < n:
            end = min(self.cursor + 2000, n)
            for idx in cands[self.cursor:end]:
                inside, unanimous = point_is_inside(
                    self.bvh, Vector(self.world[idx]), self.eps, self.samples)
                if not unanimous:
                    self.contested += 1
                if inside:
                    self.mask[idx] = True
            self.cursor = end
            if time.time() >= deadline:
                break

        elapsed = time.time() - self.start
        self.rate = self.cursor / elapsed if elapsed > 0 else 0.0
        return self.cursor >= n

    def finish(self):
        """Delete the unwanted points. Returns (removed, kept, message)."""
        keep_mask = self.mask if self.operation == 'INTERSECT' else ~self.mask
        kept = int(keep_mask.sum())
        removed = self.total - kept

        if kept == 0:
            raise ValueError("Every point would be removed, so the point "
                             "cloud was left untouched. Check that the "
                             "objects overlap and that the right button "
                             "was used")

        if removed:
            if len(self.mesh.edges) > 0 or len(self.mesh.polygons) > 0:
                self.dropped = filter_mesh_with_topology(self.mesh, keep_mask)
            else:
                self.dropped = filter_points_only_mesh(self.mesh, keep_mask)

        return removed, kept, f"Removed {removed:,} of {self.total:,} points"


# --- 1b. THICKNESS LOGIC ---

THIN_ATTR = "pc_thickness"
THIN_HEAT_ATTR = "pc_thin_heat"


def estimate_point_spacing(coords, sample=40000, probes=4000):
    """Median nearest neighbour distance of the full cloud, in world units.

    Measured on a random subsample so the cost does not follow the point
    count, then scaled back up: in a volumetric distribution the spacing
    goes with the cube root of the density, so a subsample of m points out
    of n reads a spacing that is (n / m) ** (1 / 3) too large.
    """
    n = len(coords)
    if n < 2:
        return 0.0
    m = min(sample, n)
    rng = np.random.default_rng(0)
    picks = rng.choice(n, m, replace=False) if m < n else np.arange(n)
    sub = coords[picks]

    kd = KDTree(m)
    for i in range(m):
        kd.insert(Vector(sub[i]), i)
    kd.balance()

    found = []
    for i in range(min(probes, m)):
        hits = kd.find_n(Vector(sub[i]), 2)
        if len(hits) > 1:
            found.append(hits[1][2])
    if not found:
        return 0.0
    return float(np.median(found)) * (m / n) ** (1.0 / 3.0)


def voxel_neighbour_offsets(dim_y, dim_z):
    """Key space steps to the 6 face and the 6 face plus 12 edge neighbours.

    Voxels are addressed by the single packed key (i * dim_y + j) * dim_z + k,
    so a step of one voxel along an axis is a fixed integer added to that key.
    """
    step_x = dim_y * dim_z
    step_y = dim_z
    step_z = 1
    face = [step_x, -step_x, step_y, -step_y, step_z, -step_z]
    edge = []
    for a, b in ((step_x, step_y), (step_x, step_z), (step_y, step_z)):
        edge += [a + b, a - b, -a + b, -a - b]
    return face, face + edge


def voxel_survivors(keys, offsets):
    """Positions in sorted `keys` whose every offset neighbour is also in it.

    Candidates are narrowed one offset at a time, so the later and more
    numerous offsets only get tested against the few voxels that are still
    in the running.
    """
    if keys.size == 0:
        return np.empty(0, dtype=np.int64)
    sel = np.arange(keys.size, dtype=np.int64)
    last = keys.size - 1
    for off in offsets:
        if sel.size == 0:
            break
        probe = keys[sel] + off
        idx = np.searchsorted(keys, probe)
        np.clip(idx, 0, last, out=idx)
        sel = sel[keys[idx] == probe]
    return sel


def voxel_dilate(keys, offsets, limit=None, group=6):
    """Every voxel of `keys` plus every offset neighbour of one, sorted.

    Offsets are folded in a few at a time rather than all at once, which
    keeps the largest temporary array to about seven times the input even
    for the 18 neighbour set. `limit`, when given, is a sorted voxel set the
    result is cut back to.
    """
    out = keys
    for start in range(0, len(offsets), group):
        batch = offsets[start:start + group]
        out = np.unique(np.concatenate([out] + [keys + off for off in batch]))
    if limit is not None and out.size:
        idx = np.searchsorted(limit, out)
        np.clip(idx, 0, limit.size - 1, out=idx)
        out = out[limit[idx] == out]
    return out


class ThicknessJob:
    """Per point wall thickness of a volumetric point cloud.

    Runs on a sparse voxel set rather than on the points themselves, so the
    cost follows the number of occupied voxels, a few million, and not the
    number of points, which can be tens of millions.

    A point counts as r thick where the largest ball that both covers it and
    fits entirely inside the cloud has radius r. That is a morphological
    opening, and it is measured in two phases: erosion shrinks the cloud
    until only the voxels that can still centre a ball of r voxels survive,
    then those survivors are grown back by r to recover everything their
    balls covered. Growing back is what separates a thin part from the skin
    of a thick one. Distance to the nearest empty voxel alone would call
    both thin, because the outer layer of a solid block is just as close to
    open air as a whole thin sheet is.

    Because a ball has no preferred direction, sheets and struts are found
    by the same test and no set of candidate directions is ever searched.
    Passes alternate a face neighbourhood with a face plus edge one: either
    alone grows a diamond or a cube and would read a diagonal part as up to
    70% thinner or thicker than the same part axis aligned, while
    alternating them approximates a sphere and brings that down to about
    13%.

    Raises ValueError with a user facing message if the setup is not usable.
    """

    CHUNK = 2_000_000

    def __init__(self, context):
        scene = context.scene
        obj = context.active_object

        if obj is None or obj.type != 'MESH':
            raise ValueError("Make a point cloud mesh the active object first")
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        self.obj = obj
        self.mesh = obj.data
        self.total = len(self.mesh.vertices)
        if self.total == 0:
            raise ValueError(f"'{obj.name}' has no vertices")

        self.voxel = float(scene.pc_thin_voxel)
        self.max_thickness = float(scene.pc_thin_max)
        self.threshold = float(scene.pc_thin_threshold)
        self.close_gaps = bool(scene.pc_thin_close)
        if self.max_thickness <= 0.0:
            raise ValueError("Measure Up To must be larger than zero")

        self.stage = 0
        self.cursor = 0
        self.pass_index = 0
        self.passes = 0
        self.voxel_count = 0
        self.thin_count = 0
        self.auto_voxel = self.voxel <= 0.0
        self.start = time.time()

        self.open_q = 0
        self.open_total = 0

        self.stages = (
            ("Reading points", self._read),
            ("Estimating spacing", self._spacing),
            ("Voxelising", self._voxelise),
            ("Sorting voxels", self._sort),
            ("Filling gaps", self._close),
            ("Shrinking", self._erode),
            ("Growing back", self._open),
            ("Writing attribute", self._write),
        )
        self.weights = (0.09, 0.03, 0.17, 0.22, 0.07, 0.15, 0.19, 0.08)

    # -- progress reporting -------------------------------------------------

    @property
    def stage_name(self):
        if self.stage >= len(self.stages):
            return "Done"
        return self.stages[self.stage][0]

    @property
    def progress(self):
        done = sum(self.weights[:self.stage])
        if self.stage < len(self.stages):
            done += self.weights[self.stage] * self._stage_fraction()
        return min(done, 1.0)

    def _stage_fraction(self):
        handler = self.stages[self.stage][1]
        if handler is self._voxelise and self.total:
            return self.cursor / self.total
        if handler is self._erode and self.passes:
            return self.pass_index / self.passes
        if handler is self._open and self.open_total:
            return (self.open_total - self.open_q) / self.open_total
        return 0.0

    def step(self, budget=0.08):
        """Work for up to `budget` seconds. Returns True when finished."""
        if self.stage >= len(self.stages):
            return True
        if self.stages[self.stage][1](budget):
            self.stage += 1
        return self.stage >= len(self.stages)

    # -- stages -------------------------------------------------------------

    def _read(self, budget):
        coords = np.empty(self.total * 3, dtype=np.float32)
        self.mesh.vertices.foreach_get('co', coords)
        coords = coords.reshape(-1, 3)

        matrix = np.array(self.obj.matrix_world, dtype=np.float32)
        if not np.allclose(matrix, np.identity(4), atol=1e-6):
            # In place and chunked, so a 30M point cloud does not need a
            # second full sized array just to be moved into world space.
            rot = np.ascontiguousarray(matrix[:3, :3].T)
            offset = matrix[:3, 3]
            for start in range(0, self.total, self.CHUNK):
                end = min(start + self.CHUNK, self.total)
                coords[start:end] = coords[start:end] @ rot + offset
        self.coords = coords
        return True

    def _spacing(self, budget):
        if self.auto_voxel:
            spacing = estimate_point_spacing(self.coords)
            if spacing <= 0.0:
                raise ValueError("Could not estimate the point spacing, set "
                                 "Voxel Size by hand")
            # Around eight points per voxel. Sparser than that and the gaps
            # between the points read as holes, which erosion eats into.
            self.voxel = spacing * 3.5
        self.passes = max(1, int(np.ceil(self.max_thickness / (2.0 * self.voxel))))
        return True

    def _voxelise(self, budget):
        coords = self.coords
        inv = 1.0 / self.voxel

        if self.cursor == 0:
            self.origin = coords.min(axis=0)
            span = np.floor((coords.max(axis=0) - self.origin) * inv).astype(np.int64) + 1
            # A margin keeps every occupied voxel away from the edge of the
            # grid, so stepping to a neighbour can never wrap onto the far
            # side of it and be mistaken for one.
            self.margin = self.passes + 2
            self.dims = span + 2 * self.margin
            if float(self.dims[0]) * float(self.dims[1]) * float(self.dims[2]) > 4.0e18:
                raise ValueError("Voxel Size is too small for a cloud this "
                                 "large, raise it and run again")
            self.keys = np.empty(self.total, dtype=np.int64)

        dim_y, dim_z = int(self.dims[1]), int(self.dims[2])
        deadline = time.time() + budget
        while self.cursor < self.total:
            end = min(self.cursor + self.CHUNK, self.total)
            block = coords[self.cursor:end]
            i = ((block[:, 0] - self.origin[0]) * inv).astype(np.int64) + self.margin
            j = ((block[:, 1] - self.origin[1]) * inv).astype(np.int64) + self.margin
            k = ((block[:, 2] - self.origin[2]) * inv).astype(np.int64) + self.margin
            self.keys[self.cursor:end] = (i * dim_y + j) * dim_z + k
            self.cursor = end
            if time.time() >= deadline:
                break

        if self.cursor < self.total:
            return False
        self.coords = None
        return True

    def _sort(self, budget):
        uniq, inverse = np.unique(self.keys, return_inverse=True)
        self.keys = None
        self.uniq = uniq
        self.inverse = inverse.ravel().astype(np.int32, copy=False)
        self.voxel_count = int(uniq.size)
        self.face, self.ball = voxel_neighbour_offsets(int(self.dims[1]),
                                                       int(self.dims[2]))
        return True

    def _close(self, budget):
        if self.close_gaps:
            # Grow by one voxel then shrink back. Thickness is preserved but
            # single voxel holes inside the bulk, left by uneven sampling,
            # are plugged before they can be mistaken for a surface.
            grown = voxel_dilate(self.uniq, self.face)
            self.grid = grown[voxel_survivors(grown, self.face)]
            if self.grid.size < self.uniq.size:
                self.grid = np.union1d(self.grid, self.uniq)
        else:
            self.grid = self.uniq
        self.work = self.grid
        self.alive = np.arange(self.grid.size, dtype=np.int64)
        self.depth = np.zeros(self.grid.size, dtype=np.int32)
        return True

    def _erode(self, budget):
        if self.pass_index < self.passes and self.work.size:
            offsets = self.face if self.pass_index % 2 == 0 else self.ball
            sel = voxel_survivors(self.work, offsets)
            self.work = self.work[sel]
            self.alive = self.alive[sel]
            self.depth[self.alive] += 1
            self.pass_index += 1
        if self.pass_index < self.passes and self.work.size:
            return False
        self.pass_index = self.passes
        self.thick = np.zeros(self.grid.size, dtype=np.int32)
        self.open_q = int(self.depth.max()) if self.depth.size else 0
        self.open_total = self.open_q
        return True

    def _open(self, budget):
        """Grow one radius of erosion survivors back over what its balls cover.

        Radii are taken largest first, so a voxel reached by a big ball keeps
        that reading and a later, smaller ball cannot talk it back down.
        """
        if self.open_q <= 0:
            return True
        radius = self.open_q
        # Only voxels of exactly this depth need to be grown. A deeper one
        # was already grown by a larger radius, over a superset of what it
        # would cover here, and the larger reading is the one that wins.
        keys = self.grid[self.depth == radius]
        if keys.size:
            for j in range(radius):
                offsets = self.face if j % 2 == 0 else self.ball
                keys = voxel_dilate(keys, offsets, limit=self.grid)
            covered = np.searchsorted(self.grid, keys)
            self.thick[covered] = np.maximum(self.thick[covered], radius)
        self.open_q -= 1
        return self.open_q <= 0

    def _write(self, budget):
        radius = self.thick[np.searchsorted(self.grid, self.uniq)]
        # A ball of r voxels radius spans 2r + 1 voxels, and a voxel no ball
        # at all fits in still holds its own voxel of material.
        per_voxel = (2.0 * radius.astype(np.float32) + 1.0) * np.float32(self.voxel)
        values = per_voxel[self.inverse]

        # Fetched fresh rather than reused from the start of the job: the
        # mesh can have been rebuilt since, which would leave a stale pointer.
        mesh = self.obj.data
        if len(mesh.vertices) != self.total:
            raise ValueError("The point cloud changed while it was being "
                             "measured, nothing was written")
        attr = mesh.attributes.get(THIN_ATTR)
        if attr is not None and (attr.domain != 'POINT' or attr.data_type != 'FLOAT'):
            mesh.attributes.remove(attr)
            attr = None
        if attr is None:
            attr = mesh.attributes.new(name=THIN_ATTR, type='FLOAT',
                                       domain='POINT')
        attr.data.foreach_set("value", values)
        mesh.update()

        self.thin_count = int(np.count_nonzero(values <= self.threshold))
        return True

    def finish(self):
        """Returns the one line result shown in the panel."""
        elapsed = time.time() - self.start
        return (f"{self.thin_count:,} of {self.total:,} points under "
                f"{self.threshold:g}  ({self.voxel_count:,} voxels at "
                f"{self.voxel:.4g}, {elapsed:.1f}s)")


class MESH_OT_Thickness(bpy.types.Operator):
    bl_idname = "mesh.pc_thickness"
    bl_label = "Analyze"
    bl_description = (
        "Step 2: Measures how thick the cloud is around every point and "
        "stores it in metres in the 'pc_thickness' attribute.\n\n"
        "Works by fitting a ball inside the cloud at each point, so it finds "
        "thin sheets and thin struts in any orientation, and unlike "
        "clustering it still finds them when they are joined to the rest of "
        "the cloud. Run this once, then move the Thin Below slider freely: "
        "selecting and the heatmap reuse this result and are instant. "
        "Press Esc while it runs to cancel"
    )
    bl_options = {'REGISTER'}

    _timer = None
    _job = None

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and not context.scene.pc_thin_running)

    def _cleanup(self, context):
        scene = context.scene
        scene.pc_thin_running = False
        scene.pc_thin_progress = 0.0
        scene.pc_thin_stage = ""
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._job = None

    def execute(self, context):
        """Blocking path: used from scripts and in background mode."""
        scene = context.scene
        try:
            job = ThicknessJob(context)
            while not job.step(1.0):
                pass
            message = job.finish()
        except ValueError as exc:
            scene.pc_thin_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        if job.auto_voxel:
            scene.pc_thin_voxel = job.voxel
        scene.pc_thin_status = message
        self.report({'INFO'}, message)
        return {'FINISHED'}

    def invoke(self, context, event):
        """Modal path: used when the button is clicked in the sidebar."""
        if getattr(context, "window", None) is None:
            return self.execute(context)

        scene = context.scene
        try:
            self._job = ThicknessJob(context)
        except ValueError as exc:
            scene.pc_thin_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.pc_thin_running = True
        scene.pc_thin_progress = 0.0
        scene.pc_thin_stage = self._job.stage_name
        scene.pc_thin_status = ""
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        if event.type == 'ESC':
            self._cleanup(context)
            scene.pc_thin_status = "Cancelled"
            tag_redraw_sidebar(context)
            self.report({'WARNING'}, "Thickness cancelled, nothing was written")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        try:
            done = self._job.step(0.08)
        except ValueError as exc:
            self._cleanup(context)
            scene.pc_thin_status = ""
            tag_redraw_sidebar(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.pc_thin_progress = self._job.progress
        scene.pc_thin_stage = self._job.stage_name
        tag_redraw_sidebar(context)

        if not done:
            return {'RUNNING_MODAL'}

        job = self._job
        message = job.finish()
        self._cleanup(context)
        if job.auto_voxel:
            scene.pc_thin_voxel = job.voxel
        scene.pc_thin_status = message
        tag_redraw_sidebar(context)
        self.report({'INFO'}, message)
        return {'FINISHED'}


def show_color_attribute(context, mesh, name):
    """Make `name` the attribute the viewport paints with, and show it.

    Writing a color attribute is not enough to see anything: it has to be the
    active color layer, and solid shading has to be told to paint attributes
    instead of the object color. Returns False if the viewport could not be
    switched over, so the caller can say what to do by hand.
    """
    try:
        for index, layer in enumerate(mesh.color_attributes):
            if layer.name == name:
                mesh.color_attributes.active_color_index = index
                break
        else:
            return False
    except (AttributeError, TypeError):
        return False

    switched = False
    screen = getattr(context, "screen", None)
    for area in getattr(screen, "areas", ()):
        if area.type != 'VIEW_3D':
            continue
        for space in area.spaces:
            if space.type != 'VIEW_3D':
                continue
            try:
                if space.shading.type not in ('SOLID', 'MATERIAL', 'RENDERED'):
                    space.shading.type = 'SOLID'
                if space.shading.type == 'SOLID':
                    space.shading.color_type = 'VERTEX'
                switched = True
            except (AttributeError, TypeError):
                pass
        area.tag_redraw()
    return switched


def read_thickness(context):
    """Mesh and per point thickness of the active cloud, in object mode.

    Raises ValueError with a user facing message if there is nothing to read.
    """
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        raise ValueError("Make a point cloud mesh the active object first")

    # Leave edit mode before touching the attribute, never after. Coming out
    # of edit mode rebuilds the mesh from the edit cage and frees whatever
    # the attribute was pointing at, so an attribute picked up first is a
    # dangling pointer by the time it is read, and reading it kills Blender.
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    mesh = obj.data
    attr = mesh.attributes.get(THIN_ATTR)
    if attr is None:
        raise ValueError("No thickness found, press Analyze first")
    if attr.domain != 'POINT' or attr.data_type != 'FLOAT':
        raise ValueError(f"'{THIN_ATTR}' is not a per point number any more, "
                         f"press Analyze again")
    if len(attr.data) != len(mesh.vertices):
        raise ValueError("The thickness no longer matches the point count, "
                         "press Analyze again")

    values = np.empty(len(mesh.vertices), dtype=np.float32)
    attr.data.foreach_get("value", values)
    return mesh, values


class MESH_OT_SelectThin(bpy.types.Operator):
    bl_idname = "mesh.pc_select_thin"
    bl_label = "Select Thin"
    bl_description = (
        "Step 3: Selects every point where the cloud is thinner than the "
        "Thin Below value, using the thickness measured by Analyze. Move the "
        "slider and press again to try a different cut, it does not measure "
        "again"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            mesh, values = read_thickness(context)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        mask = values <= context.scene.pc_thin_threshold
        count = int(np.count_nonzero(mask))

        # Selection is written straight onto the mesh rather than through
        # bmesh: a python loop over the vertices of a 30M point cloud, which
        # is what the other Select buttons do, takes minutes at this size.
        if len(mesh.polygons):
            mesh.polygons.foreach_set("select", np.zeros(len(mesh.polygons), dtype=bool))
        if len(mesh.edges):
            mesh.edges.foreach_set("select", np.zeros(len(mesh.edges), dtype=bool))
        mesh.vertices.foreach_set("select", mask)
        mesh.update()

        context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, f"Selected {count:,} of {len(values):,} points")
        return {'FINISHED'}


class MESH_OT_ThinHeatmap(bpy.types.Operator):
    bl_idname = "mesh.pc_thin_heatmap"
    bl_label = "Heatmap"
    bl_description = (
        "Creates a color attribute 'pc_thin_heat' shading the cloud by "
        "thickness, red where it is thin and blue where it is thick, scaled "
        "over the Measure Up To range (requires attribute display)"
    )

    def execute(self, context):
        try:
            mesh, values = read_thickness(context)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        span = max(float(context.scene.pc_thin_max), 1e-9)
        ramp = np.clip(values / span, 0.0, 1.0)
        colors = np.empty((len(values), 4), dtype=np.float32)
        colors[:, 0] = 1.0 - ramp
        colors[:, 1] = 0.0
        colors[:, 2] = ramp
        colors[:, 3] = 1.0

        attr = mesh.attributes.get(THIN_HEAT_ATTR)
        if attr is not None and (attr.domain != 'POINT' or attr.data_type != 'FLOAT_COLOR'):
            mesh.attributes.remove(attr)
            attr = None
        if attr is None:
            attr = mesh.attributes.new(name=THIN_HEAT_ATTR, type='FLOAT_COLOR',
                                       domain='POINT')
        attr.data.foreach_set("color", colors.ravel())
        mesh.update()
        shown = show_color_attribute(context, mesh, THIN_HEAT_ATTR)

        thin = int(np.count_nonzero(values <= context.scene.pc_thin_threshold))
        note = (f"Heatmap '{THIN_HEAT_ATTR}': red under {span:g}, blue at it. "
                f"{thin:,} points are under the Thin Below cut")
        if not shown:
            note += (". Set the viewport to Solid shading with Color set to "
                     "Attribute to see it")
        self.report({'INFO'}, note)
        return {'FINISHED'}


# --- 1c. LATTICE LOGIC ---

GRID_RESIDUAL_ATTR = "pc_lattice_residual"


def sample_patches(coords, spacing, patches=8, cap=2500, seed=0):
    """Small dense neighbourhoods, not a thin global subsample.

    Lattice steps can only be read off points that are actually adjacent. A
    random subsample of a huge cloud has its members far apart, so its
    nearest neighbour vectors say nothing about the lattice. Patches keep the
    local density intact, and taking several spread over the cloud shows up
    an orientation that is not the same everywhere.
    """
    n = len(coords)
    half = spacing * 6.0
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(patches * 2):
        if len(out) >= patches:
            break
        centre = coords[rng.integers(n)]
        keep = coords[:, 0] >= centre[0] - half
        keep &= coords[:, 0] <= centre[0] + half
        keep &= coords[:, 1] >= centre[1] - half
        keep &= coords[:, 1] <= centre[1] + half
        keep &= coords[:, 2] >= centre[2] - half
        keep &= coords[:, 2] <= centre[2] + half
        idx = np.nonzero(keep)[0]
        if idx.size < 40:
            continue
        if idx.size > cap:
            idx = rng.choice(idx, cap, replace=False)
        out.append(coords[idx])
    return out


def neighbour_displacements(patch, neighbours=14):
    """Vectors from each point of a patch to its nearest neighbours."""
    count = len(patch)
    if count < 8:
        return np.empty((0, 3), dtype=np.float64)
    kd = KDTree(count)
    for i in range(count):
        kd.insert(Vector(patch[i]), i)
    kd.balance()

    take = min(neighbours + 1, count)
    out = np.empty((count * (take - 1), 3), dtype=np.float64)
    written = 0
    for i in range(count):
        origin = patch[i]
        for _, j, _ in kd.find_n(Vector(origin), take)[1:]:
            out[written] = patch[j] - origin
            written += 1
    return out[:written]


def canonical_directions(vectors):
    """Fold each vector and its opposite onto the same half space."""
    probe = np.array([0.62, 0.47, 0.63], dtype=np.float64)
    side = vectors @ probe
    # Vectors square on to the probe get sorted by their largest component
    # instead, so the fold stays stable rather than flipping on noise.
    unsure = np.abs(side) < 1e-9
    if np.any(unsure):
        side = np.where(unsure, vectors[:, 0] + 1e-3 * vectors[:, 1], side)
    return np.where(side[:, None] < 0.0, -vectors, vectors)


def cluster_lattice_directions(vectors, length_tol=0.18, cos_min=0.97,
                               min_share=0.005):
    """Group displacement vectors into candidate lattice steps.

    Shortest first, absorbing everything of about the same length pointing
    about the same way. In a lattice these pile up hard around the real
    steps, so a cluster holding a decent share of all the vectors is a real
    direction and a thin one is noise.
    """
    if len(vectors) == 0:
        return []
    lengths = np.linalg.norm(vectors, axis=1)
    alive = lengths > 1e-12
    vectors, lengths = vectors[alive], lengths[alive]
    if len(vectors) == 0:
        return []

    order = np.argsort(lengths)
    vectors, lengths = vectors[order], lengths[order]
    units = vectors / lengths[:, None]

    total = len(vectors)
    floor = max(8, int(min_share * total))
    used = np.zeros(total, dtype=bool)
    found = []
    for i in range(total):
        if used[i]:
            continue
        # The seed is the shortest vector left, which sits at the bottom of
        # its own cluster, so a window centred on it is lopsided and its mean
        # comes out short. Re-centre on the mean and gather again until it
        # settles, which lands on the middle of the cluster instead.
        centre, near = vectors[i], None
        for _ in range(8):
            length = np.linalg.norm(centre)
            if length < 1e-12:
                break
            window = (~used)
            window &= np.abs(lengths - length) <= length_tol * length
            window &= (units @ (centre / length)) >= cos_min
            if not np.any(window):
                break
            moved = vectors[window].mean(axis=0)
            settled = np.linalg.norm(moved - centre) <= 1e-6 * length
            centre, near = moved, window
            if settled:
                break
        if near is None:
            used[i] = True
            continue
        population = int(np.count_nonzero(near))
        if population >= floor:
            found.append((centre, population))
        used |= near
    return found


def choose_lattice_basis(candidates, parallel_cos=0.94, plane_sin=0.12):
    """The three shortest candidate steps that actually span a volume.

    Taken shortest first, skipping anything parallel to a step already held
    or lying in the plane of two of them, which is what rules out diagonals.
    """
    ranked = sorted(candidates, key=lambda item: np.linalg.norm(item[0]))
    chosen = []
    for vector, _ in ranked:
        unit = vector / np.linalg.norm(vector)
        if len(chosen) == 1:
            if abs(unit @ (chosen[0] / np.linalg.norm(chosen[0]))) > parallel_cos:
                continue
        elif len(chosen) == 2:
            normal = np.cross(chosen[0], chosen[1])
            norm = np.linalg.norm(normal)
            if norm < 1e-12 or abs(unit @ (normal / norm)) < plane_sin:
                continue
        chosen.append(vector)
        if len(chosen) == 3:
            return np.column_stack(chosen)
    return None


def fit_lattice(coords, basis, origin, chunk=2_000_000, rounds=3):
    """Least squares refine the basis and origin against every point.

    Each round snaps the points to whole cells, then solves for the basis and
    origin that best explain those cells. Fitting over the whole cloud rather
    than one neighbourhood is what stops a small error in the step length
    from drifting into whole cells far from the origin.
    """
    total = len(coords)
    for _ in range(rounds):
        inverse = np.linalg.inv(basis)
        # Normal equations for coords ~ cells @ basis.T + origin, gathered in
        # chunks so a 30M point cloud never needs a design matrix of its own.
        left = np.zeros((4, 4), dtype=np.float64)
        right = np.zeros((4, 3), dtype=np.float64)
        for start in range(0, total, chunk):
            block = coords[start:start + chunk].astype(np.float64)
            cells = np.rint((block - origin) @ inverse.T)
            design = np.empty((len(block), 4), dtype=np.float64)
            design[:, :3] = cells
            design[:, 3] = 1.0
            left += design.T @ design
            right += design.T @ block
        try:
            solved = np.linalg.solve(left, right)
        except np.linalg.LinAlgError:
            break
        basis = np.ascontiguousarray(solved[:3].T)
        origin = solved[3].copy()
    return basis, origin


def lattice_report(pitch, angles, within, worst, collisions, drift):
    """One line for the panel, plus the longer version for the info bar."""
    short = (f"fit {within * 100:.1f}%  pitch "
             f"{pitch[0]:.4g} / {pitch[1]:.4g} / {pitch[2]:.4g}")
    detail = (f"Lattice: pitch {pitch[0]:.4g}, {pitch[1]:.4g}, {pitch[2]:.4g}; "
              f"angles {angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f} deg; "
              f"{within * 100:.1f}% of points within a quarter cell; "
              f"worst offset {worst:.4g}; {collisions:,} shared cells; "
              f"{drift}")
    return short, detail


class LatticeJob:
    """Find the grid the points sit on, and say how well they really sit on it.

    Reads nothing but the geometry and writes nothing but a residual
    attribute, so it is safe to run on a cloud before deciding whether to
    generate anything from it. Everything happens in the object's own
    coordinates, which is where a generated grid normally lives and where
    new points would have to be written.

    Raises ValueError with a user facing message if the setup is not usable.
    """

    CHUNK = 2_000_000

    def __init__(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            raise ValueError("Make a point cloud mesh the active object first")
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        self.obj = obj
        self.mesh = obj.data
        self.total = len(self.mesh.vertices)
        if self.total < 64:
            raise ValueError(f"'{obj.name}' has too few points to find a grid in")

        self.spacing = 0.0
        self.basis = None
        self.origin = None
        self.pitch = np.zeros(3)
        self.angles = np.zeros(3)
        self.within = 0.0
        self.worst = 0.0
        self.collisions = 0
        self.drift = ""
        self.short = ""
        self.detail = ""
        self.start = time.time()

        self.stage = 0
        self.stages = (
            ("Reading points", self._read),
            ("Estimating spacing", self._spacing),
            ("Reading grid steps", self._steps),
            ("Fitting the grid", self._fit),
            ("Checking the fit", self._check),
            ("Writing residuals", self._write),
        )
        self.weights = (0.10, 0.08, 0.22, 0.30, 0.20, 0.10)

    @property
    def stage_name(self):
        if self.stage >= len(self.stages):
            return "Done"
        return self.stages[self.stage][0]

    @property
    def progress(self):
        return min(sum(self.weights[:self.stage]), 1.0)

    def step(self, budget=0.08):
        if self.stage >= len(self.stages):
            return True
        if self.stages[self.stage][1](budget):
            self.stage += 1
        return self.stage >= len(self.stages)

    # -- stages -------------------------------------------------------------

    def _read(self, budget):
        coords = np.empty(self.total * 3, dtype=np.float32)
        self.mesh.vertices.foreach_get('co', coords)
        self.coords = coords.reshape(-1, 3)
        return True

    def _spacing(self, budget):
        self.spacing = estimate_point_spacing(self.coords)
        if self.spacing <= 0.0:
            raise ValueError("Could not measure the point spacing")
        return True

    def _steps(self, budget):
        patches = sample_patches(self.coords, self.spacing)
        if not patches:
            raise ValueError("Could not find a dense enough patch to read the "
                             "grid from, the cloud may be too sparse")
        vectors = np.vstack([neighbour_displacements(p) for p in patches])
        candidates = cluster_lattice_directions(canonical_directions(vectors))
        if len(candidates) < 3:
            raise ValueError("The points do not repeat on a regular grid, "
                             "only "
                             f"{len(candidates)} clear step direction(s) found")
        basis = choose_lattice_basis(candidates)
        if basis is None:
            raise ValueError("Found grid steps but they do not span a volume, "
                             "so no three dimensional grid fits these points")
        self.basis = basis.astype(np.float64)
        return True

    def _fit(self, budget):
        # Anchor on a real point so the first rounding is sane, then let the
        # least squares fit move the origin where it belongs.
        origin = self.coords[len(self.coords) // 2].astype(np.float64)
        self.basis, self.origin = fit_lattice(self.coords, self.basis, origin)

        self.pitch = np.linalg.norm(self.basis, axis=0)
        if np.any(self.pitch < 1e-9):
            raise ValueError("The fitted grid collapsed, the points are "
                             "probably not on a regular grid")
        units = self.basis / self.pitch
        self.angles = np.degrees([
            np.arccos(np.clip(units[:, 1] @ units[:, 2], -1.0, 1.0)),
            np.arccos(np.clip(units[:, 0] @ units[:, 2], -1.0, 1.0)),
            np.arccos(np.clip(units[:, 0] @ units[:, 1], -1.0, 1.0)),
        ])
        return True

    def _check(self, budget):
        inverse = np.linalg.inv(self.basis)
        quarter = 0.25 * float(self.pitch.min())

        self.cells = np.empty((self.total, 3), dtype=np.int32)
        self.residual = np.empty(self.total, dtype=np.float32)
        radius = np.empty(self.total, dtype=np.float32)
        for start in range(0, self.total, self.CHUNK):
            end = min(start + self.CHUNK, self.total)
            block = self.coords[start:end].astype(np.float64)
            offset = block - self.origin
            cells = np.rint(offset @ inverse.T)
            self.cells[start:end] = cells.astype(np.int32)
            self.residual[start:end] = np.linalg.norm(
                offset - cells @ self.basis.T, axis=1)
            radius[start:end] = np.linalg.norm(offset, axis=1)

        self.within = float(np.count_nonzero(self.residual <= quarter)) / self.total
        self.worst = float(self.residual.max())

        # Drift shows up as the fit falling off with distance from the origin,
        # while plain jitter fits equally well everywhere. Measured as the
        # share of points that land on a cell rather than as the raw offset:
        # rounding caps an offset at half a cell, so once drift passes that
        # the offset stops growing and hides, but the share keeps dropping.
        far = float(radius.max())
        if far > 0:
            edges = np.linspace(0.0, far, 6)
            shares = []
            for i in range(5):
                band = (radius >= edges[i])
                band &= (radius <= edges[i + 1]) if i == 4 else (radius < edges[i + 1])
                counted = int(np.count_nonzero(band))
                shares.append(float(np.count_nonzero(self.residual[band] <= quarter))
                              / counted if counted else float('nan'))
            shape = " ".join("--" if s != s else f"{s * 100:.0f}%" for s in shares)
            real = [s for s in shares if s == s]
            falls = (len(real) >= 2 and real[0] > 0.5
                     and real[-1] < min(0.7 * real[0], real[0] - 0.15))
            self.drift = (f"fit {'FALLS OFF' if falls else 'steady'} with "
                          f"distance ({shape})")
        self.coords = None
        return True

    def _write(self, budget):
        keys = ((self.cells[:, 0].astype(np.int64) << 42)
                ^ (self.cells[:, 1].astype(np.int64) << 21)
                ^ self.cells[:, 2].astype(np.int64))
        unique = np.unique(keys)
        self.collisions = int(self.total - unique.size)
        self.cells = None

        mesh = self.obj.data
        if len(mesh.vertices) != self.total:
            raise ValueError("The point cloud changed while it was measured")
        attr = mesh.attributes.get(GRID_RESIDUAL_ATTR)
        if attr is not None and (attr.domain != 'POINT' or attr.data_type != 'FLOAT'):
            mesh.attributes.remove(attr)
            attr = None
        if attr is None:
            attr = mesh.attributes.new(name=GRID_RESIDUAL_ATTR, type='FLOAT',
                                       domain='POINT')
        attr.data.foreach_set("value", self.residual)
        mesh.update()

        self.short, self.detail = lattice_report(
            self.pitch, self.angles, self.within, self.worst,
            self.collisions, self.drift)
        return True

    def finish(self):
        return self.detail

    def apply_to_scene(self, scene):
        """Hand the fitted grid to the growth settings, pitch included."""
        scene.pc_grid_basis = tuple(self.basis.T.ravel())
        scene.pc_grid_origin = tuple(self.origin)
        scene.pc_grid_pitch = tuple(self.pitch)
        scene.pc_grid_fit = self.within
        scene.pc_grid_status = self.short


class MESH_OT_AnalyzeLattice(bpy.types.Operator):
    bl_idname = "mesh.pc_analyze_lattice"
    bl_label = "Analyze Lattice"
    bl_description = (
        "Step 1: Works out the grid the points sit on, at whatever "
        "orientation it happens to have, and reports how well they really "
        "sit on it.\n\n"
        "Reads the geometry and writes only the 'pc_lattice_residual' "
        "attribute, so it never moves a point. The step lengths it finds are "
        "written into Pitch below, where they can be corrected by hand. "
        "Check the fit before generating anything: a low percentage, or an "
        "offset that grows with distance, means one grid does not describe "
        "this cloud and generated points would land in the wrong places. "
        "Press Esc while it runs to cancel"
    )
    bl_options = {'REGISTER'}

    _timer = None
    _job = None

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and not context.scene.pc_grid_running)

    def _cleanup(self, context):
        scene = context.scene
        scene.pc_grid_running = False
        scene.pc_grid_progress = 0.0
        scene.pc_grid_stage = ""
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._job = None

    def _announce(self, context, job):
        job.apply_to_scene(context.scene)
        level = 'WARNING' if job.within < 0.9 or "FALLS OFF" in job.drift else 'INFO'
        self.report({level}, job.detail)

    def execute(self, context):
        scene = context.scene
        try:
            job = LatticeJob(context)
            while not job.step(1.0):
                pass
            job.finish()
        except ValueError as exc:
            scene.pc_grid_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self._announce(context, job)
        return {'FINISHED'}

    def invoke(self, context, event):
        if getattr(context, "window", None) is None:
            return self.execute(context)

        scene = context.scene
        try:
            self._job = LatticeJob(context)
        except ValueError as exc:
            scene.pc_grid_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.pc_grid_running = True
        scene.pc_grid_progress = 0.0
        scene.pc_grid_stage = self._job.stage_name
        scene.pc_grid_status = ""
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        if event.type == 'ESC':
            self._cleanup(context)
            scene.pc_grid_status = "Cancelled"
            tag_redraw_sidebar(context)
            self.report({'WARNING'}, "Lattice analysis cancelled")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        try:
            done = self._job.step(0.08)
        except ValueError as exc:
            self._cleanup(context)
            scene.pc_grid_status = ""
            tag_redraw_sidebar(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.pc_grid_progress = self._job.progress
        scene.pc_grid_stage = self._job.stage_name
        tag_redraw_sidebar(context)

        if not done:
            return {'RUNNING_MODAL'}

        job = self._job
        job.finish()
        self._cleanup(context)
        self._announce(context, job)
        tag_redraw_sidebar(context)
        return {'FINISHED'}


# --- 1d. GROWTH LOGIC ---

# Attribute kinds a new point can inherit: RNA property, width, numpy type.
GROW_ATTR_KINDS = {
    'FLOAT': ("value", 1, np.float32),
    'INT': ("value", 1, np.int32),
    'INT8': ("value", 1, np.int32),
    'BOOLEAN': ("value", 1, bool),
    'FLOAT2': ("vector", 2, np.float32),
    'FLOAT_VECTOR': ("vector", 3, np.float32),
    'FLOAT_COLOR': ("color", 4, np.float32),
    'BYTE_COLOR': ("color", 4, np.float32),
}


def pack_lattice_cells(cells, margin=2):
    """Whole cells to one sorted integer each, plus the key space steps.

    A margin of empty cells is left around the occupied range so that
    stepping to a neighbour can never wrap onto the far side of the grid and
    be mistaken for one. `margin` therefore has to cover every step a caller
    intends to take.
    """
    low = cells.min(axis=0) - margin
    span = (cells.max(axis=0) - low + 1 + margin).astype(np.int64)
    if float(span[0]) * float(span[1]) * float(span[2]) > 4.0e18:
        raise ValueError("The grid spans too many cells to index, which "
                         "usually means the Pitch is far too small")
    shifted = (cells - low).astype(np.int64)
    keys = (shifted[:, 0] * span[1] + shifted[:, 1]) * span[2] + shifted[:, 2]
    return keys, low, span


def unpack_lattice_keys(keys, low, span):
    """Back from packed keys to whole cells."""
    k = keys % span[2]
    rest = keys // span[2]
    j = rest % span[1]
    i = rest // span[1]
    return np.stack([i, j, k], axis=1).astype(np.int64) + low


def scene_lattice(scene):
    """The grid to grow on: directions as found, lengths as they now read.

    Editing Pitch rescales the step it belongs to and leaves its direction
    alone, so a corrected number never rotates the grid.
    """
    steps = np.array(scene.pc_grid_basis, dtype=np.float64).reshape(3, 3)
    lengths = np.linalg.norm(steps, axis=1)
    if np.any(lengths < 1e-9):
        raise ValueError("No grid yet, press Analyze Lattice first")
    pitch = np.array(scene.pc_grid_pitch, dtype=np.float64)
    if np.all(pitch > 1e-9):
        steps = steps / lengths[:, None] * pitch[:, None]
    basis = np.ascontiguousarray(steps.T)
    if abs(np.linalg.det(basis)) < 1e-12:
        raise ValueError("The grid steps do not span a volume, press Analyze "
                         "Lattice again")
    return basis, np.array(scene.pc_grid_origin, dtype=np.float64)


def lattice_shape_offsets(basis, reach=3, radius=2.2, cap=150):
    """Cells around a point to judge the shape of the part it belongs to.

    Chosen by real distance rather than by cell counts, so an uneven grid
    still gets a roughly round window. One cell in every direction is not
    enough: well inside a part every immediate neighbour is filled and the
    point looks the same in all directions, which says nothing about whether
    it sits in a sheet or in a strut.
    """
    span = float(np.linalg.norm(basis, axis=0).max()) * radius
    cells, vectors = [], []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            for k in range(-reach, reach + 1):
                if i == j == k == 0:
                    continue
                vector = basis @ np.array([i, j, k], dtype=np.float64)
                if np.linalg.norm(vector) <= span:
                    cells.append((i, j, k))
                    vectors.append(vector)
    if not cells:
        return (np.array([(1, 0, 0)], dtype=np.int64),
                np.array([basis @ np.array([1.0, 0.0, 0.0])]))
    cells = np.array(cells, dtype=np.int64)
    vectors = np.array(vectors, dtype=np.float64)
    order = np.argsort(np.linalg.norm(vectors, axis=1))[:cap]
    return cells[order], vectors[order]


def lattice_spread(occupied, seed_keys, offset_keys, offset_vectors):
    """How the filled neighbours of each seed are spread out, as a covariance.

    Kept as its six distinct terms rather than as a matrix, because all that
    is ever asked of it is how much spread lies along one direction, and that
    is a short sum over those terms.
    """
    count = seed_keys.size
    spread = np.zeros((count, 6), dtype=np.float64)
    filled = np.zeros(count, dtype=np.int32)
    last = occupied.size - 1
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

    for offset, vector in zip(offset_keys, offset_vectors):
        probe = seed_keys + offset
        idx = np.searchsorted(occupied, probe)
        np.clip(idx, 0, last, out=idx)
        here = occupied[idx] == probe
        filled += here
        weight = here.astype(np.float64)
        for slot, (a, b) in enumerate(pairs):
            spread[:, slot] += weight * (vector[a] * vector[b])
    return spread, filled


def spread_along(spread, directions):
    """Spread of each seed's neighbours along each direction, d dot C dot d."""
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    return (np.outer(spread[:, 0], x * x)
            + np.outer(spread[:, 1], y * y)
            + np.outer(spread[:, 2], z * z)
            + 2.0 * np.outer(spread[:, 3], x * y)
            + 2.0 * np.outer(spread[:, 4], x * z)
            + 2.0 * np.outer(spread[:, 5], y * z))


class GrowthJob:
    """Add points around a selection, on the grid the cloud already uses.

    Every candidate is a whole cell of the fitted grid, so new points land in
    the pattern the existing ones follow, and a cell that is already taken is
    dropped rather than doubled up. That single rule is what makes the tool
    step: growing along a strut only produces points at its tip, because
    every cell behind the tip is occupied, and growth stops by itself once it
    reaches another part of the cloud.

    Raises ValueError with a user facing message if the setup is not usable.
    """

    CHUNK = 2_000_000
    ALIGNED = 0.5      # cosine splitting "across the thin direction" from "along it"

    def __init__(self, context):
        scene = context.scene
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            raise ValueError("Make a point cloud mesh the active object first")
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        self.obj = obj
        self.mesh = obj.data
        self.total = len(self.mesh.vertices)
        if self.total == 0:
            raise ValueError(f"'{obj.name}' has no vertices")

        self.basis, self.origin = scene_lattice(scene)
        self.mode = scene.pc_grow_mode
        self.steps = int(scene.pc_grow_steps)
        self.added = 0
        self.per_step = []
        self.skipped = []
        self.start = time.time()

        self.stage = 0
        self.done_steps = 0
        self.stages = (
            ("Reading points", self._read),
            ("Mapping to the grid", self._map),
            ("Growing", self._grow),
            ("Placing points", self._place),
            ("Copying attributes", self._inherit),
        )
        self.weights = (0.10, 0.22, 0.38, 0.16, 0.14)

    @property
    def stage_name(self):
        if self.stage >= len(self.stages):
            return "Done"
        name = self.stages[self.stage][0]
        if name == "Growing" and self.steps > 1:
            return f"Growing {min(self.done_steps + 1, self.steps)}/{self.steps}"
        return name

    @property
    def progress(self):
        done = sum(self.weights[:self.stage])
        if self.stage < len(self.stages) and self.stages[self.stage][1] is self._grow:
            done += self.weights[self.stage] * (self.done_steps / max(self.steps, 1))
        return min(done, 1.0)

    def step(self, budget=0.08):
        if self.stage >= len(self.stages):
            return True
        if self.stages[self.stage][1](budget):
            self.stage += 1
        return self.stage >= len(self.stages)

    # -- stages -------------------------------------------------------------

    def _read(self, budget):
        coords = np.empty(self.total * 3, dtype=np.float32)
        self.mesh.vertices.foreach_get('co', coords)
        self.coords = coords.reshape(-1, 3)

        chosen = np.empty(self.total, dtype=bool)
        self.mesh.vertices.foreach_get('select', chosen)
        self.chosen = np.nonzero(chosen)[0]
        if self.chosen.size == 0:
            raise ValueError("Select the points to grow from first, in edit "
                             "mode, then run this")
        return True

    def _map(self, budget):
        inverse = np.linalg.inv(self.basis)
        cells = np.empty((self.total, 3), dtype=np.int64)
        for start in range(0, self.total, self.CHUNK):
            end = min(start + self.CHUNK, self.total)
            block = self.coords[start:end].astype(np.float64) - self.origin
            cells[start:end] = np.rint(block @ inverse.T).astype(np.int64)

        keys, self.low, self.span = pack_lattice_cells(cells, margin=self.steps + 2)
        order = np.argsort(keys)
        self.occupied = keys[order]
        # First point in each cell, so an inherited attribute has one source
        # even where two points share a cell.
        unique, first = np.unique(self.occupied, return_index=True)
        self.occupied = unique
        self.owner = order[first]

        self.seed_keys = np.unique(keys[self.chosen])
        # Ancestor of every seed, so points grown from grown points still
        # inherit from a real point rather than from one that does not exist.
        idx = np.searchsorted(self.occupied, self.seed_keys)
        np.clip(idx, 0, self.occupied.size - 1, out=idx)
        self.seed_parents = self.owner[idx]

        step_x = int(self.span[1] * self.span[2])
        step_y = int(self.span[2])
        moves = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        self.dir_keys = np.array([i * step_x + j * step_y + k for i, j, k in moves],
                                 dtype=np.int64)
        vectors = np.array(moves, dtype=np.float64) @ self.basis.T
        self.dir_units = vectors / np.linalg.norm(vectors, axis=1)[:, None]

        if self.mode != 'UNIFORM':
            around, self.near_vectors = lattice_shape_offsets(self.basis)
            self.near_keys = (around[:, 0] * step_x + around[:, 1] * step_y
                              + around[:, 2])

        self.new_keys = []
        self.new_parents = []
        return True

    def _grow(self, budget):
        if self.done_steps >= self.steps or self.seed_keys.size == 0:
            self.done_steps = self.steps
            return True

        count = self.seed_keys.size
        seat = np.tile(np.arange(count), 6)
        way = np.repeat(np.arange(6), count)
        keys = (self.seed_keys[None, :] + self.dir_keys[:, None]).ravel()
        parents = np.tile(self.seed_parents, 6)

        # A cell that already holds a point is never filled twice.
        idx = np.searchsorted(self.occupied, keys)
        np.clip(idx, 0, self.occupied.size - 1, out=idx)
        free = self.occupied[idx] != keys

        if self.mode != 'UNIFORM' and np.any(free):
            # Only seeds with somewhere to go need their shape read, which on
            # a solid selection is just the points on its surface.
            busy = np.unique(seat[free])
            spread, filled = lattice_spread(self.occupied, self.seed_keys[busy],
                                            self.near_keys, self.near_vectors)
            along = spread_along(spread, self.dir_units)
            widest = along.max(axis=1)[:, None]
            # A sheet has one direction its neighbours barely spread along and
            # a strut has two, so asking how much spread lies along a
            # direction handles both without having to tell them apart.
            # Growing where there is little spread makes a part thicker;
            # growing where there is a lot carries it further along itself.
            safe = np.where(widest > 1e-12, widest, 1.0)
            share = np.where(widest > 1e-12, along / safe, 1.0)
            allowed = (share <= self.ALIGNED if self.mode == 'THICKEN'
                       else share >= self.ALIGNED)
            # Nothing around it to judge by: let it grow either way rather
            # than guess and grow nowhere.
            allowed |= (filled == 0)[:, None]

            wanted = np.zeros(seat.size, dtype=bool)
            wanted[free] = allowed[np.searchsorted(busy, seat[free]), way[free]]
            free &= wanted

        keys, parents = keys[free], parents[free]
        if keys.size == 0:
            self.done_steps = self.steps
            return True

        fresh, first = np.unique(keys, return_index=True)
        parents = parents[first]

        self.new_keys.append(fresh)
        self.new_parents.append(parents)
        self.occupied = np.union1d(self.occupied, fresh)
        self.seed_keys, self.seed_parents = fresh, parents
        self.per_step.append(int(fresh.size))

        self.done_steps += 1
        return self.done_steps >= self.steps

    def _place(self, budget):
        if not self.new_keys:
            self.added = 0
            return True
        keys = np.concatenate(self.new_keys)
        self.parents = np.concatenate(self.new_parents)
        cells = unpack_lattice_keys(keys, self.low, self.span)
        self.positions = (cells.astype(np.float64) @ self.basis.T
                          + self.origin).astype(np.float32)
        self.added = int(len(keys))
        self.new_keys = self.new_parents = None
        return True

    def _inherit(self, budget):
        if self.added == 0:
            return True

        mesh = self.obj.data
        if len(mesh.vertices) != self.total:
            raise ValueError("The point cloud changed while it was growing, "
                             "nothing was added")

        grown = self.total + self.added
        mesh.vertices.add(self.added)

        coords = np.empty(grown * 3, dtype=np.float32)
        coords[:self.total * 3] = self.coords.ravel()
        coords[self.total * 3:] = self.positions.ravel()
        mesh.vertices.foreach_set('co', coords)
        self.coords = None

        for attr in list(mesh.attributes):
            name = attr.name
            if (attr.domain != 'POINT' or name == 'position'
                    or name.startswith('.')):
                continue
            kind = GROW_ATTR_KINDS.get(attr.data_type)
            if kind is None:
                self.skipped.append(name)
                continue
            prop, width, dtype = kind
            values = np.empty(grown * width, dtype=dtype)
            try:
                attr.data.foreach_get(prop, values)
            except (TypeError, RuntimeError):
                self.skipped.append(name)
                continue
            if width == 1:
                values[self.total:] = values[self.parents]
            else:
                block = values.reshape(-1, width)
                block[self.total:] = block[self.parents]
            attr.data.foreach_set(prop, values)

        # Only the new points stay selected, so pressing again carries the
        # growth on from the front rather than from where it started.
        selected = np.zeros(grown, dtype=bool)
        selected[self.total:] = True
        if len(mesh.polygons):
            mesh.polygons.foreach_set("select",
                                      np.zeros(len(mesh.polygons), dtype=bool))
        if len(mesh.edges):
            mesh.edges.foreach_set("select", np.zeros(len(mesh.edges), dtype=bool))
        mesh.vertices.foreach_set("select", selected)
        mesh.update()
        return True

    def finish(self):
        elapsed = time.time() - self.start
        if self.added == 0:
            return ("No room to grow: every neighbouring cell of the "
                    "selection is already filled")
        shape = " + ".join(f"{n:,}" for n in self.per_step)
        note = (f"Added {self.added:,} points ({shape}) around "
                f"{self.chosen.size:,} selected, {elapsed:.1f}s")
        if self.skipped:
            note += ". Could not carry over: " + ", ".join(sorted(set(self.skipped)))
        return note


class MESH_OT_Grow(bpy.types.Operator):
    bl_idname = "mesh.pc_grow"
    bl_label = "Grow"
    bl_description = (
        "Step 3: Adds points around the selected ones, on the grid found by "
        "Analyze Lattice, and never where a point already is.\n\n"
        "Select the points to grow from in edit mode first, for instance the "
        "thin parts found by Thickness. Only the new points are left "
        "selected, so pressing again carries the growth on from the new "
        "front. Press Esc while it runs to cancel"
    )
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _job = None

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and not context.scene.pc_grow_running)

    def _cleanup(self, context):
        scene = context.scene
        scene.pc_grow_running = False
        scene.pc_grow_progress = 0.0
        scene.pc_grow_stage = ""
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._job = None

    def _announce(self, context, job, message):
        context.scene.pc_grow_status = message
        self.report({'INFO' if job.added else 'WARNING'}, message)

    def _back_to_edit(self, context):
        context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode='EDIT')

    def execute(self, context):
        scene = context.scene
        try:
            job = GrowthJob(context)
            while not job.step(1.0):
                pass
            message = job.finish()
        except ValueError as exc:
            scene.pc_grow_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self._announce(context, job, message)
        self._back_to_edit(context)
        return {'FINISHED'}

    def invoke(self, context, event):
        if getattr(context, "window", None) is None:
            return self.execute(context)

        scene = context.scene
        try:
            self._job = GrowthJob(context)
        except ValueError as exc:
            scene.pc_grow_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.pc_grow_running = True
        scene.pc_grow_progress = 0.0
        scene.pc_grow_stage = self._job.stage_name
        scene.pc_grow_status = ""
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        if event.type == 'ESC':
            self._cleanup(context)
            scene.pc_grow_status = "Cancelled"
            tag_redraw_sidebar(context)
            self.report({'WARNING'}, "Growth cancelled, no points were added")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        try:
            done = self._job.step(0.08)
        except ValueError as exc:
            self._cleanup(context)
            scene.pc_grow_status = ""
            tag_redraw_sidebar(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.pc_grow_progress = self._job.progress
        scene.pc_grow_stage = self._job.stage_name
        tag_redraw_sidebar(context)

        if not done:
            return {'RUNNING_MODAL'}

        job = self._job
        message = job.finish()
        self._cleanup(context)
        self._announce(context, job, message)
        self._back_to_edit(context)
        tag_redraw_sidebar(context)
        return {'FINISHED'}


# --- 1e. VIDEO STACK LOGIC ---

VIDEO_COLOR_ATTR = "Col"
VIDEO_FRAME_ATTR = "frame"
VIDEO_PX_ATTR = "px"
VIDEO_PY_ATTR = "py"

# Past this Blender goes down instead of finishing, so the job refuses.
VIDEO_POINT_CAP = 100_000_000

# Rec.709 weights, matching the linear values Blender reads out of a PNG.
VIDEO_LUMA = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)


def probe_video(path):
    """(width, height, frames) of a movie file, zeros if it cannot be read.

    Loading the file as an image datablock is enough to get its size and its
    length and costs almost nothing, so the panel can show the real numbers
    the moment a file is picked, without going near the sequencer.
    """
    if not path:
        return 0, 0, 0
    full = bpy.path.abspath(path)
    if not os.path.isfile(full):
        return 0, 0, 0
    try:
        image = bpy.data.images.load(full, check_existing=False)
    except RuntimeError:
        return 0, 0, 0
    try:
        width, height = image.size
        frames = image.frame_duration if image.source == 'MOVIE' else 0
    finally:
        bpy.data.images.remove(image)
    return int(width), int(height), max(int(frames), 0)


def video_plan(scene):
    """(width, height, frame numbers) for the current settings, or None.

    None means the range is empty, which is the one case both the panel and
    the job have to say no to.
    """
    width = max(int(scene.pc_vid_width), 1)
    height = max(int(scene.pc_vid_height), 1)
    total = int(scene.pc_vid_src_frames)
    start = max(int(scene.pc_vid_start), 1)
    end = int(scene.pc_vid_end)

    if total > 0:
        # The length is known, so the range can be trimmed to the video.
        start = min(start, total)
        end = total if end <= 0 else min(end, total)
    elif end <= 0:
        # Length unknown and no end typed in: nothing to work with.
        return None
    if end < start:
        return None

    step = max(int(scene.pc_vid_step), 1)
    return width, height, list(range(start, end + 1, step))


def video_estimate(scene):
    """(frames, points) the current settings would produce, before culling."""
    plan = video_plan(scene)
    if plan is None:
        return 0, 0
    width, height, frames = plan
    return len(frames), len(frames) * width * height


def srgb_to_linear(values):
    """sRGB display values to the scene linear colour Blender stores.

    Blender hands back the raw bytes of an 8 bit image divided by 255, so
    the numbers read out of a rendered PNG are still sRGB encoded, while a
    FLOAT_COLOR attribute is taken as scene linear. Skipping this step is
    what makes a point cloud look washed out next to its own video.
    """
    low = values / 12.92
    high = ((values + 0.055) / 1.055) ** 2.4
    return np.where(values <= 0.04045, low, high).astype(np.float32)


def video_grid(width, height, pixel):
    """Point positions and pixel indices for one frame, centred on the origin.

    Every frame shares this layout, so it is built once and reused. Blender
    hands back image rows bottom up, so row 0 is the bottom of the frame and
    the stack comes out the right way up on its own. The stored py counts
    from the top instead, the way an image editor reports a pixel.
    """
    cols = np.arange(width, dtype=np.int32)
    rows = np.arange(height, dtype=np.int32)
    gx = np.tile((cols - (width - 1) * 0.5) * pixel, height).astype(np.float32)
    gy = np.repeat((rows - (height - 1) * 0.5) * pixel,
                   width).astype(np.float32)
    return gx, gy, np.tile(cols, height), np.repeat(height - 1 - rows, width)


class VideoStackJob:
    """Turn a video into one point per pixel per frame, stacked along Z.

    Blender will not hand a script the pixels of an arbitrary video frame, so
    the frames go through the sequencer of a throwaway scene: each one is
    rendered at the requested resolution into a single reused PNG in a temp
    folder, read back with numpy, and dropped. That is one render per frame,
    which is exactly what makes the progress line and Esc worth having.

    Raises ValueError with a user facing message if the setup is not usable.
    """

    def __init__(self, context):
        scene = context.scene
        self.scene = scene
        self.collection = context.collection
        self.view_layer = context.view_layer

        if not scene.pc_vid_path:
            raise ValueError("Pick a video file first")
        self.path = bpy.path.abspath(scene.pc_vid_path)
        if not os.path.isfile(self.path):
            raise ValueError(f"'{scene.pc_vid_path}' is not a file")

        plan = video_plan(scene)
        if plan is None:
            raise ValueError("Set a frame range first, End has to be at "
                             "least Start")
        self.width, self.height, self.frames = plan
        if not self.frames:
            raise ValueError("The frame range is empty")

        estimate = len(self.frames) * self.width * self.height
        if estimate > VIDEO_POINT_CAP:
            raise ValueError(
                f"{estimate:,} points is past what Blender survives. Lower "
                f"the resolution, raise Frame Step, or shorten the range")

        self.pixel = scene.pc_vid_pixel
        self.z_step = scene.pc_vid_z
        self.cull = scene.pc_vid_cull
        self.name = os.path.splitext(os.path.basename(self.path))[0] or "Video"

        self.gx, self.gy, self.px, self.py = video_grid(
            self.width, self.height, self.pixel)

        self.parts = []        # one (co, colour, px, py, frame) per frame
        self.kept = 0
        self.done_frames = 0
        self.tmp_dir = None
        self.tmp_scene = None
        self.camera = None
        self.obj = None
        self.start = time.time()

        self.stage = 0
        self.stages = (
            ("Preparing", self._prepare),
            ("Decoding", self._decode),
            ("Placing points", self._build),
            ("Writing attributes", self._write),
        )
        self.weights = (0.03, 0.82, 0.09, 0.06)

    @property
    def stage_name(self):
        if self.stage >= len(self.stages):
            return "Done"
        name = self.stages[self.stage][0]
        if name == "Decoding":
            shown = min(self.done_frames + 1, len(self.frames))
            return f"Decoding frame {shown}/{len(self.frames)}"
        return name

    @property
    def progress(self):
        done = sum(self.weights[:self.stage])
        if (self.stage < len(self.stages)
                and self.stages[self.stage][1] is self._decode):
            done += self.weights[self.stage] * (self.done_frames
                                                / max(len(self.frames), 1))
        return min(done, 1.0)

    def step(self, budget=0.08):
        if self.stage >= len(self.stages):
            return True
        if self.stages[self.stage][1](budget):
            self.stage += 1
        return self.stage >= len(self.stages)

    # -- stages -------------------------------------------------------------

    def _source_size(self, strip):
        """Pixel size of the video, asked of the strip before the probe."""
        elements = getattr(strip, "elements", ())
        element = elements[0] if len(elements) else None
        width = getattr(element, "orig_width", 0)
        height = getattr(element, "orig_height", 0)
        if width and height:
            return int(width), int(height)
        return (max(int(self.scene.pc_vid_src_w), 1),
                max(int(self.scene.pc_vid_src_h), 1))

    def _prepare(self, budget):
        self.tmp_dir = tempfile.mkdtemp(prefix="pc_video_stack_")

        scene = bpy.data.scenes.new("PC Video Decode")
        self.tmp_scene = scene
        editor = scene.sequence_editor or scene.sequence_editor_create()

        # The collection was renamed from `sequences` to `strips` in 4.4.
        # Both are handled so the file still runs on the older Blender that
        # bl_info promises.
        strips = getattr(editor, "strips", None)
        if strips is None:
            strips = editor.sequences
        try:
            strip = strips.new_movie(name="source", filepath=self.path,
                                     channel=1, frame_start=1)
        except RuntimeError as exc:
            raise ValueError(f"Blender could not open the video: {exc}")

        # A new movie strip sits at its own pixel size, so a 4K clip rendered
        # at 512 wide would come out cropped to its middle. Scaling the strip
        # onto the output maps the whole frame across it, and doing the two
        # axes apart lets a deliberately squashed resolution through.
        src_w, src_h = self._source_size(strip)
        strip.transform.offset_x = 0
        strip.transform.offset_y = 0
        strip.transform.scale_x = self.width / src_w
        strip.transform.scale_y = self.height / src_h
        try:
            # Subsampling is the filter that holds up when shrinking.
            strip.transform.filter = 'SUBSAMPLING_3X3'
        except TypeError:
            pass

        render = scene.render
        render.resolution_x = self.width
        render.resolution_y = self.height
        render.resolution_percentage = 100
        render.use_sequencer = True
        render.use_compositing = False
        render.use_overwrite = True
        render.use_file_extension = True
        render.filepath = os.path.join(self.tmp_dir, "f")
        try:
            # 5.0 split video out of file_format behind media_type, and a
            # still is the only thing that can be read back frame by frame.
            render.image_settings.media_type = 'IMAGE'
        except (AttributeError, TypeError):
            pass
        render.image_settings.file_format = 'PNG'
        render.image_settings.color_mode = 'RGBA'
        render.image_settings.color_depth = '8'
        render.image_settings.compression = 0   # write speed over file size
        try:
            render.engine = 'BLENDER_WORKBENCH'
        except TypeError:
            pass

        # Anything other than a straight view transform grades the colours on
        # the way out, and the points have to carry what the video holds.
        for name, value in (("view_transform", 'Standard'), ("look", 'None'),
                            ("exposure", 0.0), ("gamma", 1.0)):
            try:
                setattr(scene.view_settings, name, value)
            except TypeError:
                pass

        # The sequencer is what gets rendered here, but a scene with no
        # camera still refuses to start a render at all.
        camera_data = bpy.data.cameras.new("PC Video Camera")
        self.camera = bpy.data.objects.new("PC Video Camera", camera_data)
        scene.collection.objects.link(self.camera)
        scene.camera = self.camera

        scene.frame_start = 1
        scene.frame_end = max(self.frames)
        return True

    def _decode(self, budget):
        began = time.time()
        while self.done_frames < len(self.frames):
            self._one_frame(self.frames[self.done_frames])
            self.done_frames += 1
            if time.time() - began >= budget:
                break
        return self.done_frames >= len(self.frames)

    def _one_frame(self, number):
        """Render one video frame and keep its pixels as points.

        Rendering a one frame animation range is what actually moves the
        video on. Moving the playhead of the throwaway scene does not: the
        render still comes back as frame 1 every time, and the only other
        way round it is to drag the playhead of the scene the user is
        sitting in, which is not ours to touch.
        """
        scene = self.tmp_scene
        try:
            bpy.ops.render.render(animation=True, scene=scene.name,
                                  frame_start=number, frame_end=number)
        except RuntimeError as exc:
            raise ValueError(f"Rendering frame {number} failed: {exc}")

        path = scene.render.frame_path(frame=number)
        if not os.path.isfile(path):
            raise ValueError(f"Frame {number} produced no image")

        image = bpy.data.images.load(path, check_existing=False)
        try:
            width, height = image.size
            if width != self.width or height != self.height:
                raise ValueError(
                    f"Frame {number} came back {width}x{height} instead of "
                    f"{self.width}x{self.height}")
            buf = np.empty(width * height * 4, dtype=np.float32)
            image.pixels.foreach_get(buf)
        finally:
            bpy.data.images.remove(image)
            # Every frame writes its own file, so drop it now and keep the
            # temp folder at one image whatever the clip length is.
            try:
                os.remove(path)
            except OSError:
                pass

        rgba = buf.reshape(-1, 4)
        rgba[:, :3] = srgb_to_linear(rgba[:, :3])   # alpha stays as it is
        gx, gy, px, py = self.gx, self.gy, self.px, self.py
        if self.cull > 0.0:
            keep = np.nonzero(rgba[:, :3] @ VIDEO_LUMA >= self.cull)[0]
            if keep.size == 0:
                return
            rgba = rgba[keep]
            gx, gy = gx[keep], gy[keep]
            px, py = px[keep], py[keep]

        count = len(rgba)
        coords = np.empty((count, 3), dtype=np.float32)
        coords[:, 0] = gx
        coords[:, 1] = gy
        coords[:, 2] = self.done_frames * self.z_step
        self.parts.append((coords, rgba, px, py,
                           np.full(count, number, dtype=np.int32)))
        self.kept += count

    def _build(self, budget):
        if self.kept == 0:
            raise ValueError("Every pixel was skipped, lower Skip Below")

        self.coords = np.concatenate([part[0] for part in self.parts])
        self.colors = np.concatenate([part[1] for part in self.parts])
        self.pixel_x = np.concatenate([part[2] for part in self.parts])
        self.pixel_y = np.concatenate([part[3] for part in self.parts])
        self.numbers = np.concatenate([part[4] for part in self.parts])
        self.parts = []

        mesh = bpy.data.meshes.new(self.name)
        mesh.vertices.add(self.kept)
        mesh.vertices.foreach_set('co', self.coords.ravel())
        mesh.update()

        self.obj = bpy.data.objects.new(self.name, mesh)
        self.collection.objects.link(self.obj)
        return True

    def _write(self, budget):
        mesh = self.obj.data
        for name, kind, prop, values in (
                (VIDEO_COLOR_ATTR, 'FLOAT_COLOR', "color", self.colors),
                (VIDEO_FRAME_ATTR, 'INT', "value", self.numbers),
                (VIDEO_PX_ATTR, 'INT', "value", self.pixel_x),
                (VIDEO_PY_ATTR, 'INT', "value", self.pixel_y)):
            attr = mesh.attributes.get(name)
            if attr is None:
                attr = mesh.attributes.new(name=name, type=kind,
                                           domain='POINT')
            attr.data.foreach_set(prop, values.ravel())

        if VIDEO_COLOR_ATTR in mesh.color_attributes:
            mesh.color_attributes.active_color_name = VIDEO_COLOR_ATTR
            mesh.color_attributes.default_color_name = VIDEO_COLOR_ATTR
        mesh.update()

        for other in self.view_layer.objects:
            other.select_set(False)
        self.obj.select_set(True)
        self.view_layer.objects.active = self.obj
        return True

    # -- teardown -----------------------------------------------------------

    def cleanup(self):
        """Drop the throwaway scene and the temp folder, whatever happened."""
        if self.tmp_scene is not None:
            camera, self.camera = self.camera, None
            try:
                bpy.data.scenes.remove(self.tmp_scene, do_unlink=True)
            except (ReferenceError, RuntimeError):
                pass
            self.tmp_scene = None
            if camera is not None:
                data = camera.data
                try:
                    bpy.data.objects.remove(camera, do_unlink=True)
                    bpy.data.cameras.remove(data)
                except (ReferenceError, RuntimeError):
                    pass
        if self.tmp_dir:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            self.tmp_dir = None

    def cancel(self):
        """Throw away a run that was stopped part way through."""
        if self.obj is not None:
            mesh = self.obj.data
            try:
                bpy.data.objects.remove(self.obj, do_unlink=True)
                bpy.data.meshes.remove(mesh)
            except (ReferenceError, RuntimeError):
                pass
            self.obj = None
        self.parts = []
        self.cleanup()

    def finish(self):
        self.cleanup()
        seconds = time.time() - self.start
        return (f"{self.kept:,} points, {len(self.frames)} frames, "
                f"{seconds:.1f}s")


class MESH_OT_VideoStack(bpy.types.Operator):
    bl_idname = "mesh.pc_video_stack"
    bl_label = "Generate"
    bl_description = (
        "Step 5: Turns the video into a point cloud. Every pixel of every "
        "frame in the range becomes one point, and the frames are stacked "
        "along Z at the spacing set in step 4.\n\n"
        "Each point carries the colour of its pixel in 'Col', the video "
        "frame it came from in 'frame', and where it sat in that frame in "
        "'px' and 'py'. The result is one new object, so the rest of the "
        "suite can work on it straight away. Press Esc while it runs to "
        "cancel"
    )
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _job = None

    @classmethod
    def poll(cls, context):
        return not context.scene.pc_vid_running

    def _cleanup(self, context):
        scene = context.scene
        scene.pc_vid_running = False
        scene.pc_vid_progress = 0.0
        scene.pc_vid_stage = ""
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._job = None

    def _announce(self, context, job, message):
        context.scene.pc_vid_status = message
        if job.obj is not None:
            show_color_attribute(context, job.obj.data, VIDEO_COLOR_ATTR)
        self.report({'INFO'}, message)

    def execute(self, context):
        """Blocking path: used from scripts and in background mode."""
        scene = context.scene
        job = None
        try:
            job = VideoStackJob(context)
            while not job.step(1.0):
                pass
            message = job.finish()
        except ValueError as exc:
            if job is not None:
                job.cancel()
            scene.pc_vid_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self._announce(context, job, message)
        return {'FINISHED'}

    def invoke(self, context, event):
        """Modal path: used when the button is clicked in the sidebar."""
        if getattr(context, "window", None) is None:
            return self.execute(context)

        scene = context.scene
        try:
            self._job = VideoStackJob(context)
        except ValueError as exc:
            self._job = None
            scene.pc_vid_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.pc_vid_running = True
        scene.pc_vid_progress = 0.0
        scene.pc_vid_stage = self._job.stage_name
        scene.pc_vid_status = ""
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        if event.type == 'ESC':
            self._job.cancel()
            self._cleanup(context)
            scene.pc_vid_status = "Cancelled"
            tag_redraw_sidebar(context)
            self.report({'WARNING'}, "Cancelled, no points were made")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        try:
            done = self._job.step(0.08)
        except ValueError as exc:
            self._job.cancel()
            self._cleanup(context)
            scene.pc_vid_status = ""
            tag_redraw_sidebar(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.pc_vid_progress = self._job.progress
        scene.pc_vid_stage = self._job.stage_name
        tag_redraw_sidebar(context)

        if not done:
            return {'RUNNING_MODAL'}

        job = self._job
        message = job.finish()
        self._cleanup(context)
        self._announce(context, job, message)
        tag_redraw_sidebar(context)
        return {'FINISHED'}


# --- 2. OPERATORS ---

class MESH_OT_InvertSelection(bpy.types.Operator):
    bl_idname = "mesh.pc_invert_selection"
    bl_label = "Invert Selection"
    bl_description = "Step 5b: Inverts the current vertex selection (select unselected, deselect selected)"
    def execute(self, context):
        if context.active_object is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        if context.active_object.mode != 'EDIT': bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='INVERT')
        return {'FINISHED'}

class MESH_OT_SelectNoise(bpy.types.Operator):
    bl_idname = "mesh.pc_select_noise"
    bl_label = "Select Noise"
    bl_description = "Step 3a: Selects unclustered/noise vertices (those with cluster ID = -1)"
    attr_target: bpy.props.StringProperty()
    def execute(self, context):
        if context.active_object is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        obj = context.active_object
        if obj.mode != 'EDIT': bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        layer = bm.verts.layers.int.get(self.attr_target)
        if layer:
            for v in bm.verts: v.select = (v[layer] == -1)
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}

class MESH_OT_SelectCluster(bpy.types.Operator):
    bl_idname = "mesh.pc_select_cluster"
    bl_label = "Select Cluster"
    bl_description = "Step 5a: Selects all clustered vertices (those with cluster ID != -1)"
    attr_target: bpy.props.StringProperty()
    def execute(self, context):
        if context.active_object is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        obj = context.active_object
        if obj.mode != 'EDIT': bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        layer = bm.verts.layers.int.get(self.attr_target)
        if layer:
            for v in bm.verts: v.select = (v[layer] != -1)
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}

class MESH_OT_PickColor(bpy.types.Operator):
    bl_idname = "mesh.pick_color_selected"
    bl_label = "Sample"
    bl_description = "Step 2: Averages color from selected vertices and sets as reference color"
    def execute(self, context):
        if context.active_object is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        obj = context.active_object
        if obj.mode != 'EDIT': bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        sel = [v.index for v in bm.verts if v.select]
        if not sel:
            self.report({'WARNING'}, "No vertices selected")
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='OBJECT')
        color_attr = obj.data.attributes.get(context.scene.pc_color_attr_enum)
        if not color_attr:
            self.report({'ERROR'}, f"Color attribute '{context.scene.pc_color_attr_enum}' not found")
            return {'CANCELLED'}
        avg = Vector((0,0,0))
        for i in sel: avg += Vector(color_attr.data[i].color[:3])
        context.scene.pc_ref_color = (avg.x/len(sel), avg.y/len(sel), avg.z/len(sel), 1.0)
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}

class MESH_OT_Heatmap(bpy.types.Operator):
    bl_idname = "mesh.pc_heatmap"
    bl_label = "Heatmap"
    bl_description = "Creates a color attribute showing matches in green and non-matches in blue (requires attribute display)"
    def execute(self, context):
        if context.active_object is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        s, obj = context.scene, context.active_object
        bpy.ops.object.mode_set(mode='OBJECT')
        color_attr = obj.data.attributes.get(s.pc_color_attr_enum)
        if not color_attr:
            self.report({'ERROR'}, f"Color attribute '{s.pc_color_attr_enum}' not found")
            return {'CANCELLED'}
        hm_attr = obj.data.attributes.get("pc_heatmap") or obj.data.attributes.new(name="pc_heatmap", type='FLOAT_COLOR', domain='POINT')
        ref_hsv = Color((s.pc_ref_color[0], s.pc_ref_color[1], s.pc_ref_color[2])).hsv
        for i, item in enumerate(color_attr.data):
            match = 1.0 if check_color_match(item.color[:3], ref_hsv, s.pc_color_tol, s.pc_material_mode) else 0.0
            hm_attr.data[i].color = (match, 0.0, 1.0 - match, 1.0)
        obj.data.update()
        self.report({'INFO'}, "Heatmap created: 'pc_heatmap'. Switch viewport to attribute display mode to visualize.")
        return {'FINISHED'}

class MESH_OT_AnalyzeColors(bpy.types.Operator):
    bl_idname = "mesh.pc_analyze_colors"
    bl_label = "Analyze Colors"
    bl_description = "Creates a numeric attribute (0-1) showing how well each point matches the reference color"
    def execute(self, context):
        if context.active_object is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        scene, obj = context.scene, context.active_object
        bpy.ops.object.mode_set(mode='OBJECT')
        color_attr = obj.data.attributes.get(scene.pc_color_attr_enum)
        if not color_attr:
            self.report({'ERROR'}, f"Color attribute '{scene.pc_color_attr_enum}' not found")
            return {'CANCELLED'}
        score_attr = obj.data.attributes.get("pc_color_score") or obj.data.attributes.new(name="pc_color_score", type='FLOAT', domain='POINT')
        ref_hsv = Color((scene.pc_ref_color[0], scene.pc_ref_color[1], scene.pc_ref_color[2])).hsv
        ref_h, ref_s, ref_v = ref_hsv
        tol = scene.pc_color_tol
        mat_mode = scene.pc_material_mode
        print(f"DEBUG: tolerance={tol}, material_mode={mat_mode}")
        for i, item in enumerate(color_attr.data):
            c = Color((item.color[0], item.color[1], item.color[2]))
            h, sat, v = c.hsv
            h_diff = abs(h - ref_h)
            if h_diff > 0.5: h_diff = 1.0 - h_diff
            s_diff = abs(sat - ref_s)
            v_diff = abs(v - ref_v)
            h_score = max(0.0, 1.0 - (h_diff / (tol + 0.001)))
            s_score = max(0.0, 1.0 - (s_diff / (tol + 0.001)))
            v_score = max(0.0, 1.0 - (v_diff / (tol + 0.001))) if not mat_mode else 1.0
            score_attr.data[i].value = (h_score * s_score * v_score)
        obj.data.update()
        self.report({'INFO'}, "Color analysis complete: 'pc_color_score' attribute created (0=no match, 1=perfect match)")
        return {'FINISHED'}

class MESH_OT_DeleteNoise(bpy.types.Operator):
    bl_idname = "mesh.pc_delete_noise"
    bl_label = "Trash"
    bl_description = "Step 5d: Deletes currently selected vertices"
    attr_target: bpy.props.StringProperty()
    def execute(self, context):
        if context.active_object is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        obj = context.active_object
        if obj.mode != 'EDIT': bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.delete(type='VERT')
        return {'FINISHED'}

class MESH_OT_SplitClusters(bpy.types.Operator):
    bl_idname = "mesh.pc_split_clusters"
    bl_label = "Split"
    bl_description = "Step 5c: Extracts selected vertices to a new object"
    attr_target: bpy.props.StringProperty()
    def execute(self, context):
        if context.active_object is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        obj = context.active_object
        if obj.mode != 'EDIT': bpy.ops.object.mode_set(mode='EDIT')

        # Get selected vertices
        bm = bmesh.from_edit_mesh(obj.data)
        selected_verts = [v for v in bm.verts if v.select]
        if not selected_verts:
            self.report({'WARNING'}, "No vertices selected")
            return {'CANCELLED'}

        # Switch to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create new mesh with selected vertices
        new_mesh = bpy.data.meshes.new(name=f"{obj.name}_split")
        new_obj = bpy.data.objects.new(name=f"{obj.name}_split", object_data=new_mesh)
        context.collection.objects.link(new_obj)

        # Copy selected vertices to new mesh
        bm_new = bmesh.new()
        vert_map = {}
        for v in selected_verts:
            vert_map[v.index] = bm_new.verts.new(v.co)

        # Copy faces that are entirely in selected vertices
        for face in bm.faces:
            if all(v.select for v in face.verts):
                bm_new.faces.new([vert_map[v.index] for v in face.verts])

        bm_new.to_mesh(new_mesh)
        bm_new.free()
        new_mesh.update()

        # Delete selected from original
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.delete(type='VERT')
        bpy.ops.object.mode_set(mode='OBJECT')

        self.report({'INFO'}, f"Split {len(selected_verts)} vertices to new object")
        return {'FINISHED'}

class MESH_OT_ClusterDist(bpy.types.Operator):
    bl_idname = "mesh.cluster_dist"
    bl_label = "Spatial Cluster"
    bl_description = "Step 2: Clusters vertices by spatial proximity within the radius, groups minimum points together"
    def execute(self, context):
        if context.active_object is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        run_dist_clustering(context.active_object, context.scene.pc_radius, context.scene.pc_min_points)
        return {'FINISHED'}

class MESH_OT_ClusterColor(bpy.types.Operator):
    bl_idname = "mesh.cluster_color"
    bl_label = "Cluster by Color"
    bl_description = "Step 4b: Groups vertices matching the reference color within tolerance (respects material mode)"
    def execute(self, context):
        if context.active_object is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        s = context.scene
        run_global_color_clustering(context.active_object, s.pc_color_tol, s.pc_material_mode, s.pc_color_attr_enum, s.pc_ref_color)
        return {'FINISHED'}

class MESH_OT_PreviewColorMatches(bpy.types.Operator):
    bl_idname = "mesh.preview_color_matches"
    bl_label = "Preview Matches"
    bl_description = "Step 4a: Counts and reports how many vertices match the reference color with current settings"
    def execute(self, context):
        if context.active_object is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        s = context.scene
        obj = context.active_object
        bpy.ops.object.mode_set(mode='OBJECT')
        color_attr = obj.data.attributes.get(s.pc_color_attr_enum)
        if not color_attr:
            self.report({'ERROR'}, f"Color attribute '{s.pc_color_attr_enum}' not found")
            return {'CANCELLED'}
        ref_hsv = Color((s.pc_ref_color[0], s.pc_ref_color[1], s.pc_ref_color[2])).hsv
        count = 0
        for item in color_attr.data:
            if check_color_match(item.color[:3], ref_hsv, s.pc_color_tol, s.pc_material_mode):
                count += 1
        self.report({'INFO'}, f"Found {count} matching vertices")
        return {'FINISHED'}

class PC_BooleanBase:
    """Shared behaviour for the two Boolean Logic buttons.

    invoke() runs the work in a modal timer so the panel can show live
    progress. execute() runs the same job to completion in one blocking call,
    which is what happens when the operator is driven from a script or in
    background mode, where there is no window to run modal on.
    """

    operation = 'INTERSECT'
    _timer = None
    _job = None

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return (scene.pc_bool_container is not None
                and scene.pc_bool_cloud is not None
                and not scene.pc_bool_running)

    def _announce(self, context, job, message):
        scene = context.scene
        scene.pc_bool_status = message
        notes = []
        if job.open_edges:
            notes.append(f"'{job.container.name}' is not watertight "
                         f"({job.open_edges} boundary edges)")
        if job.contested:
            notes.append(f"{job.contested} points needed a tie-break")
        if job.dropped:
            notes.append("dropped unsupported attributes: "
                         + ", ".join(job.dropped))
        self.report({'INFO'}, message + (". " + "; ".join(notes) if notes else ""))

    def _cleanup(self, context):
        scene = context.scene
        scene.pc_bool_running = False
        scene.pc_bool_progress = 0.0
        scene.pc_bool_rate = 0.0
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._job = None

    def execute(self, context):
        """Blocking path: used from scripts and in background mode."""
        scene = context.scene
        try:
            job = BooleanJob(context, self.operation)
            while not job.step(1.0):
                pass
            removed, kept, message = job.finish()
        except ValueError as exc:
            scene.pc_bool_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self._announce(context, job, message)
        return {'FINISHED'}

    def invoke(self, context, event):
        """Modal path: used when the button is clicked in the sidebar."""
        if getattr(context, "window", None) is None:
            return self.execute(context)

        scene = context.scene
        try:
            self._job = BooleanJob(context, self.operation)
        except ValueError as exc:
            scene.pc_bool_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.pc_bool_running = True
        scene.pc_bool_progress = 0.0
        scene.pc_bool_rate = 0.0
        scene.pc_bool_status = ""
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        if event.type == 'ESC':
            self._cleanup(context)
            scene.pc_bool_status = "Cancelled"
            tag_redraw_sidebar(context)
            self.report({'WARNING'}, "Boolean cancelled, nothing was deleted")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        done = self._job.step(0.08)
        scene.pc_bool_progress = self._job.progress
        scene.pc_bool_rate = self._job.rate
        tag_redraw_sidebar(context)

        if not done:
            return {'RUNNING_MODAL'}

        job = self._job
        try:
            removed, kept, message = job.finish()
        except ValueError as exc:
            self._cleanup(context)
            scene.pc_bool_status = ""
            tag_redraw_sidebar(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        self._cleanup(context)
        self._announce(context, job, message)
        tag_redraw_sidebar(context)
        return {'FINISHED'}

class MESH_OT_BooleanIntersect(PC_BooleanBase, bpy.types.Operator):
    bl_idname = "mesh.pc_boolean_intersect"
    bl_label = "Intersect"
    bl_description = (
        "Step 3a: Boolean intersection. Keeps only the points of the point "
        "cloud that lie INSIDE the closed mesh, and deletes everything "
        "outside it.\n\n"
        "Both objects are picked in step 1. Only the point cloud is modified, "
        "the closed mesh is used as the container and left untouched. Vertex "
        "colors and every other point attribute are carried over to the "
        "result. Press Esc while it runs to cancel"
    )
    bl_options = {'REGISTER', 'UNDO'}
    operation = 'INTERSECT'

class MESH_OT_BooleanDifference(PC_BooleanBase, bpy.types.Operator):
    bl_idname = "mesh.pc_boolean_difference"
    bl_label = "Difference"
    bl_description = (
        "Step 3b: Boolean difference. Keeps only the points of the point "
        "cloud that lie OUTSIDE the closed mesh, and deletes everything "
        "inside it.\n\n"
        "Both objects are picked in step 1. Only the point cloud is modified, "
        "the closed mesh is used as the container and left untouched. Vertex "
        "colors and every other point attribute are carried over to the "
        "result. Press Esc while it runs to cancel"
    )
    bl_options = {'REGISTER', 'UNDO'}
    operation = 'DIFFERENCE'

# --- 3. UI SYNC & PANEL ---

def get_attributes_enum(self, context):
    items = []
    if context.active_object and context.active_object.type == 'MESH':
        for attr in context.active_object.data.attributes:
            if attr.domain == 'POINT': items.append((attr.name, attr.name, ""))
    return items if items else [("NONE", "None", "")]

def poll_bool_container(self, obj):
    """Only meshes with faces can act as a closed container."""
    return (obj.type == 'MESH' and len(obj.data.polygons) > 0
            and obj != self.pc_bool_cloud)

def poll_bool_cloud(self, obj):
    """Any mesh can be the point cloud, but not the container itself."""
    return obj.type == 'MESH' and obj != self.pc_bool_container

def sync_video_resolution(scene):
    """Pull the output height back onto the aspect ratio of the video."""
    if not scene.pc_vid_lock_aspect:
        return
    if scene.pc_vid_src_w <= 0 or scene.pc_vid_src_h <= 0:
        return
    height = round(scene.pc_vid_width * scene.pc_vid_src_h
                   / scene.pc_vid_src_w)
    scene.pc_vid_height = max(int(height), 1)


def update_video_path(self, context):
    """Read the new file and fill in everything that follows from it.

    Only pc_vid_height is written back from another property, and it has no
    update of its own, so this cannot bounce.
    """
    width, height, frames = probe_video(self.pc_vid_path)
    self.pc_vid_src_w = width
    self.pc_vid_src_h = height
    self.pc_vid_src_frames = frames
    if frames:
        self.pc_vid_start = 1
        self.pc_vid_end = frames
    sync_video_resolution(self)


def update_video_resolution(self, context):
    sync_video_resolution(self)


class MESH_PT_SelectPanel(bpy.types.Panel):
    bl_label = "Select"
    bl_idname = "MESH_PT_SelectPanel"
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'PC Tools'

    def draw(self, context):
        pass


class MESH_PT_ModifyPanel(bpy.types.Panel):
    bl_label = "Modify"
    bl_idname = "MESH_PT_ModifyPanel"
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'PC Tools'

    def draw(self, context):
        pass


class PC_ToolPanel:
    """Shared setup for the four tool sections.

    Each one is a sub panel of the suite, so Blender gives it its own
    collapse arrow and remembers whether it was left open. They all start
    closed, which keeps the sidebar short now that there are four of them.
    """

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'PC Tools'
    bl_options = {'DEFAULT_CLOSED'}
    icon = 'NONE'

    def draw_header(self, context):
        self.layout.label(text="", icon=self.icon)

    def step(self, layout, mark="  "):
        """One numbered row, matching the step markers used before."""
        row = layout.row(align=True)
        row.scale_x = 0.12
        row.label(text=mark)
        row.scale_x = 1.0
        return row

    def note(self, layout, text):
        if text:
            col = layout.column(align=True)
            col.scale_y = 0.55
            col.label(text=text)


class MESH_PT_GeneratePanel(bpy.types.Panel):
    bl_label = "Generate"
    bl_idname = "MESH_PT_GeneratePanel"
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'PC Tools'

    def draw(self, context):
        pass


class MESH_PT_VideoStackPanel(PC_ToolPanel, bpy.types.Panel):
    bl_label = "Video Stack"
    bl_parent_id = "MESH_PT_GeneratePanel"
    bl_idname = "MESH_PT_VideoStackPanel"
    icon = 'SEQUENCE'

    def draw(self, context):
        layout, scene = self.layout, context.scene

        # Step 1: Pick the video, and say what was found in it
        self.step(layout, "①").prop(scene, "pc_vid_path", text="")
        if scene.pc_vid_src_frames:
            self.note(layout, f"{scene.pc_vid_src_w} x "
                              f"{scene.pc_vid_src_h}, "
                              f"{scene.pc_vid_src_frames} frames")
        elif scene.pc_vid_src_w:
            self.note(layout, f"{scene.pc_vid_src_w} x {scene.pc_vid_src_h}, "
                              f"length unknown, set End by hand")
        elif scene.pc_vid_path:
            self.note(layout, "Could not read that file")

        # Step 2: Output resolution, one point per pixel
        self.step(layout, "②").prop(scene, "pc_vid_width")
        row = self.step(layout)
        row.prop(scene, "pc_vid_height")
        row.enabled = not scene.pc_vid_lock_aspect
        self.step(layout).prop(scene, "pc_vid_lock_aspect")

        # Step 3: Which frames to take
        row = self.step(layout, "③")
        row.prop(scene, "pc_vid_start")
        row.prop(scene, "pc_vid_end")
        self.step(layout).prop(scene, "pc_vid_step")

        # Step 4: How the stack is laid out, and what to leave out
        self.step(layout, "④").prop(scene, "pc_vid_pixel")
        self.step(layout).prop(scene, "pc_vid_z")
        self.step(layout).prop(scene, "pc_vid_cull")

        # What that adds up to, before pressing anything
        frames, points = video_estimate(scene)
        if frames:
            col = layout.column(align=True)
            col.scale_y = 0.55
            col.alert = points > 20_000_000
            col.label(text=f"{frames} frames x "
                           f"{scene.pc_vid_width * scene.pc_vid_height:,}"
                           f" = {points:,} points")

        # Step 5: Build it
        self.step(layout, "⑤").operator("mesh.pc_video_stack",
                                        text="Generate", icon='PLAY')

        if scene.pc_vid_running:
            self.note(layout, f"{scene.pc_vid_stage}  "
                              f"{scene.pc_vid_progress * 100:.0f}%")
        else:
            self.note(layout, scene.pc_vid_status)


class MESH_PT_SpatialPanel(PC_ToolPanel, bpy.types.Panel):
    bl_label = "Spatial"
    bl_parent_id = "MESH_PT_SelectPanel"
    bl_idname = "MESH_PT_SpatialPanel"
    icon = 'PHYSICS'

    def draw(self, context):
        layout, scene = self.layout, context.scene

        self.step(layout, "①").prop(scene, "pc_radius")
        self.step(layout).prop(scene, "pc_min_points")
        self.step(layout, "②").operator("mesh.cluster_dist", text="Cluster",
                                             icon='PLAY')

        row = self.step(layout, "③")
        row.operator("mesh.pc_select_noise", text="Select").attr_target = "cluster_id_dist"
        row.operator("mesh.pc_invert_selection", text="", icon='UV_SYNC_SELECT')
        row.operator("mesh.pc_split_clusters", text="Split").attr_target = "cluster_id_dist"
        row.operator("mesh.pc_delete_noise", text="Trash").attr_target = "cluster_id_dist"


class MESH_PT_ThicknessPanel(PC_ToolPanel, bpy.types.Panel):
    bl_label = "Thickness"
    bl_parent_id = "MESH_PT_SelectPanel"
    bl_idname = "MESH_PT_ThicknessPanel"
    icon = 'MOD_SOLIDIFY'

    def draw(self, context):
        layout, scene = self.layout, context.scene

        self.step(layout, "①").prop(scene, "pc_thin_max")
        self.step(layout).prop(scene, "pc_thin_voxel")
        self.step(layout).prop(scene, "pc_thin_close")
        self.step(layout, "②").operator("mesh.pc_thickness", text="Analyze",
                                             icon='PLAY')
        self.step(layout, "③").prop(scene, "pc_thin_threshold")

        row = self.step(layout)
        row.operator("mesh.pc_thin_heatmap", icon='FCURVE', text="")
        row.operator("mesh.pc_select_thin", text="Select")
        row.operator("mesh.pc_invert_selection", text="", icon='UV_SYNC_SELECT')

        if scene.pc_thin_running:
            self.note(layout, f"{scene.pc_thin_stage}  "
                              f"{scene.pc_thin_progress * 100:.0f}%")
        else:
            self.note(layout, scene.pc_thin_status)


class MESH_PT_GrowthPanel(PC_ToolPanel, bpy.types.Panel):
    bl_label = "Growth"
    bl_parent_id = "MESH_PT_ModifyPanel"
    bl_idname = "MESH_PT_GrowthPanel"
    icon = 'MOD_ARRAY'

    def draw(self, context):
        layout, scene = self.layout, context.scene

        self.step(layout, "①").operator("mesh.pc_analyze_lattice",
                                             text="Analyze Lattice", icon='PLAY')
        col = layout.column(align=True)
        col.prop(scene, "pc_grid_pitch")

        row = self.step(layout, "②")
        row.prop(scene, "pc_grow_mode", text="")
        self.step(layout).prop(scene, "pc_grow_steps")
        self.step(layout, "③").operator("mesh.pc_grow", text="Grow",
                                             icon='ADD')

        if scene.pc_grow_running:
            self.note(layout, f"{scene.pc_grow_stage}  "
                              f"{scene.pc_grow_progress * 100:.0f}%")
        elif scene.pc_grow_status:
            self.note(layout, scene.pc_grow_status)

        if scene.pc_grid_running:
            self.note(layout, f"{scene.pc_grid_stage}  "
                              f"{scene.pc_grid_progress * 100:.0f}%")
        else:
            self.note(layout, scene.pc_grid_status)


class MESH_PT_SpectrumPanel(PC_ToolPanel, bpy.types.Panel):
    bl_label = "Spectrum"
    bl_parent_id = "MESH_PT_SelectPanel"
    bl_idname = "MESH_PT_SpectrumPanel"
    icon = 'COLOR'

    def draw(self, context):
        layout, scene = self.layout, context.scene

        self.step(layout, "①").prop(scene, "pc_color_attr_enum", text="")

        row = self.step(layout, "②")
        row.prop(scene, "pc_ref_color", text="")
        row.operator("mesh.pick_color_selected", icon='EYEDROPPER', text="")

        self.step(layout, "③").prop(scene, "pc_color_tol")
        self.step(layout).prop(scene, "pc_material_mode")

        row = self.step(layout, "④")
        row.operator("mesh.preview_color_matches", icon='ZOOM_ALL', text="")
        row.operator("mesh.pc_analyze_colors", icon='FCURVE', text="Analyze")
        row.operator("mesh.cluster_color", text="Cluster", icon='IMAGE_RGB_ALPHA')

        row = self.step(layout, "⑤")
        row.operator("mesh.pc_select_cluster", text="Select").attr_target = "cluster_id_color"
        row.operator("mesh.pc_invert_selection", text="", icon='UV_SYNC_SELECT')
        row.operator("mesh.pc_split_clusters", text="Split").attr_target = "cluster_id_color"
        row.operator("mesh.pc_delete_noise", text="Trash").attr_target = "cluster_id_color"


class MESH_PT_BooleanPanel(PC_ToolPanel, bpy.types.Panel):
    bl_label = "Boolean"
    bl_parent_id = "MESH_PT_ModifyPanel"
    bl_idname = "MESH_PT_BooleanPanel"
    icon = 'MOD_BOOLEAN'

    def draw(self, context):
        layout, scene = self.layout, context.scene

        self.step(layout, "①").prop(scene, "pc_bool_container")
        self.step(layout).prop(scene, "pc_bool_cloud")
        self.step(layout, "②").prop(scene, "pc_bool_samples", text="")

        row = self.step(layout, "③")
        row.operator("mesh.pc_boolean_intersect", text="Intersect",
                     icon='SELECT_INTERSECT')
        row.operator("mesh.pc_boolean_difference", text="Difference",
                     icon='SELECT_DIFFERENCE')

        if scene.pc_bool_running:
            self.note(layout, f"{scene.pc_bool_progress * 100:.1f}%    "
                              f"{scene.pc_bool_rate:,.0f} pts/s")
        else:
            self.note(layout, scene.pc_bool_status)


# --- 4. REGISTRATION ---

classes = (MESH_OT_InvertSelection, MESH_OT_PickColor, MESH_OT_Heatmap, MESH_OT_AnalyzeColors, MESH_OT_SelectNoise,
           MESH_OT_SelectCluster, MESH_OT_DeleteNoise, MESH_OT_SplitClusters,
           MESH_OT_ClusterDist, MESH_OT_ClusterColor, MESH_OT_PreviewColorMatches,
           MESH_OT_BooleanIntersect, MESH_OT_BooleanDifference,
           MESH_OT_Thickness, MESH_OT_SelectThin, MESH_OT_ThinHeatmap,
           MESH_OT_AnalyzeLattice, MESH_OT_Grow, MESH_OT_VideoStack,
           MESH_PT_GeneratePanel, MESH_PT_VideoStackPanel,
           MESH_PT_SelectPanel, MESH_PT_ModifyPanel, MESH_PT_SpatialPanel,
           MESH_PT_ThicknessPanel, MESH_PT_GrowthPanel,
           MESH_PT_SpectrumPanel,
           MESH_PT_BooleanPanel)

def register():
    bpy.types.Scene.pc_vid_path = bpy.props.StringProperty(
        name="Video", subtype='FILE_PATH', default="",
        update=update_video_path,
        description="Step 1: The video to turn into points. Its size and "
                    "length are read as soon as it is picked, and the frame "
                    "range below is set to the whole clip")
    bpy.types.Scene.pc_vid_src_w = bpy.props.IntProperty(
        name="Source Width", default=0,
        description="Width of the picked video in pixels")
    bpy.types.Scene.pc_vid_src_h = bpy.props.IntProperty(
        name="Source Height", default=0,
        description="Height of the picked video in pixels")
    bpy.types.Scene.pc_vid_src_frames = bpy.props.IntProperty(
        name="Source Frames", default=0,
        description="Length of the picked video in frames, 0 if unknown")
    bpy.types.Scene.pc_vid_width = bpy.props.IntProperty(
        name="Width", default=256, min=1, max=16384,
        update=update_video_resolution,
        description="Step 2a: Points across each frame. The video is scaled "
                    "to this before it is read, so this is what decides the "
                    "point count, not the resolution of the file")
    bpy.types.Scene.pc_vid_height = bpy.props.IntProperty(
        name="Height", default=144, min=1, max=16384,
        description="Step 2b: Points down each frame. Follows the aspect "
                    "ratio of the video while Lock Aspect is on")
    bpy.types.Scene.pc_vid_lock_aspect = bpy.props.BoolProperty(
        name="Lock Aspect", default=True,
        update=update_video_resolution,
        description="Step 2c: Keep the height on the aspect ratio of the "
                    "video. Turn off to stretch or squash the frames")
    bpy.types.Scene.pc_vid_start = bpy.props.IntProperty(
        name="Start", default=1, min=1,
        description="Step 3a: First video frame to take, counting from 1")
    bpy.types.Scene.pc_vid_end = bpy.props.IntProperty(
        name="End", default=0, min=0,
        description="Step 3b: Last video frame to take. 0 means run to the "
                    "end of the clip")
    bpy.types.Scene.pc_vid_step = bpy.props.IntProperty(
        name="Frame Step", default=1, min=1,
        description="Step 3c: Take every nth frame. 2 halves the point "
                    "count, 5 cuts it to a fifth. The stack stays evenly "
                    "spaced whatever this is set to")
    bpy.types.Scene.pc_vid_pixel = bpy.props.FloatProperty(
        name="Pixel Size", default=0.01, min=0.0001, precision=4,
        description="Step 4a: Distance between neighbouring points inside "
                    "one frame. Each frame is centred on the origin, so a "
                    "256 wide frame at 0.01 comes out 2.56 units across")
    bpy.types.Scene.pc_vid_z = bpy.props.FloatProperty(
        name="Z Distance", default=0.1, precision=4,
        description="Step 4b: Distance between one frame and the next along "
                    "Z. Negative stacks the clip downwards")
    bpy.types.Scene.pc_vid_cull = bpy.props.FloatProperty(
        name="Skip Below", default=0.0, min=0.0, max=1.0,
        description="Step 4c: Drop pixels darker than this instead of making "
                    "a point for them. At 0 every single pixel becomes a "
                    "point, which is the exact Width x Height grid. Raise it "
                    "to throw away a black background and keep the point "
                    "count down")
    bpy.types.Scene.pc_vid_progress = bpy.props.FloatProperty(
        name="Progress", default=0.0, min=0.0, max=1.0,
        description="Fraction of the work already done")
    bpy.types.Scene.pc_vid_stage = bpy.props.StringProperty(
        name="Stage", default="",
        description="What the video stack is busy with right now")
    bpy.types.Scene.pc_vid_status = bpy.props.StringProperty(
        name="Status", default="",
        description="Result of the last video stack run")
    bpy.types.Scene.pc_vid_running = bpy.props.BoolProperty(
        name="Running", default=False,
        description="True while a video is being turned into points")
    bpy.types.Scene.pc_radius = bpy.props.FloatProperty(name="Radius", default=0.05, precision=4, description="Step 1a: Distance threshold for grouping nearby vertices together")
    bpy.types.Scene.pc_min_points = bpy.props.IntProperty(name="Min Points", default=10, description="Step 1b: Minimum vertices required to form a cluster (smaller values = more clusters)")
    bpy.types.Scene.pc_color_attr_enum = bpy.props.EnumProperty(name="Attribute", items=get_attributes_enum, description="Step 1: Select which color attribute to analyze")
    bpy.types.Scene.pc_color_tol = bpy.props.FloatProperty(name="Color Tolerance", default=0.1, min=0, max=1, description="Step 3a: How similar colors must be to match (0=exact, 1=any color)")
    bpy.types.Scene.pc_material_mode = bpy.props.BoolProperty(name="Material Mode (Ignore Shadows)", default=False, description="Step 3b: When on, ignores brightness differences - matches same material under different lighting")
    bpy.types.Scene.pc_ref_color = bpy.props.FloatVectorProperty(name="Ref Color", subtype='COLOR', size=4, default=(0.5,0.5,0.5,1.0), description="Step 2: Reference color to match against (use Sample button or picker)")
    bpy.types.Scene.pc_bool_container = bpy.props.PointerProperty(
        name="Closed Mesh", type=bpy.types.Object, poll=poll_bool_container,
        description="Step 1a: The closed mesh used as the boolean container. "
                    "Only meshes that have faces are listed. This object is "
                    "read only, it is never modified")
    bpy.types.Scene.pc_bool_cloud = bpy.props.PointerProperty(
        name="Point Cloud", type=bpy.types.Object, poll=poll_bool_cloud,
        description="Step 1b: The point cloud that gets cut. This is the "
                    "object that is modified, so keep a copy if you want to "
                    "go back to the full cloud later")
    bpy.types.Scene.pc_bool_samples = bpy.props.EnumProperty(
        name="Accuracy",
        items=[
            ('1', "Fast (1 ray)",
             "One ray per point. Fastest, but a ray that clips an edge or "
             "slips through a hole in the container miscounts, which leaves "
             "isolated stray points behind"),
            ('3', "Balanced (3 rays)",
             "Three ray directions vote on each point. Clears virtually all "
             "stray points for roughly 40% more time. Recommended"),
            ('5', "Thorough (5 rays)",
             "Five ray directions vote on each point. For containers with a "
             "lot of coplanar, degenerate or non-watertight geometry"),
        ],
        default='3',
        description="Step 2: How many ray directions vote on each point when "
                    "deciding inside from outside")
    bpy.types.Scene.pc_bool_progress = bpy.props.FloatProperty(
        name="Progress", default=0.0, min=0.0, max=1.0,
        description="Fraction of the tested points already classified")
    bpy.types.Scene.pc_bool_rate = bpy.props.FloatProperty(
        name="Rate", default=0.0,
        description="Points classified per second")
    bpy.types.Scene.pc_bool_status = bpy.props.StringProperty(
        name="Status", default="",
        description="Result of the last boolean operation")
    bpy.types.Scene.pc_bool_running = bpy.props.BoolProperty(
        name="Running", default=False,
        description="True while a boolean operation is in progress")
    bpy.types.Scene.pc_thin_max = bpy.props.FloatProperty(
        name="Measure Up To", default=0.5, min=0.0001, precision=4,
        subtype='DISTANCE',
        description="Step 1a: Largest thickness worth measuring. Anything "
                    "thicker than this is simply reported as thick, so keep "
                    "it just above the thickest part you would still call "
                    "thin. It sets how many erosion passes run, so raising "
                    "it costs time")
    bpy.types.Scene.pc_thin_voxel = bpy.props.FloatProperty(
        name="Voxel Size", default=0.0, min=0.0, precision=4,
        subtype='DISTANCE',
        description="Step 1b: Resolution of the measurement. Leave at 0 to "
                    "let Analyze pick roughly 3.5 times the point spacing, "
                    "which it then writes back here. Smaller resolves finer "
                    "detail but reads the gaps between sparse points as "
                    "holes; larger is faster and safer but blurs thickness "
                    "in steps of twice this value")
    bpy.types.Scene.pc_thin_close = bpy.props.BoolProperty(
        name="Fill Gaps", default=True,
        description="Step 1c: Plugs single voxel holes left inside the cloud "
                    "by uneven sampling before measuring. Without it those "
                    "holes look like surfaces and the parts around them are "
                    "wrongly reported as thin. Costs a little time and does "
                    "not change the thickness of solid parts")
    bpy.types.Scene.pc_thin_threshold = bpy.props.FloatProperty(
        name="Thin Below", default=0.1, min=0.0, precision=4,
        subtype='DISTANCE',
        description="Step 3: Points in material thinner than this count as "
                    "thin. Free to change after Analyze, Select and the "
                    "heatmap reuse the measurement instead of redoing it")
    bpy.types.Scene.pc_thin_progress = bpy.props.FloatProperty(
        name="Progress", default=0.0, min=0.0, max=1.0,
        description="How far the running thickness measurement has got")
    bpy.types.Scene.pc_thin_stage = bpy.props.StringProperty(
        name="Stage", default="",
        description="Which part of the thickness measurement is running")
    bpy.types.Scene.pc_thin_status = bpy.props.StringProperty(
        name="Status", default="",
        description="Result of the last thickness measurement")
    bpy.types.Scene.pc_thin_running = bpy.props.BoolProperty(
        name="Running", default=False,
        description="True while a thickness measurement is in progress")
    bpy.types.Scene.pc_grid_pitch = bpy.props.FloatVectorProperty(
        name="Pitch", size=3, default=(0.0, 0.0, 0.0), precision=4,
        subtype='XYZ',
        description="Step 1b: Distance between neighbouring points along each "
                    "of the three grid directions. Filled in by Analyze "
                    "Lattice and free to correct by hand afterwards. Editing "
                    "a value keeps the direction it was found at and only "
                    "changes how far apart the steps are")
    bpy.types.Scene.pc_grid_basis = bpy.props.FloatVectorProperty(
        name="Grid Basis", size=9, default=(0.0,) * 9,
        description="The three grid step vectors found by Analyze Lattice, "
                    "which carry the orientation of the grid")
    bpy.types.Scene.pc_grid_origin = bpy.props.FloatVectorProperty(
        name="Grid Origin", size=3, default=(0.0, 0.0, 0.0),
        description="A point the fitted grid passes through")
    bpy.types.Scene.pc_grid_fit = bpy.props.FloatProperty(
        name="Fit", default=0.0, min=0.0, max=1.0,
        description="Share of points sitting within a quarter cell of the "
                    "fitted grid")
    bpy.types.Scene.pc_grid_progress = bpy.props.FloatProperty(
        name="Progress", default=0.0, min=0.0, max=1.0,
        description="How far the running lattice analysis has got")
    bpy.types.Scene.pc_grid_stage = bpy.props.StringProperty(
        name="Stage", default="",
        description="Which part of the lattice analysis is running")
    bpy.types.Scene.pc_grid_status = bpy.props.StringProperty(
        name="Status", default="",
        description="Result of the last lattice analysis")
    bpy.types.Scene.pc_grid_running = bpy.props.BoolProperty(
        name="Running", default=False,
        description="True while a lattice analysis is in progress")
    bpy.types.Scene.pc_grow_mode = bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('UNIFORM', "Uniform",
             "Grow one grid step in every direction. Makes a part both "
             "thicker and wider"),
            ('THICKEN', "Thicken",
             "Grow only across the thin direction, so a sheet gets thicker "
             "and a strut gets fatter without spreading sideways"),
            ('EXTEND', "Extend",
             "Grow only along the part, so a strut grows out at its tip and "
             "a sheet grows out at its rim without getting thicker. Use this "
             "to carry a thin part further until it meets another one"),
        ],
        default='UNIFORM',
        description="Step 2a: Which way new points are allowed to go")
    bpy.types.Scene.pc_grow_steps = bpy.props.IntProperty(
        name="Steps", default=1, min=1, max=50,
        description="Step 2b: How many grid steps to grow in one press. Each "
                    "step starts from the points the step before it added, "
                    "so ten steps carry a part ten cells further")
    bpy.types.Scene.pc_grow_progress = bpy.props.FloatProperty(
        name="Progress", default=0.0, min=0.0, max=1.0,
        description="How far the running growth has got")
    bpy.types.Scene.pc_grow_stage = bpy.props.StringProperty(
        name="Stage", default="",
        description="Which part of the growth is running")
    bpy.types.Scene.pc_grow_status = bpy.props.StringProperty(
        name="Status", default="",
        description="Result of the last growth")
    bpy.types.Scene.pc_grow_running = bpy.props.BoolProperty(
        name="Running", default=False,
        description="True while growth is in progress")
    for cls in classes: bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    props = ['pc_radius', 'pc_min_points', 'pc_color_attr_enum', 'pc_color_tol', 'pc_material_mode',
             'pc_ref_color', 'pc_bool_container', 'pc_bool_cloud',
             'pc_bool_samples', 'pc_bool_progress', 'pc_bool_rate',
             'pc_bool_status', 'pc_bool_running',
             'pc_thin_max', 'pc_thin_voxel', 'pc_thin_close',
             'pc_thin_threshold', 'pc_thin_progress', 'pc_thin_stage',
             'pc_thin_status', 'pc_thin_running',
             'pc_grid_pitch', 'pc_grid_basis', 'pc_grid_origin',
             'pc_grid_fit', 'pc_grid_progress', 'pc_grid_stage',
             'pc_grid_status', 'pc_grid_running',
             'pc_grow_mode', 'pc_grow_steps', 'pc_grow_progress',
             'pc_grow_stage', 'pc_grow_status', 'pc_grow_running',
             'pc_vid_path', 'pc_vid_src_w', 'pc_vid_src_h',
             'pc_vid_src_frames', 'pc_vid_width', 'pc_vid_height',
             'pc_vid_lock_aspect', 'pc_vid_start', 'pc_vid_end',
             'pc_vid_step', 'pc_vid_pixel', 'pc_vid_z', 'pc_vid_cull',
             'pc_vid_progress', 'pc_vid_stage', 'pc_vid_status',
             'pc_vid_running']
    for prop in props:
        if hasattr(bpy.types.Scene, prop): delattr(bpy.types.Scene, prop)

if __name__ == "__main__":
    register()
