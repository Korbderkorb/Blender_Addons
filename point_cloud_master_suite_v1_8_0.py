bl_info = {
    "name": "Point Cloud Master Suite",
    "author": "Korbinian Enzinger",
    "version": (1, 8, 0),
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
import zlib
from collections import deque

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


def ray_crossings(bvh, start, heading, unit, length, first, eps, max_hits):
    """Every crossing of a surface along one ray, in the order they are met.

    Returned as positions along the ray in steps of `length`, counted from
    `first` at the start, and whether the walk ran into max_hits, which a
    surface that is not closed can do.
    """
    found = []
    point = Vector(start)
    for _ in range(max_hits):
        hit = bvh.ray_cast(point, heading)[0]
        if hit is None:
            return found, False
        found.append(first + float((np.array(hit) - start) @ unit) / length)
        point = hit + heading * eps
    return found, True


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


# --- 1a. RUNNING TOOLS ---
#
# Every tool that works through a cloud runs a slice at a time on a timer,
# so the sidebar can show how far it has got and stop it on request. The six
# big jobs have job classes of their own; the smaller tools are written as
# generators and driven by PC_TaskOperator. Either way a request to stop,
# from Esc or from the Cancel button, lands in CANCEL_REQUESTS and is picked
# up between two slices, before anything has been written.

CANCEL_REQUESTS = set()
# The smaller tools that are running, as key: [stage, fraction done]
TASKS = {}
# What each of them said when it last finished, most recent last
TASK_STATUS = {}
# Points worked through between two checks of the clock
TASK_CHUNK = 262_144
# The six big jobs, by the key their Cancel carries, and the scene flag that
# says one is running
JOB_RUNNING = {"thickness": "pc_thin_running", "lattice": "pc_grid_running",
               "grow": "pc_grow_running", "video": "pc_vid_running",
               "fill": "pc_fill_running", "boolean": "pc_bool_running"}


def tool_key(idname):
    """An operator's id the way the panels name it, mesh.cluster_dist.

    A registered operator reads its own bl_idname back in Blender's C form,
    MESH_OT_cluster_dist, so it is put back into the Python form here.
    """
    if "_OT_" in idname:
        head, tail = idname.split("_OT_", 1)
        return f"{head.lower()}.{tail}"
    return idname


def take_cancel(tool):
    """True, once, if a stop was asked for this tool."""
    if tool in CANCEL_REQUESTS:
        CANCEL_REQUESTS.discard(tool)
        return True
    return False


class TaskError(Exception):
    """A user facing reason a tool stopped, and the level to report it at."""

    def __init__(self, message, level='ERROR'):
        super().__init__(message)
        self.level = level


class MESH_OT_CancelTask(bpy.types.Operator):
    bl_idname = "mesh.pc_cancel_task"
    bl_label = "Cancel"
    bl_description = ("Stops this tool after the step it is on. Nothing it "
                      "was working on is written")
    bl_options = {'INTERNAL'}

    tool: bpy.props.StringProperty(options={'SKIP_SAVE'})

    def execute(self, context):
        # Only while that tool runs, so a press that lands just after it
        # finished leaves nothing behind
        flag = JOB_RUNNING.get(self.tool)
        if self.tool in TASKS or (flag and getattr(context.scene, flag, False)):
            CANCEL_REQUESTS.add(self.tool)
        tag_redraw_sidebar(context)
        return {'FINISHED'}


class BigAsk:
    """Yielded by a task about to make more points than it should unasked."""

    def __init__(self, points, note):
        self.points, self.note = int(points), note


class PC_TaskOperator:
    """Runs a tool written as a generator, a slice at a time.

    work() yields (stage, fraction done) as it goes, and returns what to
    report: a message, or a (level, message) pair. It raises TaskError to
    stop with a reason. Whatever a tool writes is written after its last
    yield, so stopping it at a yield leaves the cloud as it was, and the
    generator's own finally blocks still run, which is where a mode is put
    back. self.context is the current context, renewed on every slice.
    """

    bl_options = {'REGISTER', 'UNDO'}
    BUDGET = 0.08

    _timer = None
    _work = None
    _asking = False

    @classmethod
    def poll(cls, context):
        return not TASKS

    @property
    def task_key(self):
        return tool_key(self.bl_idname)

    def work(self):
        raise NotImplementedError
        yield

    def _drive(self, budget):
        """Runs the work for a while: ('busy', None), ('ask', BigAsk) or
        ('done', what it returned)."""
        deadline = time.time() + budget
        try:
            while True:
                item = next(self._work)
                if isinstance(item, BigAsk):
                    return 'ask', item
                self._stage, self._fraction = item
                if time.time() >= deadline:
                    return 'busy', None
        except StopIteration as stop:
            return 'done', stop.value

    def _end(self, context):
        if self._asking:
            FILL_PROMPT.update(points=0, note="", answer=None)
            self._asking = False
        TASKS.pop(self.task_key, None)
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._work = None
        tag_redraw_sidebar(context)

    def _said(self, level, message, status=None):
        key = self.task_key
        TASK_STATUS.pop(key, None)
        TASK_STATUS[key] = message if status is None else status
        self.report({level}, message)

    def _done(self, result):
        level, message = (result if isinstance(result, tuple)
                          else ('INFO', result))
        if message:
            self._said(level, message)
        return {'FINISHED'}

    def execute(self, context):
        """Blocking path: used from scripts and in background mode."""
        self.context = context
        self._work = self.work()
        try:
            while True:
                state, result = self._drive(float('inf'))
                if state != 'ask':
                    break
                # A script cannot be asked, so a big job needs allow_large
                if not getattr(self, "allow_large", False):
                    self._work.close()
                    raise TaskError(f"{result.note}. Run it with "
                                    f"allow_large=True to go on")
        except TaskError as exc:
            self._said(exc.level, str(exc))
            return {'CANCELLED'}
        finally:
            self._work = None
        return self._done(result)

    def invoke(self, context, event):
        """Modal path: used when the button is pressed in the sidebar."""
        if getattr(context, "window", None) is None:
            return self.execute(context)
        key = self.task_key
        self.context = context
        self._work = self.work()
        self._stage, self._fraction = "Starting", 0.0
        CANCEL_REQUESTS.discard(key)
        TASKS[key] = [self._stage, 0.0]
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        key = self.task_key
        self.context = context
        if event.type == 'ESC' or take_cancel(key):
            self._work.close()
            self._end(context)
            self._said('WARNING', f"{self.bl_label} cancelled, nothing was "
                                  f"changed", status="Cancelled")
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._asking:
            # Held on a big job until the popup or the panel answers
            answer = FILL_PROMPT["answer"]
            if answer is None:
                return {'RUNNING_MODAL'}
            FILL_PROMPT.update(points=0, note="", answer=None)
            self._asking = False
            if not answer:
                self._work.close()
                self._end(context)
                self._said('WARNING', f"{self.bl_label} cancelled, nothing "
                                      f"was changed", status="Cancelled")
                return {'CANCELLED'}

        try:
            state, result = self._drive(self.BUDGET)
        except TaskError as exc:
            self._end(context)
            self._said(exc.level, str(exc))
            return {'CANCELLED'}
        except Exception as exc:
            # Anything unexpected, the object deleted while it ran for one,
            # still has to take the tool off the running list.
            self._end(context)
            self._said('ERROR', f"{self.bl_label} failed: {exc}")
            return {'CANCELLED'}

        if state == 'ask':
            self._asking = True
            FILL_PROMPT.update(points=result.points, note=result.note,
                               answer=None)
            TASKS[key] = ["Waiting for your OK", self._fraction]
            tag_redraw_sidebar(context)
            bpy.ops.mesh.pc_fill_confirm('INVOKE_DEFAULT')
            return {'RUNNING_MODAL'}
        if state == 'busy':
            TASKS[key] = [self._stage, self._fraction]
            tag_redraw_sidebar(context)
            return {'RUNNING_MODAL'}
        self._end(context)
        return self._done(result)


def task_mesh(context):
    """The active object, which these tools need to be a mesh."""
    obj = context.active_object
    if obj is None:
        raise TaskError("No active object")
    if obj.type != 'MESH':
        raise TaskError(f"'{obj.name}' is not a mesh")
    return obj


def object_mode(obj):
    """Out of edit mode, before any attribute is touched (see read_thickness)."""
    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def put_mode_back(obj, mode):
    try:
        if mode != 'OBJECT' and obj.mode != mode:
            bpy.ops.object.mode_set(mode=mode)
    except (RuntimeError, ReferenceError):
        pass


def still_same(obj, total):
    """Stop rather than write onto a cloud that changed while a tool ran."""
    if obj.mode != 'OBJECT' or len(obj.data.vertices) != total:
        raise TaskError(f"'{obj.name}' was changed while this ran, so "
                        f"nothing was written. Press it again")


def point_attribute(mesh, name, data_type):
    """The named point attribute, made over if it is of another kind."""
    attr = mesh.attributes.get(name)
    if attr is not None and (attr.domain != 'POINT'
                             or attr.data_type != data_type):
        mesh.attributes.remove(attr)
        attr = None
    if attr is None:
        attr = mesh.attributes.new(name=name, type=data_type, domain='POINT')
    return attr


def read_point_colors(mesh, name):
    """The RGB of every point, from the named color attribute."""
    attr = mesh.attributes.get(name)
    if attr is None:
        raise TaskError(f"Color attribute '{name}' not found")
    if attr.domain != 'POINT' or attr.data_type not in ('FLOAT_COLOR',
                                                        'BYTE_COLOR'):
        raise TaskError(f"'{name}' is not a per point color")
    flat = np.empty(len(attr.data) * 4, dtype=np.float32)
    attr.data.foreach_get("color", flat)
    return flat.reshape(-1, 4)[:, :3]


def colors_hsv(rgb):
    """Hue, saturation and value of many colors at once.

    Worked out step for step the way Blender's own rgb_to_hsv does, in 32
    bit floats, so a point matches here exactly when mathutils Color.hsv
    would have matched it one at a time. Returned in 64 bit, which is what
    the comparisons were done in.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    swap = g < b
    g, b = np.where(swap, b, g), np.where(swap, g, b)
    k = np.where(swap, np.float32(-1.0), np.float32(0.0))
    low = b
    swap = r < g
    r, g = np.where(swap, g, r), np.where(swap, r, g)
    k = np.where(swap, np.float32(-2.0 / 6.0) - k, k)
    low = np.where(swap, np.minimum(g, b), low)
    chroma = r - low
    h = np.abs(k + (g - b) / (np.float32(6.0) * chroma + np.float32(1e-20)))
    s = chroma / (r + np.float32(1e-20))
    return (h.astype(np.float64), s.astype(np.float64),
            r.astype(np.float64))


def color_matches(rgb, ref_hsv, tolerance, material_mode=False):
    """check_color_match for many points at once."""
    h, s, v = colors_hsv(rgb)
    ref_h, ref_s, ref_v = ref_hsv
    dh = np.abs(h - ref_h)
    dh = np.where(dh > 0.5, 1.0 - dh, dh)
    ok = (dh <= tolerance) & (np.abs(s - ref_s) <= tolerance)
    if not material_mode:
        ok &= np.abs(v - ref_v) <= tolerance
    return ok


def color_scores(rgb, ref_hsv, tolerance, material_mode=False):
    """How well each point matches, 0 to 1, as Analyze Colors scores it."""
    h, s, v = colors_hsv(rgb)
    ref_h, ref_s, ref_v = ref_hsv
    dh = np.abs(h - ref_h)
    dh = np.where(dh > 0.5, 1.0 - dh, dh)
    width = tolerance + 0.001
    score = (np.maximum(0.0, 1.0 - dh / width)
             * np.maximum(0.0, 1.0 - np.abs(s - ref_s) / width))
    if not material_mode:
        score *= np.maximum(0.0, 1.0 - np.abs(v - ref_v) / width)
    return score


def color_steps(rgb, measure, stage):
    """Runs measure over the colors a chunk at a time, yielding between."""
    total = len(rgb)
    parts = []
    for start in range(0, total, TASK_CHUNK):
        end = min(start + TASK_CHUNK, total)
        parts.append(measure(rgb[start:end]))
        yield stage, 0.1 + 0.85 * end / max(total, 1)
    return np.concatenate(parts) if parts else np.zeros(0)


def reference_hsv(scene):
    color = scene.pc_ref_color
    return Color((color[0], color[1], color[2])).hsv


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
        "Press Cancel, or Esc, while it runs to stop it"
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
        CANCEL_REQUESTS.discard("thickness")
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        if event.type == 'ESC' or take_cancel("thickness"):
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


class MESH_OT_SelectThin(PC_TaskOperator, bpy.types.Operator):
    bl_idname = "mesh.pc_select_thin"
    bl_label = "Select Thin"
    bl_description = (
        "Step 3: Selects every point where the cloud is thinner than the "
        "Thin Below value, using the thickness measured by Analyze. Move the "
        "slider and press again to try a different cut, it does not measure "
        "again"
    )

    def work(self):
        yield "Reading the thickness", 0.0
        context = self.context
        try:
            mesh, values = read_thickness(context)
        except ValueError as exc:
            raise TaskError(str(exc))

        mask = values <= context.scene.pc_thin_threshold
        count = int(np.count_nonzero(mask))

        # Selection is written straight onto the mesh rather than through
        # bmesh: a python loop over the vertices of a 30M point cloud takes
        # minutes at this size.
        select_points(mesh, mask)
        context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode='EDIT')
        return f"Selected {count:,} of {len(values):,} points"


def select_points(mesh, mask):
    """Selects exactly the points in mask, and nothing else."""
    if len(mesh.polygons):
        mesh.polygons.foreach_set("select", np.zeros(len(mesh.polygons),
                                                     dtype=bool))
    if len(mesh.edges):
        mesh.edges.foreach_set("select", np.zeros(len(mesh.edges),
                                                  dtype=bool))
    mesh.vertices.foreach_set("select", mask)
    mesh.update()


class MESH_OT_ThinHeatmap(PC_TaskOperator, bpy.types.Operator):
    bl_idname = "mesh.pc_thin_heatmap"
    bl_label = "Heatmap"
    bl_description = (
        "Creates a color attribute 'pc_thin_heat' shading the cloud by "
        "thickness, red where it is thin and blue where it is thick, scaled "
        "over the Measure Up To range (requires attribute display)"
    )

    def work(self):
        yield "Reading the thickness", 0.0
        context = self.context
        try:
            mesh, values = read_thickness(context)
        except ValueError as exc:
            raise TaskError(str(exc))

        span = max(float(context.scene.pc_thin_max), 1e-9)
        ramp = np.clip(values / span, 0.0, 1.0)
        colors = np.empty((len(values), 4), dtype=np.float32)
        colors[:, 0] = 1.0 - ramp
        colors[:, 1] = 0.0
        colors[:, 2] = ramp
        colors[:, 3] = 1.0

        attr = point_attribute(mesh, THIN_HEAT_ATTR, 'FLOAT_COLOR')
        attr.data.foreach_set("color", colors.ravel())
        mesh.update()
        shown = show_color_attribute(context, mesh, THIN_HEAT_ATTR)

        thin = int(np.count_nonzero(values <= context.scene.pc_thin_threshold))
        note = (f"Heatmap '{THIN_HEAT_ATTR}': red under {span:g}, blue at it. "
                f"{thin:,} points are under the Thin Below cut")
        if not shown:
            note += (". Set the viewport to Solid shading with Color set to "
                     "Attribute to see it")
        return note


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
        # New Pitch starts out as the measured one, so Rescale reads as no
        # change at all until a number is actually retyped.
        set_new_pitch(scene, self.pitch)
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
        "Press Cancel, or Esc, while it runs to stop it"
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
        CANCEL_REQUESTS.discard("lattice")
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        if event.type == 'ESC' or take_cancel("lattice"):
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
        "front. Press Cancel, or Esc, while it runs to stop it"
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
        CANCEL_REQUESTS.discard("grow")
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        if event.type == 'ESC' or take_cancel("grow"):
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
        "suite can work on it straight away. Press Cancel, or Esc, while it "
        "runs to stop it"
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
        CANCEL_REQUESTS.discard("video")
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        if event.type == 'ESC' or take_cancel("video"):
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


# --- 1f. PITCH RESCALE LOGIC ---
#
# Retypes the spacing of a cloud that already sits on a grid: Analyze Lattice
# measures the pitch, a new pitch is typed next to it, and every point is
# moved until the grid really steps that far.
#
# The stretch runs along the grid's own three directions rather than along X,
# Y and Z, because that is the only way to change a step length without
# rotating or shearing the grid. Taken around the grid origin, every point
# keeps the whole cell it sat in and its offset inside that cell grows with
# the cell, so a cloud that fitted the grid before still fits it afterwards
# and anything already grown on the same grid still lines up.

# Set while New Pitch is being written from code, so the Uniform link does
# not read its own write back as an edit made by hand.
_pitch_syncing = False


def set_new_pitch(scene, pitch):
    """Point New Pitch at a pitch, without the Uniform link firing."""
    global _pitch_syncing
    _pitch_syncing = True
    try:
        scene.pc_grid_new_pitch = tuple(float(value) for value in pitch)
        scene.pc_grid_new_pitch_prev = tuple(float(value) for value in pitch)
        scene.pc_grid_scale_status = ""
    finally:
        _pitch_syncing = False


def update_grid_new_pitch(self, context):
    """Carry an edit of one New Pitch value onto the other two.

    Blender does not say which component of a vector property was touched,
    so the one that moved is found by comparing against the copy kept from
    last time. With Uniform on it sets a single factor and the other two
    follow the pitch they were measured at, which keeps the proportions of an
    uneven grid instead of squaring it up.
    """
    if _pitch_syncing:
        return

    wanted = list(self.pc_grid_new_pitch)
    if self.pc_grid_pitch_uniform:
        pitch = list(self.pc_grid_pitch)
        previous = list(self.pc_grid_new_pitch_prev)
        moved = [i for i in range(3) if abs(wanted[i] - previous[i]) > 1e-12]
        if len(moved) == 1 and all(value > 1e-9 for value in pitch):
            factor = wanted[moved[0]] / pitch[moved[0]]
            if factor > 0.0:
                wanted = [value * factor for value in pitch]
    set_new_pitch(self, wanted)


def pitch_rescale_factors(scene):
    """How far each grid direction would stretch, or None if unanswerable."""
    pitch = np.array(scene.pc_grid_pitch, dtype=np.float64)
    target = np.array(scene.pc_grid_new_pitch, dtype=np.float64)
    if np.any(pitch < 1e-9) or np.any(target < 1e-9):
        return None
    return target / pitch


def pitch_rescale_matrix(basis, target):
    """The map that stretches a cloud until its grid steps measure `target`.

    `basis` holds the three step vectors as its columns, so scaling column i
    by the factor that direction has to grow by is the grid that is wanted,
    and mapping back through the original basis turns that into a transform
    for the points: whole cell counts come out untouched and the stretch
    happens only along the grid directions.
    """
    pitch = np.linalg.norm(basis, axis=0)
    if np.any(pitch < 1e-9):
        raise ValueError("No grid yet, press Analyze Lattice first")
    target = np.asarray(target, dtype=np.float64)
    if not np.all(np.isfinite(target)) or np.any(target < 1e-9):
        raise ValueError("New Pitch has to be greater than zero in all three "
                         "directions")
    factors = target / pitch
    scaled = basis * factors
    if abs(np.linalg.det(scaled)) < 1e-12:
        raise ValueError("That New Pitch flattens the grid, so it would not "
                         "span a volume any more")
    return scaled @ np.linalg.inv(basis), scaled, factors


def rescale_to_pitch(mesh, basis, origin, target, chunk=2_000_000):
    """Move every point of a mesh onto the new pitch, in one go."""
    steps = rescale_steps(mesh, basis, origin, target, chunk)
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


def rescale_steps(mesh, basis, origin, target, chunk=2_000_000):
    """Move every point of a mesh onto the new pitch, residuals included.

    A generator: yields the fraction done after each chunk, then 1.0 once
    more just before anything is written, and returns what rescale_to_pitch
    does. Stopped at any yield, the mesh is untouched.

    Writes nothing but the point positions and, where Analyze Lattice left
    one behind, the residual attribute, which is re-measured against the
    stretched grid rather than left reading the old one. Worked in chunks so
    a thirty million point cloud never needs a whole second copy of itself in
    float 64.
    """
    matrix, scaled, factors = pitch_rescale_matrix(basis, target)
    total = len(mesh.vertices)
    flat = np.empty(total * 3, dtype=np.float32)
    mesh.vertices.foreach_get('co', flat)
    coords = flat.reshape(-1, 3)

    attr = mesh.attributes.get(GRID_RESIDUAL_ATTR)
    if attr is not None and (attr.domain != 'POINT'
                             or attr.data_type != 'FLOAT'):
        attr = None
    residual = None if attr is None else np.empty(total, dtype=np.float32)
    inverse = np.linalg.inv(basis)

    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        offset = coords[start:end].astype(np.float64) - origin
        moved = offset @ matrix.T
        if residual is not None:
            cells = np.rint(offset @ inverse.T)
            residual[start:end] = np.linalg.norm(moved - cells @ scaled.T,
                                                 axis=1)
        coords[start:end] = (moved + origin).astype(np.float32)
        yield end / max(total, 1) * 0.95

    yield 1.0
    mesh.vertices.foreach_set('co', flat)
    if residual is not None:
        attr.data.foreach_set("value", residual)
    mesh.update()
    return scaled, factors, residual is not None


class MESH_OT_RescalePitch(PC_TaskOperator, bpy.types.Operator):
    bl_idname = "mesh.pc_rescale_pitch"
    bl_label = "Rescale"
    bl_description = (
        "Step 1d: Moves every point of the active cloud until its grid steps "
        "measure New Pitch, then reads the result back into Pitch.\n\n"
        "The stretch runs along the three grid directions found by Analyze "
        "Lattice and around the grid origin, so the grid keeps its "
        "orientation, every point keeps the cell it sat in and anything "
        "already grown on this grid still lines up. It scales the points "
        "themselves and not the object, so the object transform, its "
        "modifiers and all of its attributes are left as they are. "
        "Distances set elsewhere in the sidebar, Radius or Thin Below for "
        "instance, still read in the old scale afterwards"
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and not context.scene.pc_grid_running
                and not context.scene.pc_grow_running
                and not TASKS)

    def work(self):
        context = self.context
        scene, obj = context.scene, context.active_object
        if obj is None or obj.type != 'MESH':
            raise TaskError("Make a point cloud mesh the active object first")
        total = len(obj.data.vertices)
        if total == 0:
            raise TaskError(f"'{obj.name}' has no points to rescale")

        factors = pitch_rescale_factors(scene)
        if factors is None:
            raise TaskError("Needs a Pitch and a New Pitch that are both "
                            "greater than zero in all three directions, "
                            "press Analyze Lattice first")
        if np.all(np.abs(factors - 1.0) <= 1e-9):
            raise TaskError("New Pitch already matches Pitch, nothing to "
                            "rescale", 'WARNING')

        mode = obj.mode
        if mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        start = time.time()
        try:
            basis, origin = scene_lattice(scene)
            steps = rescale_steps(obj.data, basis, origin,
                                  scene.pc_grid_new_pitch)
            while True:
                try:
                    fraction = next(steps)
                except StopIteration as stop:
                    scaled, factors, residuals = stop.value
                    break
                if fraction >= 1.0:
                    still_same(obj, total)
                else:
                    yield "Rescaling", fraction
        except ValueError as exc:
            raise TaskError(str(exc))
        finally:
            put_mode_back(obj, mode)

        # The grid the points now sit on, so Grow and a second Rescale carry
        # straight on from here without asking for another analysis.
        pitch = np.linalg.norm(scaled, axis=0)
        scene.pc_grid_basis = tuple(scaled.T.ravel())
        scene.pc_grid_pitch = tuple(pitch)
        set_new_pitch(scene, pitch)

        scene.pc_grid_scale_status = (f"rescaled  pitch {pitch[0]:.4g} / "
                                      f"{pitch[1]:.4g} / {pitch[2]:.4g}")
        detail = (f"Rescaled {total:,} points by x{factors[0]:.4g}, "
                  f"x{factors[1]:.4g}, x{factors[2]:.4g} along the grid "
                  f"directions around the grid origin; pitch now "
                  f"{pitch[0]:.4g}, {pitch[1]:.4g}, {pitch[2]:.4g}; "
                  f"{time.time() - start:.1f}s")
        if residuals:
            detail += f"; '{GRID_RESIDUAL_ATTR}' re-measured"
        tag_redraw_sidebar(context)
        return detail


def build_cloud_copy(cloud, collection, positions, source, suffix):
    """A copy of the cloud object holding new points, linked beside it.

    Point i of the copy sits at positions[i], in the cloud's own space, and
    takes every point attribute from point source[i] of the cloud, colors
    included. The copy keeps the transform, modifiers and materials of the
    cloud, so it lines up with it straight away. Returns the new object and
    the names of the attributes that could not be carried over.
    """
    mesh = cloud.data
    dropped = []
    saved = []
    for attr in mesh.attributes:
        if (attr.domain != 'POINT' or attr.name == 'position'
                or attr.name.startswith('.')):
            continue
        layout = BOOL_ATTR_LAYOUT.get(attr.data_type)
        if layout is None:
            dropped.append(attr.name)
            continue
        size, prop, dtype = layout
        buffer = np.empty(len(attr.data) * size, dtype=dtype)
        try:
            attr.data.foreach_get(prop, buffer)
        except (TypeError, RuntimeError):
            dropped.append(attr.name)
            continue
        saved.append((attr.name, attr.data_type, prop,
                      buffer.reshape(-1, size)))
    active_color = mesh.color_attributes.active_color_name
    render_color = mesh.color_attributes.default_color_name

    # A copy of the object, so the fill arrives with the transform,
    # modifiers and materials of the cloud it was mirrored from, and
    # lines up with it in the viewport without anything being set up
    # again by hand.
    result = cloud.copy()
    result.data = mesh.copy()
    result.name = f"{cloud.name}{suffix}"
    result.data.name = f"{mesh.name}{suffix}"
    collection.objects.link(result)

    out = result.data
    out.clear_geometry()
    out.vertices.add(len(positions))
    out.vertices.foreach_set('co', positions.ravel())
    for name, data_type, prop, values in saved:
        attr = out.attributes.get(name)
        if attr is None:
            attr = out.attributes.new(name=name, type=data_type,
                                      domain='POINT')
        attr.data.foreach_set(prop, values[source].ravel())
    if active_color and active_color in out.color_attributes:
        out.color_attributes.active_color_name = active_color
    if render_color and render_color in out.color_attributes:
        out.color_attributes.default_color_name = render_color
    out.update()
    return result, dropped


# --- 1g. FILL VOLUME LOGIC ---
#
# Fills a closed mesh with mirrored copies of a point cloud.
#
# The copies are laid out in the cloud's own rows rather than in world
# space: a tile is the block of rows the cloud repeats on, stepped a whole
# number of rows, and every second tile along a direction is turned around.
# Mirroring about the plane half a row past the block's last row is what
# makes a seam come out right, because the mirrored row then lands exactly
# one spacing from the row it was mirrored off: the spacing never changes
# anywhere in the fill and no point is ever written twice. Tile zero is the
# source itself, so the cloud that was picked keeps its spacing and its
# position exactly.
#
# Both of those are measured rather than assumed, and both of them have to
# be. A grid step is not always one row of points, so what a row is comes
# from measure_rows rather than from the grid; and the rows a cloud reaches
# into are not always the block it repeats on, so where the block ends comes
# from repeating_block, which sees past the points that jitter carries over
# the end and the strays beyond them. Either one read off wrongly stretches
# the tile, and every row it is stretched by opens as a gap at every seam.
#
# A point is kept where it falls inside the closed mesh, which is the same
# crossing parity test the Boolean tool uses, counted per grid line rather
# than per point: one ray along a line of points sees every crossing on that
# line, so sorting those crossings says which points on it are inside. Per
# point voting needs millions of rays to fill a volume and hours to cast
# them, while a line needs one; the line answer is checked against the per
# point vote on a random sample, and the agreement is reported.


def mirror_tile_step(index, lo, hi, step=1):
    """Scale and offset, in rows, for tile `index` along one axis.

    Tile 0 is the source block itself. An odd tile is turned around about the
    plane half a step past the block's last row, so its first row lands
    exactly one step beyond that row; an even tile is the block stepped whole
    periods along. Either way tile k covers rows lo + k*period through
    hi + k*period, and consecutive tiles are one step apart, which is what
    lets them meet with neither a gap nor a shared row.

    `step` is how the occupied rows of the block are spaced, in rows, so a
    cloud using every second row of its grid is mirrored a whole spacing away
    rather than half of one.

    A `step` of 0 mirrors through the last row itself instead, for a cloud
    whose layers alternate (see layers_alternate): the tiles on either side
    of a seam then share that row, and tile_shared_rows says which of the two
    writes it.
    """
    period = hi - lo + step
    if index % 2 == 0:
        return 1, index * period
    return -1, (2 * hi + step) + (index - 1) * period


def tile_shared_rows(index, rows, lo, hi, seam):
    """The points a tile leaves out because the tile beside it has them.

    Only a seam mirrored through an end row (a `seam` of 0) shares a row, and
    the row has to be written once: every tile keeps the row at its low end
    and leaves out the one at its high end, which is row hi of a tile that is
    not turned around and row lo of one that is.
    """
    drop = np.zeros(len(rows), dtype=bool)
    for axis in range(3):
        if seam[axis] != 0:
            continue
        end = hi[axis] if int(index[axis]) % 2 == 0 else lo[axis]
        drop |= rows[:, axis] == end
    return drop


def layers_alternate(rows, axis, lo, hi, step, least=4):
    """Whether neighbouring layers along `axis` hold different patterns.

    On a staggered stacking - face centred and body centred grids read along
    a cube edge, a hexagonal grid read across its rows - every second layer
    is shifted sideways by half a row. Mirroring half a row past the last
    layer then sets a copy of that layer straight on top of it, closer than
    any two points of the cloud are, where the cloud itself would have put
    the shifted layer. Mirroring through the last layer is what such a
    stacking is symmetric about, so that is what is used when this is true.

    Read off a few pairs of neighbouring layers around the middle of the
    block, as the share of sideways positions the two have in common.
    """
    others = [a for a in range(3) if a != axis]
    width = int(hi[others[1]] - lo[others[1]]) + 1
    layers = np.arange(int(lo[axis]), int(hi[axis]) + 1, int(step[axis]))
    if len(layers) < 3:
        return False

    def seats(layer):
        sel = rows[:, axis] == layer
        return np.unique((rows[sel, others[0]] - lo[others[0]]) * width
                         + (rows[sel, others[1]] - lo[others[1]]))

    middle = len(layers) // 2
    shares = []
    for i in range(max(0, middle - 2), min(len(layers) - 1, middle + 2)):
        one, two = seats(layers[i]), seats(layers[i + 1])
        if len(one) < least or len(two) < least:
            continue
        shares.append(len(np.intersect1d(one, two, assume_unique=True))
                      / min(len(one), len(two)))
    return bool(shares) and float(np.median(shares)) < 0.5


def box_frame(basis, most=6, slack=0.02):
    """Grid steps that run along the object's own X, Y and Z, if it has them.

    A cloud is usually cut as a box in its own axes, and its faces are then
    the planes it has to be mirrored about. The steps Analyze Lattice finds
    are the shortest ones, which on a face centred, body centred or
    hexagonal grid lean away from those axes, so a box of the cloud is no box
    of cells and the mirror lands well past its faces. Most grids also have
    a longer step along each axis - the cube edge of a face centred grid -
    and a cell built of those is a box exactly like the cloud.

    Each step is a whole number of the found steps, so mirroring on it still
    puts every copy on the grid. Returns the three as columns, or None when
    some axis is no grid direction at all, or the steps are the found ones.
    """
    inverse = np.linalg.inv(basis)
    columns = []
    for axis in range(3):
        cells = inverse[:, axis] / np.abs(inverse[:, axis]).max()
        for times in range(1, most + 1):
            whole = np.rint(cells * times)
            if np.abs(cells * times - whole).max() <= slack:
                break
        else:
            return None
        step = basis @ whole
        length = float(np.linalg.norm(step))
        if length < 1e-12 or abs(step[axis]) < 0.999 * length:
            return None
        columns.append(step if step[axis] > 0.0 else -step)
    frame = np.column_stack(columns)
    if abs(np.linalg.det(frame)) < 1e-12:
        return None
    scale = float(np.abs(basis).max())
    if all(any(np.allclose(column, sign * basis[:, k], atol=1e-9 * scale)
               for k in range(3) for sign in (1.0, -1.0))
           for column in columns):
        return None
    return frame


def block_fill(rows, lo, hi, step):
    """How much of its block a cloud fills, about 1 for a solid one.

    Measured against the cloud's own density in the middle of the block, so
    a staggered stacking or a cloud on every second row reads as full too. A
    box cut from a grid that is turned against it fills a lot less: the
    corners of the block it reaches into are empty, and every one of those
    is a gap at a seam.
    """
    quarter = (hi - lo) // 4
    core = np.all((rows >= lo + quarter) & (rows <= hi - quarter), axis=1)
    inner = int(np.count_nonzero(core))
    if inner == 0:
        return 0.0
    whole = (hi - lo + step).astype(np.float64)
    middle = (hi - lo - 2 * quarter + step).astype(np.float64)
    return float(len(rows)) / (inner * float(np.prod(whole / middle)))


def filled_block(rows, lo, hi, step, solid=0.9, full=0.99,
                 cap=64_000_000):
    """The biggest box of rows inside the block that the cloud really fills.

    For a cloud whose faces do not run along any grid steps, a box cut from a
    grid turned against it, where no mirror about its faces keeps the grid:
    a tile of whole rows can only be seamless if every row of it is full, so
    the empty corners are pared off, one face at a time and always the
    emptiest face, until every face is full.

    Only for a cloud that is solid in the middle, where every seat of every
    row is taken. Returns (lo, hi), or None for a cloud that is not.
    """
    dims = (hi - lo) // step + 1
    if float(np.prod(dims)) > cap:
        return None
    on = np.all((rows - lo) % step == 0, axis=1)
    on &= np.all((rows >= lo) & (rows <= hi), axis=1)
    grid = np.zeros(tuple(int(d) for d in dims), dtype=bool)
    grid[tuple(((rows[on] - lo) // step).T)] = True

    edge = dims // 4
    core = grid[tuple(slice(int(edge[a]), int(dims[a] - edge[a]))
                      for a in range(3))]
    if core.size == 0 or float(core.mean()) < solid:
        return None

    start, end = np.zeros(3, dtype=np.int64), dims.astype(np.int64)
    while True:
        worst, face = full, None
        for axis in range(3):
            if end[axis] - start[axis] <= 1:
                continue
            for side, layer in ((0, start[axis]), (1, end[axis] - 1)):
                cut = [slice(int(start[a]), int(end[a])) for a in range(3)]
                cut[axis] = int(layer)
                share = float(grid[tuple(cut)].mean())
                if share < worst:
                    worst, face = share, (axis, side)
        if face is None:
            break
        if face[1] == 0:
            start[face[0]] += 1
        else:
            end[face[0]] -= 1
    return lo + start * step, lo + (end - 1) * step


def tile_affine(index, lo, hi, step, base, pitch):
    """One tile's turn and step: a scale, an offset in cells, and in rows.

    The mirror is worked out in rows, because rows are what has to meet
    across a seam, and handed back in cells as well, because cells are what
    the points are placed in. A row n of the source becomes row one*n + shift
    of the tile, and a point at u cells lands at one*u + offset.
    """
    scale = np.empty(3, dtype=np.float64)
    offset = np.empty(3, dtype=np.float64)
    shifts = np.empty(3, dtype=np.int64)
    for axis in range(3):
        one, shift = mirror_tile_step(int(index[axis]), int(lo[axis]),
                                      int(hi[axis]), int(step[axis]))
        scale[axis], shifts[axis] = one, shift
        offset[axis] = base[axis] * (1.0 - one) + shift * pitch[axis]
    return scale, offset, shifts


def rows_fit(column, base, step, quarter=0.25):
    """The share of points sitting within a quarter row of a row."""
    if step <= 0.0:
        return 0.0
    off = (column - base) / step
    off = np.abs(off - np.rint(off))
    return float(np.count_nonzero(off <= quarter)) / len(off)


def histogram_rows(column, fine=0.02, cap=4_000_000):
    """A first guess at the row spacing, for a spacing that is not a simple
    fraction of a grid step. A row is a run of neighbouring bins that hold
    something, and the spacing is the middle one of the gaps between runs.
    Jitter smears one row into the next here, which is why what this returns
    is only ever a guess and is always tested before it is used.
    """
    low, high = float(column.min()), float(column.max())
    if high - low < fine:
        return low, 0.0
    bins = min(int((high - low) / fine) + 2, cap)
    counts, edges = np.histogram(column, bins=bins,
                                 range=(low - fine, high + fine))
    filled = counts > 0
    starts = np.nonzero(filled & ~np.concatenate(([False], filled[:-1])))[0]
    ends = np.nonzero(filled & ~np.concatenate((filled[1:], [False])))[0]
    if len(starts) < 2:
        return low, 0.0
    middles = (edges[starts] + edges[ends + 1]) * 0.5
    return float(middles[0]), float(np.median(np.diff(middles)))


def measure_rows(u, finest=16, share=0.7, sample=200_000, seed=0):
    """How far apart the cloud's rows of points sit, along each direction.

    A grid step is not the spacing of the points along it. Analyze Lattice
    reports the shortest step that carries the whole cloud onto itself, and
    on a staggered or offset stacking that step spans two rows of points
    rather than one: the row in between is reached by a step that also moves
    sideways. Mirroring a tile one cell past its last row then leaves a row
    of empty space in the middle of every seam, and worse, two rows sharing
    one number puts the rays that test the fill between the rows instead of
    along them, which answers whole lines of points wrongly.

    The spacing is found by testing one rather than by hunting for it: a
    cell, then half of one, then a third, down to a sixteenth, taking the
    first that the points actually sit on. A spacing too large is given away
    at once, because the rows it misses land half way between the ones it
    keeps; a spacing too small fits as well as the right one, which is why
    the first that fits is taken and not the best.

    A spacing that is no simple fraction of a grid step - which happens when
    Pitch has been set by hand to something the cloud does not keep - is
    guessed at from a histogram instead, and used only if the points really
    sit on it. A spacing of more than a grid step needs nothing special: the
    rows are then a cell apart with every second or third one empty, and the
    block measurement reads that as a step of its own.

    Returns (base, pitch), the first row and the row spacing, both in cells.
    """
    if len(u) > sample:
        u = u[np.random.default_rng(seed).choice(len(u), sample,
                                                 replace=False)]
    base = np.zeros(3, dtype=np.float64)
    pitch = np.ones(3, dtype=np.float64)

    for axis in range(3):
        column = u[:, axis]
        if float(column.max() - column.min()) < 1e-9:
            continue

        for divide in range(1, finest + 1):
            if rows_fit(column, 0.0, 1.0 / divide) >= share:
                pitch[axis] = 1.0 / divide
                break
        else:
            start, guess = histogram_rows(column)
            if guess > 0.0 and rows_fit(column, start, guess) >= share:
                base[axis], pitch[axis] = start, guess

        # The rows are fitted against the points themselves, because any
        # drift in the spacing is multiplied by every row of every tile
        # until it shows at the far side of the fill. Only the points that
        # sit in a row are fitted to it: the ones jitter has carried out of
        # their row, and the strays, would otherwise drag the spacing out by
        # a fraction of a percent, which is enough.
        for _ in range(3):
            number = np.rint((column - base[axis]) / pitch[axis])
            near = (np.abs(column - (base[axis] + number * pitch[axis]))
                    <= 0.25 * pitch[axis])
            if np.count_nonzero(near) < 8:
                break
            sat, where = number[near], column[near]
            count = float(len(sat))
            sum_n, sum_u = float(sat.sum()), float(where.sum())
            spread = float((sat * sat).sum()) - sum_n * sum_n / count
            if spread <= 1e-9:
                break
            slope = (float((sat * where).sum())
                     - sum_n * sum_u / count) / spread
            if not (0.5 * pitch[axis] < slope < 2.0 * pitch[axis]):
                break
            pitch[axis] = slope
            base[axis] = (sum_u - slope * sum_n) / count
    return base, pitch


def repeating_block(rows, offsets=None, floor=10, cut=2, steady=0.75,
                    huddle=0.25, share=0.002, least=32, passes=4):
    """The block of rows that really repeats, and the step of its rows.

    The rows a cloud occupies are not the block it is built from. Jitter
    carries some of the points of an end row more than half a row past it, so
    they count as a row of their own, and the strays a scan leaves behind sit
    further out still. Either stretches the block, and every row it is
    stretched by opens as a gap at every seam of the fill.

    An end row is read as spill, rather than as part of the block, when any
    of three things is true of it:

    * it holds less than a tenth of the row immediately inside it. A tenth
      from one row to the next is a cliff, and a shape does not have one at
      its own surface: it is the signature of points that have rounded out of
      the row below.
    * it holds less than half of each of the two rows inside it, while those
      two hold about as much as each other. That is a layer the cloud was
      cut through part way - the last slice of a voxel or scan export, only
      partly filled - and it is the worst thing a seam can meet, because
      mirroring sets it against its own copy and the two read as a gap. A
      taper shrinks row after row instead, so the row inside it is shrinking
      too, and it is left alone.
    * its points sit, on average, more than a quarter of a row towards the
      body of the cloud from the row itself. A real row's points straddle it;
      rounded over points all huddle against the inner edge of the row they
      landed in, which is what gives them away however many of them there are.
    * it holds less than a tenth of the busiest row inside it, which sees
      through two or three sparse rows in a row where the test against the
      neighbour cannot: the row beside a stray is often another stray.
    * it is detached: the gap from it to the next occupied row inward is
      more than twice the usual gap between rows. A cloud is built of rows
      next to one another, so a row standing several empty rows off the end
      of one is something else that happens to be in the same object. This
      is what catches a stray on a small cloud, where a single stray point
      can be an eighth of what a real row holds and no test that counts
      points will reach it.

    The last two are rationed, to `share` of the points or `least` of them,
    whichever is more. They reason about how much a row holds and where it
    sits rather than about the points in it, so they are the ones that could
    eat into a cloud that genuinely tapers away at its ends or is built of
    separate pieces; the first two describe the mechanism exactly and are
    left to trim as far as they need to.

    The step is the spacing of the occupied rows, as a whole number of rows:
    1 for a cloud that fills its grid and 2 for one that uses every second
    row, and it is what a seam has to be as wide as.

    `offsets` are the points' own offsets from their rows, in rows, which is
    what the second test reads. Returns (low, high, step, keep, stalled),
    with `keep` the points inside the block and `stalled` set if the budget
    stopped a trim that was called for.
    """
    keep = np.ones(len(rows), dtype=bool)
    low, high = rows.min(axis=0), rows.max(axis=0)
    budget, spent, stalled = max(least, int(share * len(rows))), 0, False

    for _ in range(passes):
        trimmed = False
        for axis in range(3):
            column = rows[keep, axis]
            layers, counts = np.unique(column, return_counts=True)
            if offsets is None:
                means = np.zeros(len(layers), dtype=np.float64)
            else:
                means = (np.bincount(np.searchsorted(layers, column),
                                     weights=offsets[keep, axis],
                                     minlength=len(layers)) / counts)

            while len(layers) > 2:
                low_spill = (counts[0] * floor < counts[1]
                             or means[0] > huddle)
                high_spill = (counts[-1] * floor < counts[-2]
                              or means[-1] < -huddle)
                low_thin = counts[0] * floor < counts[1:].max()
                high_thin = counts[-1] * floor < counts[:-1].max()

                # A row standing several empty rows off the end of the body
                # is not part of it. Measured against the usual gap between
                # the others, so that a cloud whose rows are meant to come in
                # pairs, or every second row, is not read as detached
                # everywhere - and so that the gap under test cannot be what
                # makes itself look usual, which is what happens on a cloud
                # with few rows and a stray beyond them.
                apart = np.diff(layers)
                if len(apart) > 1:
                    low_thin = (low_thin
                                or apart[0] > 2.0 * float(np.median(apart[1:])))
                    high_thin = (high_thin
                                 or apart[-1] > 2.0 * float(np.median(apart[:-1])))

                if low_spill or (low_thin and spent + counts[0] <= budget):
                    if not low_spill:
                        spent += int(counts[0])
                    layers, counts, means = layers[1:], counts[1:], means[1:]
                elif high_spill or (high_thin and spent + counts[-1] <= budget):
                    if not high_spill:
                        spent += int(counts[-1])
                    layers, counts, means = layers[:-1], counts[:-1], means[:-1]
                else:
                    stalled = stalled or low_thin or high_thin
                    break
                trimmed = True
            low[axis], high[axis] = layers[0], layers[-1]

        keep = np.all((rows >= low) & (rows <= high), axis=1)
        if not trimmed:
            break

    # A slice cut through part way is one layer at most, so it is looked for
    # once at each end, after the spill and the strays are gone and the ends
    # are the cloud's own. Looked for any earlier, the ragged edge of a round
    # shape on a coarse grid, whose rows go up and down as they come, could
    # read as one.
    for axis in range(3):
        layers, counts = np.unique(rows[keep, axis], return_counts=True)
        if len(layers) < 4:
            continue
        if (counts[0] * cut < min(counts[1], counts[2])
                and counts[1] >= steady * counts[2]):
            low[axis] = layers[1]
        if (counts[-1] * cut < min(counts[-2], counts[-3])
                and counts[-2] >= steady * counts[-3]):
            high[axis] = layers[-2]
    keep = np.all((rows >= low) & (rows <= high), axis=1)

    # Read off the block that is actually left, so that its own end rows are
    # occupied however the trimming of the other two axes fell.
    low, high = rows[keep].min(axis=0), rows[keep].max(axis=0)
    step = np.ones(3, dtype=np.int64)
    for axis in range(3):
        sitting = np.unique(rows[keep, axis]) - low[axis]
        if len(sitting) > 1:
            step[axis] = max(int(np.gcd.reduce(sitting)), 1)
    return low, high, step, keep, stalled


def world_box_span(obj):
    """How far an object reaches along world X, Y and Z."""
    corners = np.array([obj.matrix_world @ Vector(corner)
                        for corner in obj.bound_box], dtype=np.float64)
    return corners.max(axis=0) - corners.min(axis=0)


def fill_estimate(scene):
    """Rough tile count and point count for the panel.

    Deliberately cheap: the real count needs the grid cell of every point,
    which is far too much work to redo on every redraw. Bounding boxes along
    world X, Y and Z are close enough to warn about a fill that would run
    into tens of millions of points before it is started.
    """
    volume, cloud = scene.pc_fill_volume, scene.pc_fill_cloud
    if volume is None or cloud is None:
        return None, 0
    if volume.type != 'MESH' or cloud.type != 'MESH':
        return None, 0
    total = len(cloud.data.vertices)
    if total == 0:
        return None, 0
    outer, inner = world_box_span(volume), world_box_span(cloud)
    tiles = tuple(1 if inner[axis] <= 1e-9
                  else max(1, int(np.ceil(outer[axis] / inner[axis])))
                  for axis in range(3))
    return tiles, int(np.prod(tiles)) * total


class FillVolumeJob:
    """Mirror a point cloud through a closed mesh until the mesh is full.

    Reads two objects and writes a third: the mesh and the cloud are both
    left exactly as they are, and the fill is linked beside them as
    '<cloud>_filled', carrying the transform, modifiers, materials and point
    attributes of the cloud it was mirrored from.

    Kept separate from the operator so one implementation drives both the
    modal path, which shows progress in the panel, and the blocking path used
    from scripts and in background mode.

    Raises ValueError with a user facing message if the setup is not usable.
    """

    # A fill is cheap to ask for and expensive to run, so above this many
    # points the job stops after measuring and waits to be told to go on,
    # rather than let a mistyped pitch quietly start on a billion points.
    CONFIRM_ABOVE = 100_000_000
    MAX_HITS = 2000
    # How far outwards each inside span is widened, in cells, so that a point
    # lying exactly on the surface counts as inside.
    BOUNDARY = 1e-4
    VERIFY = 300
    VERIFY_RAYS = 5
    FIT_SAMPLE = 50_000

    def __init__(self, context):
        scene = context.scene
        self.scene = scene
        self.collection = context.collection
        self.container = scene.pc_fill_volume
        self.cloud = scene.pc_fill_cloud
        # Trimmed to the shape of the mesh, or every point of every tile
        # that reaches into its box, left for a Boolean afterwards.
        self.trim = getattr(scene, "pc_fill_mode", 'TRIM') != 'COVER'

        if self.container is None:
            raise ValueError("Pick the mesh to fill in the Fill Volume panel")
        if self.cloud is None:
            raise ValueError("Pick the point cloud to fill it with")
        if self.cloud == self.container:
            raise ValueError("The mesh and the point cloud must be two "
                             "different objects")
        if self.cloud.type != 'MESH':
            raise ValueError(f"'{self.cloud.name}' is not a mesh")
        if (self.container.type != 'MESH'
                or len(self.container.data.polygons) == 0):
            raise ValueError(f"'{self.container.name}' has no faces, so it "
                             f"cannot hold a volume")
        if len(self.cloud.data.vertices) == 0:
            raise ValueError(f"'{self.cloud.name}' has no points")

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Built here rather than in a stage, so the evaluated container mesh
        # is read while the depsgraph this job was handed is current.
        depsgraph = context.evaluated_depsgraph_get()
        try:
            self.bvh, self.box_min, self.box_max, self.open_edges = \
                build_container_bvh(self.container, depsgraph)
        except RuntimeError as exc:
            raise ValueError(str(exc))
        diagonal = float(np.linalg.norm(self.box_max - self.box_min))
        self.eps = max(diagonal * 1e-6, 1e-7)

        matrix = np.array(self.cloud.matrix_world, dtype=np.float64)
        self.linear, self.shift = matrix[:3, :3], matrix[:3, 3]
        if abs(np.linalg.det(self.linear)) < 1e-12:
            raise ValueError(f"'{self.cloud.name}' has a flattened "
                             f"transform, so its points have no place in the "
                             f"world")

        self.total = 0
        self.kept = 0
        self.outside = 0
        self.stalled = False
        self.row_step = np.ones(3, dtype=np.int64)
        self.seam = np.ones(3, dtype=np.int64)
        self.through = np.zeros(3, dtype=bool)
        self.framed = False
        self.fill = 1.0
        self.cropped = 0
        self.inside = None
        self.measured = np.ones(3, dtype=np.int64)
        self.asked = np.zeros(3, dtype=np.int64)
        self.row0 = np.zeros(3, dtype=np.float64)
        self.row_pitch = np.ones(3, dtype=np.float64)
        self.period = np.ones(3, dtype=np.int64)
        self.tile_count = 0
        self.tiles = np.ones(3, dtype=np.int64)
        self.candidates = 0
        self.waiting = False
        self.fit = 0.0
        self.capped = 0
        self.stray = 0
        self.checked = 0
        self.agreed = 0
        self.dropped = []
        self.chunks = []
        self.probes = []
        self.result = None
        self.cursor = 0
        self.tile_cursor = 0
        self.probe_cursor = 0
        self.fraction = 0.0
        self.start = time.time()

        self.stage = 0
        if self.trim:
            self.stages = (
                ("Reading the cloud", self._read),
                ("Measuring the volume", self._measure),
                ("Finding the grid lines", self._lines),
                ("Casting through the volume", self._cast),
                ("Mirroring the cloud", self._fill),
                ("Checking the fill", self._verify),
                ("Building the cloud", self._build),
            )
            self.weights = (0.06, 0.01, 0.03, 0.40, 0.33, 0.10, 0.07)
        else:
            # Nothing is tested against the surface, so there is nothing to
            # cast and nothing to check.
            self.stages = (
                ("Reading the cloud", self._read),
                ("Measuring the volume", self._measure),
                ("Mirroring the cloud", self._fill),
                ("Building the cloud", self._build),
            )
            self.weights = (0.12, 0.01, 0.67, 0.20)

    @property
    def stage_name(self):
        if self.stage >= len(self.stages):
            return "Done"
        if self.waiting:
            return "Waiting for your OK"
        return self.stages[self.stage][0]

    @property
    def progress(self):
        done = sum(self.weights[:self.stage])
        if self.stage < len(self.weights):
            done += self.weights[self.stage] * min(max(self.fraction, 0.0), 1.0)
        return min(done, 1.0)

    @property
    def big_note(self):
        return (f"{self.tiles[0]} x {self.tiles[1]} x {self.tiles[2]} tiles "
                f"of {self.total:,} points comes to {self.candidates:,} "
                f"points before the ones outside the mesh are dropped")

    def go_ahead(self):
        """The big fill was confirmed, so carry on past the measurement."""
        self.waiting = False

    def step(self, budget=0.08):
        if self.stage >= len(self.stages):
            return True
        # Held after measuring a big fill until go_ahead is called
        if self.waiting:
            return False
        if self.stages[self.stage][1](budget):
            self.stage += 1
            self.fraction = 0.0
        return self.stage >= len(self.stages)

    # -- stages -------------------------------------------------------------

    def _read(self, budget):
        """The cloud, the rows it sits in, and the block those rows repeat."""
        mesh = self.cloud.data
        self.total = len(mesh.vertices)
        coords = np.empty(self.total * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', coords)
        self.coords = coords.reshape(-1, 3).astype(np.float64)

        lattice, self.origin = scene_lattice(self.scene)

        # The frame the tiles are laid out in. The grid's own steps first;
        # then, when the grid also has steps along the object's X, Y and Z,
        # those, because a cloud cut as a box in its own axes only fills a
        # box of cells built of them: see box_frame. Whichever the cloud
        # fills its block better in is the one whose faces it really has.
        survey = self._survey(lattice)
        frame = box_frame(lattice)
        if frame is not None:
            boxed = self._survey(frame)
            if boxed['fit'] >= 0.9 and boxed['fill'] > survey['fill'] + 0.05:
                survey = boxed
                self.framed = True
        for name, value in survey.items():
            setattr(self, name, value)
        inside = self.inside

        if self.fit < 0.5:
            raise ValueError(
                f"Only {self.fit * 100:.0f}% of '{self.cloud.name}' sits in "
                f"rows along the grid Analyze Lattice found, so the copies "
                f"could not be placed a row apart from it. Make this cloud "
                f"the active object and press Analyze Lattice; if the fit it "
                f"reports is poor, this cloud does not sit on one grid")

        # Typed by hand only counts while Tile Override is ticked
        asked = np.array(self.scene.pc_fill_tile
                         if getattr(self.scene, "pc_fill_tile_override",
                                    False)
                         else (0, 0, 0), dtype=np.int64)
        self.asked = np.where(asked > 0, asked, 0)

        # A cloud that leaves the corners of its block empty, a box cut from
        # a grid turned against it, has no mirror that keeps the grid at its
        # faces, so it is cut down to the box of rows it does fill, or every
        # seam would open on those corners. Not when Tile is set by hand, and
        # not when that box would keep too little of the cloud to be it.
        if self.fill < 0.9 and not np.any(self.asked > 0):
            block = filled_block(self.rows[inside], self.lo, self.hi,
                                 self.row_step)
            if block is not None:
                within = inside & np.all((self.rows >= block[0])
                                         & (self.rows <= block[1]), axis=1)
                if np.count_nonzero(within) >= 0.3 * np.count_nonzero(inside):
                    self.cropped = int(np.count_nonzero(inside & ~within))
                    inside = within
                    self.lo = self.rows[inside].min(axis=0)
                    self.hi = self.rows[inside].max(axis=0)
                    self.fill = block_fill(self.rows[inside], self.lo,
                                           self.hi, self.row_step)

        # Half a row past the last layer, or through it, along each
        # direction: through it for a stacking whose layers alternate, which
        # a mirror half a row out would set a copy of a layer straight on
        # top of: see layers_alternate.
        for axis in range(3):
            self.through[axis] = (
                self.hi[axis] - self.lo[axis] >= 2 * self.row_step[axis]
                and layers_alternate(self.rows[inside], axis, self.lo,
                                     self.hi, self.row_step))
        self.seam = np.where(self.through, 0, self.row_step)
        self.measured = self.hi - self.lo + self.seam
        self.period = self.measured.copy()

        # A period typed into Tile wins over the measured one, for a cloud
        # whose own repeat is not something that can be read off its points:
        # the last layer of a structural unit cell can be as busy as any
        # other and still not belong to the block. Anything that then falls
        # outside the block is left out, the same as spill is.
        if np.any(self.asked > 0):
            self.period = np.where(self.asked > 0, self.asked, self.period)
            self.hi = self.lo + self.period - self.seam
            inside = np.all((self.rows >= self.lo)
                            & (self.rows <= self.hi), axis=1)
            if not np.any(inside):
                raise ValueError(
                    f"A tile of {self.period[0]} x {self.period[1]} x "
                    f"{self.period[2]} points holds no points of "
                    f"'{self.cloud.name}'. Set Tile back to 0 to have the "
                    f"block measured instead")
        self.outside = int(np.count_nonzero(~inside))
        if self.outside:
            self.source_index = np.nonzero(inside)[0].astype(np.int32)
            self.coords = self.coords[inside]
            self.u = self.u[inside]
            self.rows = self.rows[inside]
            self.total = len(self.coords)
            if self.total == 0:
                raise ValueError(f"'{self.cloud.name}' has no block of points "
                                 f"that repeats")
        else:
            self.source_index = np.arange(self.total, dtype=np.int32)
        self.inside = None
        return True

    def _survey(self, basis):
        """The cloud read in one frame: its rows, its block and its fill."""
        inverse = np.linalg.inv(basis)
        u = (self.coords - self.origin) @ inverse.T

        # Where the rows of points sit along each grid direction, which is
        # not always one row to a cell: see measure_rows. Everything from
        # here counts in rows, so that a seam comes out one row wide.
        row0, row_pitch = measure_rows(u)
        rows, fit = self._read_rows(u, row0, row_pitch)
        if fit < 0.9:
            # The rows could not be read off cleanly. The grid's own cells
            # are the next best thing, and were all this used to go on.
            plain = (np.zeros(3), np.ones(3))
            cells, plain_fit = self._read_rows(u, *plain)
            if plain_fit > fit:
                row0, row_pitch = plain
                rows, fit = cells, plain_fit

        # The block of rows that repeats, which is not always the box the
        # points happen to reach into: see repeating_block. Anything outside
        # it is left out of the fill, because a point sitting past the period
        # would be mirrored into the rows of the tile next door and land on
        # top of a point already there.
        spread = (u - row0) / row_pitch
        lo, hi, step, inside, stalled = repeating_block(rows, spread - rows)
        return dict(basis=basis, inverse=inverse, u=u, row0=row0,
                    row_pitch=row_pitch, rows=rows, fit=fit, lo=lo, hi=hi,
                    row_step=step, inside=inside, stalled=stalled,
                    fill=block_fill(rows[inside], lo, hi, step))

    def _read_rows(self, u, base, pitch):
        """Row numbers for every point, and how well they really sit in rows."""
        spread = (u - base) / pitch
        rows = np.rint(spread).astype(np.int64)
        off = np.abs(spread - rows).max(axis=1)
        if self.total > self.FIT_SAMPLE:
            off = off[np.random.default_rng(0).choice(self.total,
                                                      self.FIT_SAMPLE,
                                                      replace=False)]
        return rows, float(np.count_nonzero(off <= 0.25)) / len(off)

    def _measure(self, budget):
        """Which tiles of the pattern the container can reach into."""
        # The container's world box, in cells of the cloud's grid. Taking the
        # eight corners through the transform and keeping their extremes is
        # conservative: a tilted grid can only cover more than the box, never
        # miss part of it.
        corners = np.array([[x, y, z]
                            for x in (self.box_min[0], self.box_max[0])
                            for y in (self.box_min[1], self.box_max[1])
                            for z in (self.box_min[2], self.box_max[2])],
                           dtype=np.float64)
        local = (corners - self.shift) @ np.linalg.inv(self.linear).T
        box = ((local - self.origin) @ self.inverse.T
               - self.row0) / self.row_pitch

        self.k_min = np.floor((box.min(axis=0) - self.lo)
                              / self.period).astype(np.int64)
        self.k_max = np.floor((box.max(axis=0) - self.lo)
                              / self.period).astype(np.int64)
        self.tiles = self.k_max - self.k_min + 1
        self.tile_count = int(np.prod(self.tiles))
        self.candidates = self.tile_count * self.total
        self.waiting = self.candidates > self.CONFIRM_ABOVE

        self.cover_lo = self.lo + self.k_min * self.period
        self.cover_hi = self.hi + self.k_max * self.period
        self.width = self.cover_hi - self.cover_lo + 1
        # Cast along whichever grid direction the fill is widest in, because
        # that is the one with the fewest lines to cast along.
        self.axis = int(np.argmax(self.width))
        self.other = [axis for axis in range(3) if axis != self.axis]
        return True

    def _lines(self, budget):
        """Every grid line the finished fill can put a point on.

        A line is named by its two row numbers across the cast direction,
        packed into one integer so the whole set can be sorted and searched
        at once. The lines of a tile are the lines of the source block put
        through that tile's mirror and step, which is why this can be worked
        out before a single point is placed.
        """
        across, along = self.other
        pairs, which = np.unique(self.rows[:, (across, along)], axis=0,
                                 return_inverse=True)
        which = which.ravel()

        # Where the points of each line really sit across it. A row number
        # names a line, but the points on it sit wherever they sit, and a ray
        # down the middle of them answers the line far better than one down
        # the nominal row: half a row of error there is enough to answer a
        # whole line of points wrongly against a surface that runs alongside
        # it.
        held = np.bincount(which, minlength=len(pairs)).astype(np.float64)
        drift = []
        for axis in (across, along):
            apart = ((self.u[:, axis] - self.row0[axis])
                     / self.row_pitch[axis]) - self.rows[:, axis]
            drift.append(np.bincount(which, weights=apart,
                                     minlength=len(pairs)) / held)

        keys, slips, slides = [], [], []
        for first in range(int(self.k_min[across]), int(self.k_max[across]) + 1):
            one, shift = mirror_tile_step(first, int(self.lo[across]),
                                          int(self.hi[across]),
                                          int(self.seam[across]))
            rows = one * pairs[:, 0] + shift - self.cover_lo[across]
            for second in range(int(self.k_min[along]),
                                int(self.k_max[along]) + 1):
                two, slide = mirror_tile_step(second, int(self.lo[along]),
                                              int(self.hi[along]),
                                              int(self.seam[along]))
                columns = two * pairs[:, 1] + slide - self.cover_lo[along]
                keys.append(rows * int(self.width[along]) + columns)
                slips.append(one * drift[0])
                slides.append(two * drift[1])

        keys = np.concatenate(keys)
        self.line_keys, where = np.unique(keys, return_index=True)
        self.line_slip = np.concatenate(slips)[where]
        self.line_slide = np.concatenate(slides)[where]

        # One monotone coordinate per line and position along it, so every
        # line's crossings can live in a single sorted array and a whole tile
        # can be answered with one search of it. Kept in cells, which is what
        # a ray reads, with the margins a row wide rather than a cell.
        step = float(self.row_pitch[self.axis])
        self.first = (float(self.row0[self.axis])
                      + (float(self.cover_lo[self.axis]) - 0.5) * step)
        self.x_base = self.first - 2.0 * step
        self.key_span = (float(self.width[self.axis]) + 6.0) * step
        self.crossings = []
        self.cursor = 0
        return True

    def _cast(self, budget):
        """Where the surface crosses each of those lines."""
        total = len(self.line_keys)
        across, along = self.other
        stride = self.linear @ self.basis[:, self.axis]
        length = float(np.linalg.norm(stride))
        if length < 1e-12:
            raise ValueError("The grid step along the widest direction has "
                             "no length in the world")
        unit = stride / length
        heading = Vector(unit)

        # A ray that runs in the plane of a face grazes its edges and
        # miscounts, and a grid line is very often exactly in such a plane,
        # since a container is usually built to the same axes as the cloud.
        # Every ray is therefore nudged a few ten thousandths of a cell off
        # its line, which moves the crossing it reads by far less than the
        # spacing between two points but takes it out of the plane.
        #
        # A nudge has a side to it, though, and a line that lies in the plane
        # of a face is pushed clean off the volume by the wrong side, which
        # would drop that layer of points while the layer on the opposite
        # face is kept. So all four corners of the nudge are tried before a
        # line is believed to miss the volume: a line that really does miss
        # finds nothing in all four, which costs four of the cheapest query
        # a BVH has, and a line through the volume is answered by the first.
        sideways = (self.linear @ self.basis[:, across]
                    * self.row_pitch[across]) * 4.1e-4
        onwards = (self.linear @ self.basis[:, along]
                   * self.row_pitch[along]) * 2.7e-4
        nudges = (sideways + onwards, -sideways - onwards,
                  sideways - onwards, -sideways + onwards)
        width = int(self.width[along])
        first = self.first
        deadline = time.time() + budget

        while self.cursor < total:
            key = int(self.line_keys[self.cursor])
            cell = np.empty(3, dtype=np.float64)
            cell[self.axis] = first
            cell[across] = (self.row0[across]
                            + (self.cover_lo[across] + key // width
                               + self.line_slip[self.cursor])
                            * self.row_pitch[across])
            cell[along] = (self.row0[along]
                           + (self.cover_lo[along] + key % width
                              + self.line_slide[self.cursor])
                           * self.row_pitch[along])
            base = (self.linear @ (self.origin + self.basis @ cell)
                    + self.shift)

            for nudge in nudges:
                found, capped = self._walk(base + nudge, heading, unit,
                                           length, first)
                if found:
                    break
            if capped:
                self.capped += 1
            self.crossings.append(np.array(found, dtype=np.float64))

            self.cursor += 1
            if (self.cursor & 31) == 0 and time.time() >= deadline:
                break

        self.fraction = self.cursor / total if total else 1.0
        if self.cursor < total:
            return False

        counts = np.array([len(found) for found in self.crossings],
                          dtype=np.int64)
        self.ends = np.cumsum(counts)
        flat = (np.concatenate(self.crossings) if len(self.crossings)
                else np.zeros(0, dtype=np.float64))

        # A container built to the same axes as the cloud, which is the usual
        # case, has faces that lie exactly on planes of points, and such a
        # point is neither in nor out. Widening every inside span by a ten
        # thousandth of a cell at both ends settles those points as inside,
        # and settles them the same way on both faces: without it the layer
        # on the face the surface is entered through is kept and the layer on
        # the face it leaves through is dropped, which leaves the fill one
        # pitch short along one side of the volume. Crossings alternate into
        # and out of the volume along a line, which is what says which end of
        # a span each one is.
        starts = self.ends - counts
        within = (np.arange(len(flat), dtype=np.int64)
                  - np.repeat(starts, counts))
        edge = self.BOUNDARY * float(self.row_pitch[self.axis])
        flat = flat + np.where(within % 2 == 0, -edge, edge)

        # Clipped into its own line's band before packing, so a crossing that
        # lands just past the covered range can never read as a crossing of
        # the line after it.
        flat = np.clip(flat, self.x_base, self.x_base + self.key_span - 1e-6)
        lines = np.repeat(np.arange(len(self.crossings), dtype=np.float64),
                          counts)
        self.flat = np.sort(lines * self.key_span + (flat - self.x_base))
        self.crossings = None
        return True

    def _walk(self, start, heading, unit, length, first):
        """Every crossing along one line, in the order they are met.

        Returns the crossings in cells from the start of the covered range,
        and whether the walk ran into the crossing limit, which a surface
        that is not closed can do.
        """
        return ray_crossings(self.bvh, start, heading, unit, length, first,
                             self.eps, self.MAX_HITS)

    def _fill(self, budget):
        """Lay out the tiles, keeping the points that land inside.

        Without trimming every point of every tile is kept, which covers the
        box of the volume in whole tiles.
        """
        across, along = self.other
        dims = tuple(int(count) for count in self.tiles)
        per_tile = max(1, self.VERIFY // max(self.tile_count, 1))
        rng = np.random.default_rng(1)
        deadline = time.time() + budget

        while self.tile_cursor < self.tile_count:
            index = np.array(np.unravel_index(self.tile_cursor, dims),
                             dtype=np.int64) + self.k_min
            scale, offset, shifts = tile_affine(index, self.lo, self.hi,
                                                self.seam, self.row0,
                                                self.row_pitch)

            if np.all(scale > 0.0):
                # A plain step, applied to the points as they are, so an
                # unmirrored tile is exact and tile zero is the cloud itself.
                moved = self.coords + self.basis @ offset
            else:
                turn = self.basis @ np.diag(scale) @ self.inverse
                moved = ((self.coords - self.origin) @ turn.T + self.origin
                         + self.basis @ offset)
            rows = self.rows * scale.astype(np.int64) + shifts

            if not self.trim:
                keep = np.ones(self.total, dtype=bool)
                if np.any(self.through):
                    keep &= ~tile_shared_rows(index, self.rows, self.lo,
                                              self.hi, self.seam)
                self.chunks.append((moved[keep].astype(np.float32),
                                    self.source_index[keep]))
                self.kept += int(np.count_nonzero(keep))
                self.tile_cursor += 1
                self.fraction = self.tile_cursor / self.tile_count
                if time.time() >= deadline:
                    break
                continue

            key = ((rows[:, across] - self.cover_lo[across])
                   * int(self.width[along])
                   + (rows[:, along] - self.cover_lo[along]))
            line = np.searchsorted(self.line_keys, key)
            known = line < len(self.line_keys)
            known &= self.line_keys[np.minimum(
                line, len(self.line_keys) - 1)] == key

            probe = (line * self.key_span
                     + (self.u[:, self.axis] * scale[self.axis]
                        + offset[self.axis] - self.x_base))
            ahead = self.ends[np.minimum(line, len(self.ends) - 1)] \
                - np.searchsorted(self.flat, probe, side='right')
            decided = ((ahead % 2) == 1) & known
            self.stray += int(np.count_nonzero(~known))
            keep = decided
            if np.any(self.through):
                # A seam mirrored through an end row shares it with the tile
                # beside, which writes it instead.
                keep = decided & ~tile_shared_rows(index, self.rows, self.lo,
                                                   self.hi, self.seam)

            if np.any(keep):
                self.chunks.append((moved[keep].astype(np.float32),
                                    self.source_index[keep]))
                self.kept += int(np.count_nonzero(keep))
            if len(self.probes) < self.VERIFY:
                for i in rng.choice(self.total, min(per_tile, self.total),
                                    replace=False):
                    self.probes.append((self.linear @ moved[i] + self.shift,
                                        bool(decided[i])))

            self.tile_cursor += 1
            self.fraction = self.tile_cursor / self.tile_count
            if time.time() >= deadline:
                break
        return self.tile_cursor >= self.tile_count

    def _verify(self, budget):
        """Second opinion on a sample, from the slow per point vote."""
        deadline = time.time() + budget
        while self.probe_cursor < len(self.probes):
            point, decided = self.probes[self.probe_cursor]
            inside, _ = point_is_inside(self.bvh, Vector(point), self.eps,
                                        self.VERIFY_RAYS)
            self.checked += 1
            if inside == decided:
                self.agreed += 1
            self.probe_cursor += 1
            if (self.probe_cursor & 15) == 0 and time.time() >= deadline:
                break
        self.fraction = (self.probe_cursor / len(self.probes)
                         if self.probes else 1.0)
        return self.probe_cursor >= len(self.probes)

    def _build(self, budget):
        """Write the fill as a copy of the cloud object, geometry replaced."""
        if self.kept == 0:
            raise ValueError(
                f"No point landed inside '{self.container.name}', so nothing "
                f"was built. Check that the mesh really is closed, and that "
                f"it is not inside out")

        positions = np.concatenate([chunk[0] for chunk in self.chunks])
        source = np.concatenate([chunk[1] for chunk in self.chunks])
        self.chunks = None
        self.result, dropped = build_cloud_copy(self.cloud, self.collection,
                                                positions, source, "_filled")
        self.dropped.extend(dropped)
        return True

    # -- reporting ----------------------------------------------------------

    @property
    def agreement(self):
        return self.agreed / self.checked if self.checked else 1.0

    @property
    def doubtful(self):
        spilled = self.outside - self.cropped
        left_out = spilled / (self.total + self.outside) if self.total else 0.0
        return bool((self.open_edges and self.trim) or self.capped
                    or self.stray
                    or self.agreement < 0.95 or left_out > 0.02
                    or self.stalled or self.fit < 0.9)

    @property
    def short(self):
        note = (f"{self.kept:,} points, {self.tiles[0]}x{self.tiles[1]}x"
                f"{self.tiles[2]} tiles of {self.period[0]}x{self.period[1]}"
                f"x{self.period[2]}")
        if self.agreement < 0.95:
            note += f"  CHECK: rays agreed {self.agreement * 100:.0f}%"
        elif self.stalled:
            note += "  CHECK: the tile may be too big"
        return note

    def finish(self):
        candidates = self.tile_count * self.total
        if self.trim:
            notes = [f"Filled '{self.container.name}' with {self.kept:,} of "
                     f"{candidates:,} mirrored points "
                     f"({self.tiles[0]} x {self.tiles[1]} x {self.tiles[2]} "
                     f"tiles of {self.total:,}) as '{self.result.name}', "
                     f"{time.time() - self.start:.1f}s"]
        else:
            notes = [f"Covered '{self.container.name}' with {self.kept:,} "
                     f"mirrored points ({self.tiles[0]} x {self.tiles[1]} x "
                     f"{self.tiles[2]} whole tiles of {self.total:,}), not "
                     f"trimmed to its shape, as '{self.result.name}', "
                     f"{time.time() - self.start:.1f}s"]
        notes.append(f"the block that repeats is {self.period[0]} x "
                     f"{self.period[1]} x {self.period[2]} points"
                     + (f" as set in Tile, measured as {self.measured[0]} x "
                        f"{self.measured[1]} x {self.measured[2]}"
                        if np.any(self.asked > 0)
                        and not np.array_equal(self.period, self.measured)
                        else "") + ", seams "
                     + " / ".join("through the end layer" if self.through[a]
                                  else f"{self.row_step[a]} row"
                                  f"{'' if self.row_step[a] == 1 else 's'}"
                                  for a in range(3)))
        if self.framed:
            notes.append("tiled on grid steps along the cloud's own X, Y and "
                         "Z rather than the ones Analyze Lattice found, "
                         "because the cloud is a box in those")
        if not np.allclose(self.row_pitch, 1.0, atol=1e-3):
            notes.append(f"its rows sit {self.row_pitch[0]:.4g} / "
                         f"{self.row_pitch[1]:.4g} / {self.row_pitch[2]:.4g} "
                         f"grid steps apart, so a seam one grid step wide "
                         f"would have left a gap")
        if self.cropped:
            notes.append(f"{self.cropped:,} point"
                         f"{'' if self.cropped == 1 else 's'} in the corners "
                         f"of the block were left out, because the cloud's "
                         f"faces do not run along the grid and only the box "
                         f"of rows it fills can be mirrored without a gap")
        spilled = self.outside - self.cropped
        if spilled:
            notes.append(f"{spilled:,} point"
                         f"{'' if spilled == 1 else 's'} outside that "
                         f"block were left out, so no seam has a gap in it")
        if self.fill < 0.9:
            notes.append(f"the cloud fills only {self.fill * 100:.0f}% of "
                         f"its tile; if it is meant to be solid, the seams "
                         f"will show gaps where it does not reach")
        if self.fit < 0.9:
            notes.append(f"only {self.fit * 100:.0f}% of the cloud sits "
                         f"squarely in rows along the grid, so a seam may "
                         f"wander by a fraction of a row")
        if self.stalled:
            notes.append("the block still ends in a layer far thinner than "
                         "the body, which there were too many points in to "
                         "trim away, so a seam may still show: clear the "
                         "strays off this cloud with Select Noise and Trash "
                         "first")
        if self.checked:
            notes.append(f"the line test agreed with the ray vote on "
                         f"{self.agreed} of {self.checked} sampled points")
        if self.open_edges and self.trim:
            notes.append(f"'{self.container.name}' is not watertight "
                         f"({self.open_edges} boundary edges)")
        if self.capped:
            notes.append(f"{self.capped:,} lines hit the crossing limit")
        if self.stray:
            notes.append(f"{self.stray:,} points fell on no cast line")
        if self.dropped:
            notes.append("dropped unsupported attributes: "
                         + ", ".join(sorted(set(self.dropped))))
        return "; ".join(notes)


class MESH_OT_FillVolume(bpy.types.Operator):
    bl_idname = "mesh.pc_fill_volume"
    bl_label = "Fill Volume"
    bl_description = (
        "Step 2: Fills the picked mesh with mirrored copies of the picked "
        "point cloud, and links the result beside them as a new object.\n\n"
        "The copies are laid out on the grid Analyze Lattice found at the "
        "top of Modify, so the fill carries the pitch of the cloud exactly and "
        "the cloud itself does not move: every second copy along a grid "
        "direction is mirrored, and a seam falls so that the mirrored row "
        "lands exactly one point spacing from the row it was mirrored off, "
        "which leaves neither a gap nor a doubled point. What a row is, and "
        "how many of them repeat, are both measured from the cloud first: a "
        "grid step is not always one row of points, and the strays on a "
        "cloud, or the points whose jitter carries them past its last row, "
        "are not part of the block either. The result says what it measured "
        "and how many points it left out, and Tile overrides it. Points "
        "outside the mesh are "
        "dropped, so the fill follows the real shape, and one that lands "
        "exactly on the surface is kept. Both picked objects are left "
        "untouched, and the result keeps the attributes, materials and "
        "modifiers of the cloud. Set Mode to Cover Only to skip the "
        "trimming and get every point of every tile that reaches into the "
        "mesh's box, to cut with a Boolean yourself. A fill of more than "
        "100 million points "
        "asks before it goes on. Press Cancel, or Esc, while it runs to stop it"
    )
    bl_options = {'REGISTER', 'UNDO'}

    allow_large: bpy.props.BoolProperty(
        name="Allow Large Fill", default=False, options={'SKIP_SAVE'},
        description="Go on with a fill of more than 100 million points "
                    "without asking first. Only for scripts: pressed in the "
                    "sidebar, a big fill always asks")

    _timer = None
    _job = None

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return (scene.pc_fill_volume is not None
                and scene.pc_fill_cloud is not None
                and not scene.pc_fill_running)

    def _cleanup(self, context):
        scene = context.scene
        scene.pc_fill_running = False
        scene.pc_fill_progress = 0.0
        scene.pc_fill_stage = ""
        FILL_PROMPT.update(points=0, note="", answer=None)
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._job = None

    def _announce(self, context, job, message):
        scene = context.scene
        # The tile it worked with is shown in the status, not written back
        # into Tile: a value there overrides the measurement, so writing it
        # back would freeze the first measurement into every fill after it,
        # of this cloud or any other.
        scene.pc_fill_status = job.short
        self.report({'WARNING' if job.doubtful else 'INFO'}, message)

    def execute(self, context):
        """Blocking path: used from scripts and in background mode."""
        scene = context.scene
        try:
            job = FillVolumeJob(context)
            while not job.step(1.0):
                # A script cannot be asked, so a big fill needs allow_large
                if job.waiting:
                    if not self.allow_large:
                        raise ValueError(
                            f"{job.big_note}. Run it with allow_large=True "
                            f"to place that many")
                    job.go_ahead()
            message = job.finish()
        except ValueError as exc:
            scene.pc_fill_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self._announce(context, job, message)
        return {'FINISHED'}

    def invoke(self, context, event):
        """Modal path: used when the button is pressed in the sidebar."""
        if getattr(context, "window", None) is None:
            return self.execute(context)

        scene = context.scene
        try:
            self._job = FillVolumeJob(context)
        except ValueError as exc:
            scene.pc_fill_status = ""
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.pc_fill_running = True
        scene.pc_fill_progress = 0.0
        scene.pc_fill_stage = self._job.stage_name
        scene.pc_fill_status = ""
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        CANCEL_REQUESTS.discard("fill")
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        if event.type == 'ESC' or take_cancel("fill"):
            self._cleanup(context)
            scene.pc_fill_status = "Cancelled"
            tag_redraw_sidebar(context)
            self.report({'WARNING'}, "Fill cancelled, nothing was built")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        job = self._job
        if job.waiting:
            # Held after measuring: ask once, then wait for the answer from
            # the popup or from the buttons in the panel.
            if FILL_PROMPT["points"] == 0:
                FILL_PROMPT.update(points=job.candidates, note=job.big_note,
                                   answer=None)
                tag_redraw_sidebar(context)
                bpy.ops.mesh.pc_fill_confirm('INVOKE_DEFAULT')
            answer = FILL_PROMPT["answer"]
            if answer is None:
                return {'RUNNING_MODAL'}
            if not answer:
                self._cleanup(context)
                scene.pc_fill_status = "Cancelled"
                tag_redraw_sidebar(context)
                self.report({'WARNING'}, "Fill cancelled, nothing was built")
                return {'CANCELLED'}
            FILL_PROMPT.update(points=0, note="", answer=None)
            job.go_ahead()

        try:
            done = job.step(0.08)
        except ValueError as exc:
            self._cleanup(context)
            scene.pc_fill_status = ""
            tag_redraw_sidebar(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.pc_fill_progress = self._job.progress
        scene.pc_fill_stage = self._job.stage_name
        tag_redraw_sidebar(context)

        if not done:
            return {'RUNNING_MODAL'}

        message = job.finish()
        self._cleanup(context)
        self._announce(context, job, message)
        tag_redraw_sidebar(context)
        return {'FINISHED'}


# The question a running fill, or a running Invert, is waiting on. Module
# level rather than on the scene, because a point count can be past what an
# IntProperty holds.
FILL_PROMPT = {"points": 0, "note": "", "answer": None}


class MESH_OT_FillConfirm(bpy.types.Operator):
    bl_idname = "mesh.pc_fill_confirm"
    bl_label = "Confirm Large Fill"
    bl_description = ("Answers a fill or an Invert of more than 100 "
                      "million points that is waiting to be confirmed")
    bl_options = {'INTERNAL'}

    accept: bpy.props.BoolProperty(default=True, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return FILL_PROMPT["points"] > 0 and FILL_PROMPT["answer"] is None

    def invoke(self, context, event):
        wm = context.window_manager
        try:
            return wm.invoke_confirm(
                self, event, title=f"Place {FILL_PROMPT['points']:,} points?",
                message=(f"{FILL_PROMPT['note']}. That can take a long time "
                         f"and a lot of memory"),
                confirm_text="Place Them", icon='WARNING')
        except TypeError:
            # Before Blender 4.1 the popup takes no text of its own
            return wm.invoke_confirm(self, event)

    def execute(self, context):
        FILL_PROMPT["answer"] = self.accept
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

class PC_ClusterSelect(PC_TaskOperator):
    """Select Noise and Select Cluster, which differ in which ids they take."""

    noise = True

    @property
    def task_key(self):
        return f"{tool_key(self.bl_idname)}:{self.attr_target}"

    def work(self):
        yield "Reading the clusters", 0.0
        context = self.context
        obj = task_mesh(context)
        object_mode(obj)
        mesh = obj.data
        attr = mesh.attributes.get(self.attr_target)
        if attr is None or attr.domain != 'POINT' or attr.data_type != 'INT':
            bpy.ops.object.mode_set(mode='EDIT')
            raise TaskError(f"No '{self.attr_target}' on '{obj.name}' yet, "
                            f"press Cluster first", 'WARNING')
        ids = np.empty(len(mesh.vertices), dtype=np.int32)
        attr.data.foreach_get("value", ids)
        mask = (ids == -1) if self.noise else (ids != -1)
        select_points(mesh, mask)
        context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode='EDIT')
        return (f"Selected {int(np.count_nonzero(mask)):,} of {len(ids):,} "
                f"points")


class MESH_OT_SelectNoise(PC_ClusterSelect, bpy.types.Operator):
    bl_idname = "mesh.pc_select_noise"
    bl_label = "Select Noise"
    bl_description = "Step 3a: Selects unclustered/noise vertices (those with cluster ID = -1)"
    attr_target: bpy.props.StringProperty()
    noise = True


class MESH_OT_SelectCluster(PC_ClusterSelect, bpy.types.Operator):
    bl_idname = "mesh.pc_select_cluster"
    bl_label = "Select Cluster"
    bl_description = "Step 5a: Selects all clustered vertices (those with cluster ID != -1)"
    attr_target: bpy.props.StringProperty()
    noise = False


class MESH_OT_PickColor(PC_TaskOperator, bpy.types.Operator):
    bl_idname = "mesh.pick_color_selected"
    bl_label = "Sample"
    bl_description = "Step 2: Averages color from selected vertices and sets as reference color"

    def work(self):
        yield "Reading the selection", 0.0
        context = self.context
        obj = task_mesh(context)
        object_mode(obj)
        try:
            mesh = obj.data
            chosen = np.zeros(len(mesh.vertices), dtype=bool)
            mesh.vertices.foreach_get("select", chosen)
            if not chosen.any():
                raise TaskError("No vertices selected", 'WARNING')
            rgb = read_point_colors(mesh, context.scene.pc_color_attr_enum)
            average = rgb[chosen].astype(np.float64).mean(axis=0)
        finally:
            bpy.ops.object.mode_set(mode='EDIT')
        context.scene.pc_ref_color = (float(average[0]), float(average[1]),
                                      float(average[2]), 1.0)
        return (f"Reference color sampled from "
                f"{int(np.count_nonzero(chosen)):,} points")


class PC_ColorTask(PC_TaskOperator):
    """Reads the colors of the active cloud for the color tools."""

    def colors(self):
        context = self.context
        scene = context.scene
        obj = task_mesh(context)
        object_mode(obj)
        rgb = read_point_colors(obj.data, scene.pc_color_attr_enum)
        return (obj, rgb, reference_hsv(scene), float(scene.pc_color_tol),
                bool(scene.pc_material_mode))


class MESH_OT_Heatmap(PC_ColorTask, bpy.types.Operator):
    bl_idname = "mesh.pc_heatmap"
    bl_label = "Heatmap"
    bl_description = "Creates a color attribute showing matches in green and non-matches in blue (requires attribute display)"

    def work(self):
        yield "Reading the colors", 0.0
        obj, rgb, ref, tol, material = self.colors()
        match = yield from color_steps(
            rgb, lambda part: color_matches(part, ref, tol, material),
            "Comparing colors")
        still_same(obj, len(rgb))
        heat = np.zeros((len(rgb), 4), dtype=np.float32)
        heat[:, 0] = match
        heat[:, 2] = 1.0 - match
        heat[:, 3] = 1.0
        mesh = obj.data
        point_attribute(mesh, "pc_heatmap", 'FLOAT_COLOR').data.foreach_set(
            "color", heat.ravel())
        mesh.update()
        return ("Heatmap created: 'pc_heatmap'. Switch viewport to attribute "
                "display mode to visualize.")


class MESH_OT_AnalyzeColors(PC_ColorTask, bpy.types.Operator):
    bl_idname = "mesh.pc_analyze_colors"
    bl_label = "Analyze Colors"
    bl_description = "Creates a numeric attribute (0-1) showing how well each point matches the reference color"

    def work(self):
        yield "Reading the colors", 0.0
        obj, rgb, ref, tol, material = self.colors()
        score = yield from color_steps(
            rgb, lambda part: color_scores(part, ref, tol, material),
            "Scoring colors")
        still_same(obj, len(rgb))
        mesh = obj.data
        point_attribute(mesh, "pc_color_score", 'FLOAT').data.foreach_set(
            "value", score.astype(np.float32))
        mesh.update()
        return ("Color analysis complete: 'pc_color_score' attribute created "
                "(0=no match, 1=perfect match)")


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


class MESH_OT_SplitClusters(PC_TaskOperator, bpy.types.Operator):
    bl_idname = "mesh.pc_split_clusters"
    bl_label = "Split"
    bl_description = ("Step 5c: Moves the selected vertices into a new "
                      "object, keeping their colors and other attributes and "
                      "the transform of the cloud")
    attr_target: bpy.props.StringProperty()

    @property
    def task_key(self):
        return f"{tool_key(self.bl_idname)}:{self.attr_target}"

    def work(self):
        yield "Reading the selection", 0.0
        obj = task_mesh(self.context)
        object_mode(obj)
        chosen = np.zeros(len(obj.data.vertices), dtype=bool)
        obj.data.vertices.foreach_get("select", chosen)
        count = int(np.count_nonzero(chosen))
        if count == 0:
            bpy.ops.object.mode_set(mode='EDIT')
            raise TaskError("No vertices selected", 'WARNING')

        # Blender's own Separate does the move in one call, attributes and
        # all, which no loop over the points here could come near.
        before = set(bpy.data.objects)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='SELECTED')
        bpy.ops.object.mode_set(mode='OBJECT')
        made = [other for other in bpy.data.objects if other not in before]
        for other in made:
            other.name = f"{obj.name}_split"
            other.data.name = f"{obj.name}_split"
        return f"Split {count:,} vertices to new object"


class MESH_OT_ClusterDist(PC_TaskOperator, bpy.types.Operator):
    bl_idname = "mesh.cluster_dist"
    bl_label = "Spatial Cluster"
    bl_description = "Step 2: Clusters vertices by spatial proximity within the radius, groups minimum points together. Press Cancel, or Esc, while it runs to stop it"

    def work(self):
        yield "Reading the points", 0.0
        scene = self.context.scene
        radius, least = float(scene.pc_radius), int(scene.pc_min_points)
        obj = task_mesh(self.context)
        mode = obj.mode
        object_mode(obj)
        try:
            mesh = obj.data
            total = len(mesh.vertices)
            if total == 0:
                raise TaskError(f"'{obj.name}' has no points")
            flat = np.empty(total * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", flat)
            coords = flat.reshape(-1, 3).tolist()

            tree = KDTree(total)
            for i, co in enumerate(coords):
                tree.insert(co, i)
                if (i & 16383) == 16383:
                    yield "Sorting the points", 0.15 * i / total
            tree.balance()

            # Grown out from each point not yet reached, neighbour by
            # neighbour, so a cluster is everything joined by steps no
            # longer than Radius.
            labels = np.full(total, -1, dtype=np.int32)
            reached = bytearray(total)
            clusters = done = 0
            for i in range(total):
                if reached[i]:
                    continue
                reached[i] = 1
                queue, members = deque([i]), []
                while queue:
                    index = queue.popleft()
                    members.append(index)
                    for _, near, _ in tree.find_range(coords[index], radius):
                        if not reached[near]:
                            reached[near] = 1
                            queue.append(near)
                    done += 1
                    if (done & 4095) == 0:
                        yield "Growing clusters", 0.15 + 0.8 * done / total
                if len(members) >= least:
                    labels[members] = clusters
                    clusters += 1

            still_same(obj, total)
            point_attribute(mesh, "cluster_id_dist", 'INT').data.foreach_set(
                "value", labels)
            mesh.update()
        finally:
            put_mode_back(obj, mode)
        noise = int(np.count_nonzero(labels < 0))
        return (f"Found {clusters:,} clusters of {least} points or more, "
                f"{noise:,} points left as noise")


class MESH_OT_ClusterColor(PC_ColorTask, bpy.types.Operator):
    bl_idname = "mesh.cluster_color"
    bl_label = "Cluster by Color"
    bl_description = "Step 4b: Groups vertices matching the reference color within tolerance (respects material mode)"

    def work(self):
        yield "Reading the colors", 0.0
        obj, rgb, ref, tol, material = self.colors()
        match = yield from color_steps(
            rgb, lambda part: color_matches(part, ref, tol, material),
            "Comparing colors")
        still_same(obj, len(rgb))
        mesh = obj.data
        point_attribute(mesh, "cluster_id_color", 'INT').data.foreach_set(
            "value", np.where(match, 0, -1).astype(np.int32))
        mesh.update()
        return (f"{int(np.count_nonzero(match)):,} of {len(rgb):,} points "
                f"match the reference color")


class MESH_OT_PreviewColorMatches(PC_ColorTask, bpy.types.Operator):
    bl_idname = "mesh.preview_color_matches"
    bl_label = "Preview Matches"
    bl_description = "Step 4a: Counts and reports how many vertices match the reference color with current settings"
    bl_options = {'REGISTER'}

    def work(self):
        yield "Reading the colors", 0.0
        obj, rgb, ref, tol, material = self.colors()
        match = yield from color_steps(
            rgb, lambda part: color_matches(part, ref, tol, material),
            "Comparing colors")
        return f"Found {int(np.count_nonzero(match)):,} matching vertices"

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
        CANCEL_REQUESTS.discard("boolean")
        wm.modal_handler_add(self)
        tag_redraw_sidebar(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        scene = context.scene

        if event.type == 'ESC' or take_cancel("boolean"):
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
        "result. Press Cancel, or Esc, while it runs to stop it"
    )
    bl_options = {'REGISTER', 'UNDO'}
    operation = 'INTERSECT'

# --- 1h. INVERT LOGIC ---
#
# The empty sites of the cloud's own grid, inside a closed mesh or, with
# none picked, inside the box of grid rows the cloud spans.
#
# A site is a place the grid puts a point. Those are the cells of the grid
# Analyze Lattice found, whenever the points sit on them, which they do when
# its steps are the shortest ones that carry the cloud onto itself. When
# they do not, a centred cloud read on a cubic grid say, the sites are read
# off the points instead: the rows measure_rows finds along each step, and
# of those only the ones the cloud uses, so no point is ever put half way
# between two of its own.
#
# A site is taken when a point of the cloud lies within half a row of it
# along every direction, which is the site nearest that point, so a cloud a
# little off its grid still takes the sites it means to. Every other site
# inside the boundary becomes a point, carrying the color, and every other
# attribute, of the point of the cloud nearest to it. Inside is the same
# one ray per line of sites test Fill Volume makes.

def invert_frame(u, sample=50_000):
    """(base, pitch, rows, fit, read) of the cloud's grid sites.

    read is False when the sites are the grid's own cells, True when they
    had to be read off the points. The grid's cells are fitted to the points
    too, by measure_rows, because a pitch read a hair long by Analyze
    Lattice adds up row after row: a hundredth of a percent is a whole
    hundredth of a pitch by the far side of a hundred rows.
    """
    if len(u) > sample:
        probe = u[np.random.default_rng(0).choice(len(u), sample,
                                                  replace=False)]
    else:
        probe = u

    def fit_of(base, pitch):
        spread = (probe - base) / pitch
        off = np.abs(spread - np.rint(spread)).max(axis=1)
        return float(np.count_nonzero(off <= 0.25)) / max(len(off), 1)

    base, pitch = measure_rows(u)
    fit = fit_of(base, pitch)
    plain_fit = fit_of(np.zeros(3), np.ones(3))
    if plain_fit > fit:
        base, pitch, fit = np.zeros(3), np.ones(3), plain_fit
    # A row a cell apart is the grid's own cell, fitted; anything else, half
    # a cell say, means the points sit between the grid's cells
    read = bool(np.any(np.abs(pitch - 1.0) > 0.05))
    rows = np.rint((u - base) / pitch).astype(np.int64)
    return base, pitch, rows, fit, read


def site_map(u, rows, base, pitch, origin, basis, sample=500_000,
             weight=1e-6):
    """Where each site sits in the cloud's own space: start + rows @ steps.

    Fitted to the points by least squares, spacing, direction and offset all
    at once, starting from the rows measure_rows read. Only the points that
    sit squarely on a site are fitted, so strays do not pull it about, and
    the fit is held, very lightly, to the rows as read, so a cloud one row
    thick, which says nothing about the direction across it, keeps that
    direction as it was.
    """
    spread = (u - base) / pitch
    near = np.nonzero(np.abs(spread - rows).max(axis=1) <= 0.25)[0]
    if len(near) > sample:
        near = np.random.default_rng(0).choice(near, sample, replace=False)
    fitted = np.hstack([rows[near].astype(np.float64),
                        np.ones((len(near), 1))])
    default = np.vstack([np.diag(pitch), base])
    hold = weight * max(len(near), 1)
    solved = np.linalg.solve(fitted.T @ fitted + hold * np.eye(4),
                             fitted.T @ u[near] + hold * default)
    return origin + solved[3] @ basis.T, solved[:3] @ basis.T


def used_classes(rows, least=0.2):
    """Which classes of rows, odd or even along each direction, hold points.

    Read off a cloud whose sites are the points' own rows, where a centred
    or staggered stacking only fills some of them: its sites are the
    classes it uses. A direction too short to tell, under three rows, is not
    split. Returns (anchor, split, used): the row the classes count from,
    which directions are split, and a 2 x 2 x 2 table of the classes used.
    """
    anchor = rows.min(axis=0)
    split = (rows.max(axis=0) - anchor) >= 2
    parity = ((rows - anchor) % 2) * split
    counts = np.zeros((2, 2, 2), dtype=np.int64)
    np.add.at(counts, (parity[:, 0], parity[:, 1], parity[:, 2]), 1)
    return anchor, split, counts >= least * counts.max()


class MESH_OT_BooleanInvert(PC_TaskOperator, bpy.types.Operator):
    bl_idname = "mesh.pc_boolean_invert"
    bl_label = "Invert"
    bl_description = (
        "Step 3c: The inverse of the point cloud. Fills the closed mesh with "
        "points on the cloud's grid, the one Analyze Lattice found, and "
        "keeps only those that land in a void: every site no point of the "
        "cloud sits within half a pitch of.\n\n"
        "With no closed mesh picked, the box of grid rows the cloud spans is "
        "filled instead. The result is a new object beside the cloud, every "
        "point carrying the color and other attributes of the cloud's point "
        "nearest it; the cloud and the mesh are left untouched. Holes of a "
        "single point count too. Press Cancel, or Esc, while it runs to stop "
        "it"
    )

    CONFIRM_ABOVE = 100_000_000
    MAX_HITS = 2000
    # How far outwards each inside span is widened, in rows, so a site lying
    # exactly on the surface counts as inside: see FillVolumeJob._cast
    BOUNDARY = 1e-4
    LINES = 256

    allow_large: bpy.props.BoolProperty(
        name="Allow Large Invert", default=False, options={'SKIP_SAVE'},
        description="Go on with an Invert of more than 100 million sites "
                    "without asking first. Only for scripts: pressed in the "
                    "sidebar, a big Invert always asks")

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return (scene.pc_bool_cloud is not None
                and not scene.pc_bool_running and not TASKS)

    def work(self):
        context = self.context
        scene = context.scene
        cloud, container = scene.pc_bool_cloud, scene.pc_bool_container
        if cloud is None:
            raise TaskError("Pick a point cloud in the Boolean panel")
        if cloud.type != 'MESH':
            raise TaskError(f"'{cloud.name}' is not a mesh")
        if container is not None:
            if container == cloud:
                raise TaskError("The closed mesh and the point cloud must be "
                                "two different objects")
            if container.type != 'MESH' or len(container.data.polygons) == 0:
                raise TaskError(f"'{container.name}' has no faces, so it "
                                f"cannot bound a volume")
        total = len(cloud.data.vertices)
        if total == 0:
            raise TaskError(f"'{cloud.name}' has no points")
        try:
            basis, origin = scene_lattice(scene)
        except ValueError as exc:
            raise TaskError(str(exc))
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bvh, open_edges = None, 0
        if container is not None:
            # Built now, while the depsgraph handed over is current
            try:
                bvh, box_min, box_max, open_edges = build_container_bvh(
                    container, context.evaluated_depsgraph_get())
            except RuntimeError as exc:
                raise TaskError(str(exc))
        matrix = np.array(cloud.matrix_world, dtype=np.float64)
        linear, shift = matrix[:3, :3], matrix[:3, 3]
        if abs(np.linalg.det(linear)) < 1e-12:
            raise TaskError(f"'{cloud.name}' has a flattened transform")
        collection = context.collection
        scene.pc_bool_status = ""
        start = time.time()
        yield "Reading the cloud", 0.02

        flat = np.empty(total * 3, dtype=np.float32)
        cloud.data.vertices.foreach_get('co', flat)
        coords = flat.reshape(-1, 3).astype(np.float64)
        inverse = np.linalg.inv(basis)
        u = (coords - origin) @ inverse.T
        base, pitch, rows, fit, read = invert_frame(u)
        start_at, steps = site_map(u, rows, base, pitch, origin, basis)
        if fit < 0.5:
            raise TaskError(
                f"Only {fit * 100:.0f}% of '{cloud.name}' sits on the grid "
                f"Analyze Lattice found, so its empty sites cannot be told "
                f"apart. Make this cloud the active object and press Analyze "
                f"Lattice; if the fit it reports is poor, this cloud does not "
                f"sit on one grid")
        if read:
            anchor, split, used = used_classes(rows)
            restrict = not used.all()
        else:
            restrict = False
        yield "Reading the cloud", 0.05

        if bvh is None:
            lo, hi = rows.min(axis=0), rows.max(axis=0)
        else:
            corners = np.array([[x, y, z] for x in (box_min[0], box_max[0])
                                for y in (box_min[1], box_max[1])
                                for z in (box_min[2], box_max[2])],
                               dtype=np.float64)
            local = (corners - shift) @ np.linalg.inv(linear).T
            box = (local - start_at) @ np.linalg.inv(steps)
            lo = np.floor(box.min(axis=0)).astype(np.int64)
            hi = np.ceil(box.max(axis=0)).astype(np.int64)
        width = hi - lo + 1
        strides = np.array([width[1] * width[2], width[2], 1], dtype=np.int64)
        sites = int(np.prod(width))
        # How many of the box's rows are sites at all, counted exactly
        if restrict:
            per_axis = []
            for axis in range(3):
                parity = (np.arange(lo[axis], hi[axis] + 1)
                          - anchor[axis]) % 2 * int(split[axis])
                per_axis.append(np.bincount(parity, minlength=2))
            possible = int(sum(per_axis[0][i] * per_axis[1][j] * per_axis[2][k]
                               for i, j, k in zip(*np.nonzero(used))))
        else:
            possible = sites
        if possible > self.CONFIRM_ABOVE:
            yield BigAsk(possible,
                         f"{width[0]} x {width[1]} x {width[2]} grid rows "
                         f"come to {possible:,} sites to fill")

        within = np.all((rows >= lo) & (rows <= hi), axis=1)
        taken = np.unique(((rows[within] - lo) * strides).sum(axis=1))

        def empty(keys):
            """The keys of sites no point has taken, and that are sites."""
            if len(taken):
                at = np.searchsorted(taken, keys)
                keys = keys[taken[np.minimum(at, len(taken) - 1)] != keys]
            if restrict and len(keys):
                r = np.stack([keys // strides[0],
                              (keys // strides[1]) % width[1],
                              keys % width[2]], axis=1) + lo
                p = ((r - anchor) % 2) * split
                keys = keys[used[p[:, 0], p[:, 1], p[:, 2]]]
            return keys

        voids = []
        if bvh is None:
            for begin in range(0, sites, TASK_CHUNK * 4):
                end = min(begin + TASK_CHUNK * 4, sites)
                voids.append(empty(np.arange(begin, end, dtype=np.int64)))
                yield "Finding the voids", 0.05 + 0.45 * end / sites
            capped = 0
        else:
            voids, capped = yield from self._cast(
                bvh, box_min, box_max, linear, shift, start_at, steps, lo,
                hi, width, strides, empty)

        void_keys = (np.concatenate(voids) if voids
                     else np.zeros(0, dtype=np.int64))
        count = len(void_keys)
        if count == 0:
            raise TaskError("No empty site found inside the boundary, so "
                            "nothing was built", 'WARNING')
        r = np.stack([void_keys // strides[0],
                      (void_keys // strides[1]) % width[1],
                      void_keys % width[2]], axis=1) + lo
        positions = start_at + r @ steps

        # The point of the cloud nearest each void lends it its attributes
        tree = KDTree(total)
        for index, co in enumerate(coords.tolist()):
            tree.insert(co, index)
            if (index & 32767) == 32767:
                yield "Matching colors", 0.5 + 0.15 * index / total
        tree.balance()
        source = np.empty(count, dtype=np.int64)
        for index, co in enumerate(positions.tolist()):
            source[index] = tree.find(co)[1]
            if (index & 16383) == 16383:
                yield "Matching colors", 0.65 + 0.3 * index / count

        still_same(cloud, total)
        result, dropped = build_cloud_copy(cloud, collection,
                                           positions.astype(np.float32),
                                           source, "_inverted")
        where = (f"inside '{container.name}'" if container is not None
                 else "inside its box along the grid")
        scene.pc_bool_status = (f"inverted  {count:,} void points, "
                                f"{width[0]}x{width[1]}x{width[2]} rows")
        notes = [f"Inverted '{cloud.name}' {where}: {count:,} empty sites "
                 f"as '{result.name}', {time.time() - start:.1f}s"]
        if read:
            notes.append(f"its points do not sit on the grid's own cells, so "
                         f"the sites were read off its rows, "
                         f"{pitch[0]:.4g} / {pitch[1]:.4g} / {pitch[2]:.4g} "
                         f"cells apart" + (", every second one of which it "
                                           "uses" if restrict else ""))
        if fit < 0.9:
            notes.append(f"only {fit * 100:.0f}% of the cloud sits squarely "
                         f"on a site, so some sites near its points may "
                         f"read as empty")
        if open_edges:
            notes.append(f"'{container.name}' is not watertight "
                         f"({open_edges} boundary edges)")
        if capped:
            notes.append(f"{capped:,} lines hit the crossing limit")
        if dropped:
            notes.append("dropped unsupported attributes: "
                         + ", ".join(sorted(set(dropped))))
        return "; ".join(notes)

    def _cast(self, bvh, box_min, box_max, linear, shift, start_at, steps, lo,
              hi, width, strides, empty):
        """The empty sites inside the mesh, one ray along each line of them.

        Cast along whichever grid direction the box is widest in, which has
        the fewest lines. Every ray is nudged a few ten thousandths of a row
        off its line, all four ways round before a line is taken to miss, so
        a face lying in the plane of a line is neither grazed nor dodged: see
        FillVolumeJob._cast, which this is the same test as.
        """
        axis = int(np.argmax(width))
        across, along = [a for a in range(3) if a != axis]
        stride = linear @ steps[axis]
        length = float(np.linalg.norm(stride))
        if length < 1e-12:
            raise TaskError("The grid step along the widest direction has no "
                            "length in the world")
        unit = stride / length
        heading = Vector(unit)
        sideways = linear @ steps[across] * 4.1e-4
        onwards = linear @ steps[along] * 2.7e-4
        nudges = (sideways + onwards, -sideways - onwards,
                  sideways - onwards, -sideways + onwards)
        eps = max(float(np.linalg.norm(box_max - box_min)) * 1e-6, 1e-7)
        first = float(lo[axis] - 2)
        along_rows = np.arange(lo[axis], hi[axis] + 1, dtype=np.int64)
        offsets = (along_rows - lo[axis]) * strides[axis]

        lines = int(width[across] * width[along])
        voids, batch, capped = [], [], 0
        for line in range(lines):
            ia, ib = divmod(line, int(width[along]))
            r = np.empty(3, dtype=np.float64)
            r[across], r[along], r[axis] = lo[across] + ia, lo[along] + ib, first
            point = linear @ (start_at + r @ steps) + shift
            for nudge in nudges:
                found, hit_cap = ray_crossings(bvh, point + nudge, heading,
                                               unit, length, first, eps,
                                               self.MAX_HITS)
                if found:
                    break
            capped += int(hit_cap)
            if found:
                cross = np.array(found, dtype=np.float64)
                cross += np.where(np.arange(len(cross)) % 2 == 0,
                                  -self.BOUNDARY, self.BOUNDARY)
                inside = (np.searchsorted(cross, along_rows, side='right')
                          % 2 == 1)
                if inside.any():
                    batch.append(ia * strides[across] + ib * strides[along]
                                 + offsets[inside])
            if (line + 1) % self.LINES == 0 or line + 1 == lines:
                if batch:
                    voids.append(empty(np.concatenate(batch)))
                    batch = []
                yield "Casting through the mesh", 0.05 + 0.45 * (line + 1) / lines
        return voids, capped


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
        "result. Press Cancel, or Esc, while it runs to stop it"
    )
    bl_options = {'REGISTER', 'UNDO'}
    operation = 'DIFFERENCE'

# --- 3. UI SYNC & PANEL ---

# Blender keeps the strings of a dynamic enum only as long as Python does
_ATTRIBUTE_ITEMS = []


def get_attributes_enum(self, context):
    """The color attributes of the active cloud.

    Each item is numbered from its name rather than its place in the list:
    Blender stores the number, so with numbers by place a tool that adds an
    attribute, Cluster by Color adding 'cluster_id_color' say, would shift
    the list and leave the choice pointing at another attribute.
    """
    items = []
    obj = context.active_object
    if obj and obj.type == 'MESH':
        for attr in obj.data.attributes:
            if (attr.domain == 'POINT'
                    and attr.data_type in ('FLOAT_COLOR', 'BYTE_COLOR')):
                items.append((attr.name, attr.name, "", 'NONE',
                              zlib.crc32(attr.name.encode()) & 0x7FFFFFFF))
    _ATTRIBUTE_ITEMS[:] = items if items else [("NONE", "None", "", 'NONE', 0)]
    return _ATTRIBUTE_ITEMS

def poll_bool_container(self, obj):
    """Only meshes with faces can act as a closed container."""
    return (obj.type == 'MESH' and len(obj.data.polygons) > 0
            and obj != self.pc_bool_cloud)

def poll_bool_cloud(self, obj):
    """Any mesh can be the point cloud, but not the container itself."""
    return obj.type == 'MESH' and obj != self.pc_bool_container

def poll_fill_volume(self, obj):
    """Only meshes with faces can hold a volume to fill."""
    return (obj.type == 'MESH' and len(obj.data.polygons) > 0
            and obj != self.pc_fill_cloud)


def update_fill_override(self, context):
    """Turned off, the tile goes back to measured, so it starts at 0 again."""
    if not self.pc_fill_tile_override and any(self.pc_fill_tile):
        self.pc_fill_tile = (0, 0, 0)


def poll_fill_cloud(self, obj):
    """Any mesh can be the cloud that is mirrored, but not the volume."""
    return obj.type == 'MESH' and obj != self.pc_fill_volume


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

    def progress(self, layout, tool, text, fraction):
        """How far a running tool is, with a red Cancel button beside it."""
        fraction = min(max(float(fraction), 0.0), 1.0)
        row = layout.row(align=True)
        label = f"{text}  {fraction * 100:.0f}%"
        if hasattr(row, "progress"):
            # Blender 4.0 and later draw a real bar
            row.progress(factor=fraction, type='BAR', text=label)
        else:
            row.label(text=label)
        stop = row.row(align=True)
        stop.alert = True
        stop.operator("mesh.pc_cancel_task", text="",
                      icon='CANCEL').tool = tool

    def ask(self, layout):
        """A big job waiting for an OK. The popup asks too, but it closes
        when the mouse leaves it, so the answer stays here."""
        col = layout.column(align=True)
        col.scale_y = 0.55
        col.alert = True
        col.label(text=f"{FILL_PROMPT['points']:,} points to place, go on?")
        row = layout.row(align=True)
        row.operator_context = 'EXEC_DEFAULT'
        row.operator("mesh.pc_fill_confirm", text="Place Them",
                     icon='CHECKMARK')
        op = row.operator("mesh.pc_fill_confirm", text="Cancel",
                          icon='CANCEL')
        op.accept = False

    def tasks(self, layout, keys, status=True):
        """The progress of whichever of these tools runs, else what the
        last of them said. True while one of them runs."""
        for key in keys:
            if key in TASKS:
                stage, fraction = TASKS[key]
                self.progress(layout, key, stage, fraction)
                return True
        if status:
            said = [key for key in TASK_STATUS if key in keys]
            if said:
                self.note(layout, TASK_STATUS[said[-1]])
        return False


class MESH_PT_ModifyPanel(PC_ToolPanel, bpy.types.Panel):
    """Analyze Lattice sits on the Modify section itself, above the tools.

    Rescale, Growth and Fill Volume all work on the grid it finds, so it is
    the first thing seen on opening the section rather than a step inside
    one of them.
    """

    bl_label = "Modify"
    bl_idname = "MESH_PT_ModifyPanel"
    bl_options = set()

    def draw_header(self, context):
        pass

    def draw(self, context):
        layout, scene = self.layout, context.scene

        layout.operator("mesh.pc_analyze_lattice", text="Analyze Lattice",
                        icon='PLAY')
        col = layout.column(align=True)
        col.prop(scene, "pc_grid_pitch")
        if scene.pc_grid_running:
            self.progress(layout, "lattice", scene.pc_grid_stage,
                          scene.pc_grid_progress)
        else:
            self.note(layout, scene.pc_grid_status)


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
            self.progress(layout, "video", scene.pc_vid_stage,
                          scene.pc_vid_progress)
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

        self.tasks(layout, ("mesh.cluster_dist",
                            "mesh.pc_select_noise:cluster_id_dist",
                            "mesh.pc_split_clusters:cluster_id_dist"))


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
            self.progress(layout, "thickness", scene.pc_thin_stage,
                          scene.pc_thin_progress)
        elif not self.tasks(layout, ("mesh.pc_select_thin",
                                     "mesh.pc_thin_heatmap"), status=False):
            self.note(layout, scene.pc_thin_status)


class MESH_PT_RescalePanel(PC_ToolPanel, bpy.types.Panel):
    bl_label = "Rescale"
    bl_parent_id = "MESH_PT_ModifyPanel"
    bl_idname = "MESH_PT_RescalePanel"
    icon = 'FULLSCREEN_ENTER'

    def draw(self, context):
        layout, scene = self.layout, context.scene

        # The spacing wanted instead of the measured Pitch above, and the
        # press that moves the points onto it.
        col = layout.column(align=True)
        col.prop(scene, "pc_grid_new_pitch")
        self.step(layout, "①").prop(scene, "pc_grid_pitch_uniform")

        factors = pitch_rescale_factors(scene)
        if factors is not None:
            note = " / ".join(f"x{factor:.4g}" for factor in factors)
            if factors.max() - factors.min() > 1e-4 * factors.max():
                note += "   uneven"
            self.note(layout, note)
        self.step(layout, "②").operator("mesh.pc_rescale_pitch",
                                        text="Rescale",
                                        icon='FULLSCREEN_ENTER')
        if not self.tasks(layout, ("mesh.pc_rescale_pitch",), status=False):
            self.note(layout, scene.pc_grid_scale_status)


class MESH_PT_GrowthPanel(PC_ToolPanel, bpy.types.Panel):
    bl_label = "Growth"
    bl_parent_id = "MESH_PT_ModifyPanel"
    bl_idname = "MESH_PT_GrowthPanel"
    icon = 'MOD_ARRAY'

    def draw(self, context):
        layout, scene = self.layout, context.scene

        row = self.step(layout, "①")
        row.prop(scene, "pc_grow_mode", text="")
        self.step(layout).prop(scene, "pc_grow_steps")
        self.step(layout, "②").operator("mesh.pc_grow", text="Grow",
                                             icon='ADD')

        if scene.pc_grow_running:
            self.progress(layout, "grow", scene.pc_grow_stage,
                          scene.pc_grow_progress)
        elif scene.pc_grow_status:
            self.note(layout, scene.pc_grow_status)


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

        self.tasks(layout, ("mesh.pick_color_selected",
                            "mesh.preview_color_matches",
                            "mesh.pc_analyze_colors", "mesh.pc_heatmap",
                            "mesh.cluster_color",
                            "mesh.pc_select_cluster:cluster_id_color",
                            "mesh.pc_split_clusters:cluster_id_color"))


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
        self.step(layout).operator("mesh.pc_boolean_invert", text="Invert",
                                   icon='SELECT_SUBTRACT')
        if scene.pc_bool_container is None:
            self.note(layout, "no mesh: Invert fills the cloud's box")

        invert = "mesh.pc_boolean_invert"
        if scene.pc_bool_running:
            self.progress(layout, "boolean",
                          f"{scene.pc_bool_rate:,.0f} pts/s",
                          scene.pc_bool_progress)
        elif invert in TASKS and FILL_PROMPT["points"]:
            self.ask(layout)
        elif not self.tasks(layout, (invert,), status=False):
            if scene.pc_bool_status:
                self.note(layout, scene.pc_bool_status)
            else:
                self.tasks(layout, (invert,))


class MESH_PT_FillVolumePanel(PC_ToolPanel, bpy.types.Panel):
    bl_label = "Fill Volume"
    bl_parent_id = "MESH_PT_ModifyPanel"
    bl_idname = "MESH_PT_FillVolumePanel"
    icon = 'MOD_MIRROR'

    def draw(self, context):
        layout, scene = self.layout, context.scene

        self.step(layout, "①").prop(scene, "pc_fill_volume")
        self.step(layout).prop(scene, "pc_fill_cloud")
        self.step(layout).prop(scene, "pc_fill_mode", expand=True)

        # A tile typed by hand, overriding the measured one. Only read while
        # the box is ticked, so the tile is measured unless asked otherwise.
        self.step(layout).prop(scene, "pc_fill_tile_override")
        if scene.pc_fill_tile_override:
            col = layout.column(align=True)
            col.prop(scene, "pc_fill_tile")
            if any(scene.pc_fill_tile):
                col = layout.column(align=True)
                col.scale_y = 0.55
                col.alert = True
                col.label(text=f"tile set by hand: {scene.pc_fill_tile[0]} x "
                               f"{scene.pc_fill_tile[1]} x "
                               f"{scene.pc_fill_tile[2]}, 0 measures it")
            else:
                self.note(layout, "0 in a direction measures it from the "
                                  "cloud")

        # The grid the copies are mirrored on comes from Analyze Lattice at
        # the top of Modify, so say which one that is, or that there is none.
        pitch = tuple(scene.pc_grid_pitch)
        if min(pitch) <= 0.0:
            self.note(layout, "No grid yet, press Analyze Lattice above")
        else:
            self.note(layout, f"grid pitch {pitch[0]:.4g} / {pitch[1]:.4g} / "
                              f"{pitch[2]:.4g}")

        # What that adds up to, roughly, before pressing anything
        tiles, points = fill_estimate(scene)
        if tiles:
            col = layout.column(align=True)
            col.scale_y = 0.55
            col.alert = points > 20_000_000
            col.label(text=f"about {tiles[0]} x {tiles[1]} x {tiles[2]} "
                           f"tiles = {points:,} points to place")

        self.step(layout, "②").operator("mesh.pc_fill_volume",
                                             text="Fill Volume", icon='PLAY')

        if scene.pc_fill_running and FILL_PROMPT["points"]:
            self.ask(layout)
        elif scene.pc_fill_running:
            self.progress(layout, "fill", scene.pc_fill_stage,
                          scene.pc_fill_progress)
        else:
            self.note(layout, scene.pc_fill_status)


# --- 4. REGISTRATION ---

classes = (MESH_OT_CancelTask, MESH_OT_InvertSelection, MESH_OT_PickColor, MESH_OT_Heatmap, MESH_OT_AnalyzeColors, MESH_OT_SelectNoise,
           MESH_OT_SelectCluster, MESH_OT_DeleteNoise, MESH_OT_SplitClusters,
           MESH_OT_ClusterDist, MESH_OT_ClusterColor, MESH_OT_PreviewColorMatches,
           MESH_OT_BooleanIntersect, MESH_OT_BooleanDifference,
           MESH_OT_BooleanInvert,
           MESH_OT_Thickness, MESH_OT_SelectThin, MESH_OT_ThinHeatmap,
           MESH_OT_AnalyzeLattice, MESH_OT_RescalePitch, MESH_OT_Grow,
           MESH_OT_FillVolume, MESH_OT_FillConfirm,
           MESH_OT_VideoStack,
           MESH_PT_GeneratePanel, MESH_PT_VideoStackPanel,
           MESH_PT_SelectPanel, MESH_PT_ModifyPanel, MESH_PT_SpatialPanel,
           MESH_PT_ThicknessPanel, MESH_PT_SpectrumPanel,
           # The Modify tools are listed in the order they are drawn in
           MESH_PT_RescalePanel, MESH_PT_GrowthPanel,
           MESH_PT_FillVolumePanel, MESH_PT_BooleanPanel)

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
    bpy.types.Scene.pc_grid_new_pitch = bpy.props.FloatVectorProperty(
        name="New Pitch", size=3, default=(0.0, 0.0, 0.0), min=0.0,
        precision=4, subtype='XYZ', update=update_grid_new_pitch,
        description="Step 1c: The spacing to stretch the cloud onto. Starts "
                    "out as the measured Pitch, so it reads as no change "
                    "until a number is retyped, and Rescale then moves every "
                    "point until the grid really steps this far")
    bpy.types.Scene.pc_grid_pitch_uniform = bpy.props.BoolProperty(
        name="Uniform", default=True,
        description="Step 1c: Keeps the three New Pitch values in the "
                    "proportions they were measured in, so typing one of "
                    "them scales the cloud evenly and the grid keeps its "
                    "shape. Turn off to stretch the three grid directions by "
                    "different amounts")
    bpy.types.Scene.pc_grid_new_pitch_prev = bpy.props.FloatVectorProperty(
        name="New Pitch Before", size=3, default=(0.0, 0.0, 0.0),
        description="New Pitch as it last stood, which is how an edit of one "
                    "of its three values is told from the other two")
    bpy.types.Scene.pc_grid_scale_status = bpy.props.StringProperty(
        name="Rescale Status", default="",
        description="Result of the last rescale")
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
    bpy.types.Scene.pc_fill_volume = bpy.props.PointerProperty(
        name="Volume", type=bpy.types.Object, poll=poll_fill_volume,
        description="Step 1a: The closed mesh to fill. Only meshes that have "
                    "faces are listed. This object is read only, it is never "
                    "modified")
    bpy.types.Scene.pc_fill_cloud = bpy.props.PointerProperty(
        name="Point Cloud", type=bpy.types.Object, poll=poll_fill_cloud,
        description="Step 1b: The point cloud to fill the volume with. "
                    "Mirrored copy after mirrored copy of it is laid out on "
                    "the grid from Analyze Lattice until the volume is "
                    "covered. This object is read only too, the fill is "
                    "written as a new object beside it")
    bpy.types.Scene.pc_fill_mode = bpy.props.EnumProperty(
        name="Mode", default='TRIM',
        items=[('TRIM', "Fill and Trim",
                "Keep only the points that land inside the mesh, so the "
                "fill follows its shape"),
               ('COVER', "Cover Only",
                "Mirror the cloud until the mesh's box is covered in whole "
                "tiles and keep every point, with nothing trimmed, to cut "
                "it with a Boolean yourself afterwards. Faster, since "
                "nothing is tested against the surface")],
        description="Step 1c: Trim the fill to the shape of the mesh, or "
                    "only cover it in mirrored tiles")
    bpy.types.Scene.pc_fill_tile_override = bpy.props.BoolProperty(
        name="Tile Override", default=False, update=update_fill_override,
        description="Type the tile size by hand instead of having it "
                    "measured from the cloud. Left off, the tile is always "
                    "measured, and turning it off sets Tile back to 0, 0, 0")
    bpy.types.Scene.pc_fill_tile = bpy.props.IntVectorProperty(
        name="Tile", size=3, default=(0, 0, 0), min=0, soft_max=4096,
        subtype='XYZ',
        description="Step 1e, with Tile Override ticked: How many grid "
                    "cells of the cloud one tile is, "
                    "along each of the three grid directions, which is how "
                    "far apart the mirrored copies are placed. Left at 0 it "
                    "is measured from the cloud: the block its layers really "
                    "fill, ignoring the points that jitter rounds past the "
                    "last layer and the strays beyond them, either of which "
                    "would stretch the tile and open that much of a gap at "
                    "every seam. The measured tile is shown in the status "
                    "after a fill; type a value here only to override it, "
                    "and set it back to 0 to have it measured again. Points "
                    "that fall outside the tile are left out of the fill"
    )
    bpy.types.Scene.pc_fill_progress = bpy.props.FloatProperty(
        name="Progress", default=0.0, min=0.0, max=1.0,
        description="How far the running fill has got")
    bpy.types.Scene.pc_fill_stage = bpy.props.StringProperty(
        name="Stage", default="",
        description="Which part of the fill is running")
    bpy.types.Scene.pc_fill_status = bpy.props.StringProperty(
        name="Status", default="",
        description="Result of the last fill")
    bpy.types.Scene.pc_fill_running = bpy.props.BoolProperty(
        name="Running", default=False,
        description="True while a fill is in progress")
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
             'pc_grid_pitch', 'pc_grid_new_pitch',
             'pc_grid_pitch_uniform', 'pc_grid_new_pitch_prev',
             'pc_grid_scale_status', 'pc_grid_basis', 'pc_grid_origin',
             'pc_grid_fit', 'pc_grid_progress', 'pc_grid_stage',
             'pc_grid_status', 'pc_grid_running',
             'pc_fill_volume', 'pc_fill_cloud', 'pc_fill_tile',
             'pc_fill_tile_override', 'pc_fill_mode',
             'pc_fill_progress',
             'pc_fill_stage', 'pc_fill_status', 'pc_fill_running',
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
