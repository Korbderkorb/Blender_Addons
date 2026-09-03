bl_info = {
    "name": "Point Cloud Master Suite",
    "author": "Korbinian Enzinger",
    "version": (1, 2, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > PC Tools",
    "description": "Comprehensive toolkit for point cloud clustering, color filtering, and boolean cutting against a closed mesh.",
    "warning": "",
    "category": "Mesh",
}

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

class MESH_PT_MainPanel(bpy.types.Panel):
    bl_label = "Point Cloud Master Suite"
    bl_idname = "MESH_PT_MainPanel"
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'PC Tools'

    def draw(self, context):
        layout, scene, obj = self.layout, context.scene, context.active_object
        box = layout.box()
        box.label(text="Spatial Logic", icon='PHYSICS')

        # Step 1: Configure Parameters
        row = box.row(align=False)
        row.scale_x = 0.12
        row.label(text="①")
        row.scale_x = 1.0
        row.prop(scene, "pc_radius")

        row = box.row(align=False)
        row.scale_x = 0.12
        row.label(text="  ")
        row.scale_x = 1.0
        row.prop(scene, "pc_min_points")

        # Step 2: Run Clustering
        row = box.row(align=True)
        row.scale_x = 0.12
        row.label(text="②")
        row.scale_x = 1.0
        row.operator("mesh.cluster_dist", text="Cluster", icon='PLAY')

        # Step 3: Manage Results
        row = box.row(align=True)
        row.scale_x = 0.12
        row.label(text="③")
        row.scale_x = 1.0
        row.operator("mesh.pc_select_noise", text="Select").attr_target = "cluster_id_dist"
        row.operator("mesh.pc_invert_selection", text="", icon='UV_SYNC_SELECT')
        row.operator("mesh.pc_split_clusters", text="Split").attr_target = "cluster_id_dist"
        row.operator("mesh.pc_delete_noise", text="Trash").attr_target = "cluster_id_dist"
        box = layout.box()
        box.label(text="Spectrum Logic", icon='COLOR')

        # Step 1: Select Attribute
        row = box.row(align=False)
        row.scale_x = 0.12
        row.label(text="①")
        row.scale_x = 1.0
        row.prop(scene, "pc_color_attr_enum", text="")

        # Step 2: Pick Reference Color
        row = box.row(align=True)
        row.scale_x = 0.12
        row.label(text="②")
        row.scale_x = 1.0
        row.prop(scene, "pc_ref_color", text="")
        row.operator("mesh.pick_color_selected", icon='EYEDROPPER', text="")

        # Step 3: Configure Tolerance
        row = box.row(align=False)
        row.scale_x = 0.12
        row.label(text="③")
        row.scale_x = 1.0
        row.prop(scene, "pc_color_tol")

        row = box.row(align=False)
        row.scale_x = 0.12
        row.label(text="  ")
        row.scale_x = 1.0
        row.prop(scene, "pc_material_mode")

        # Step 4: Preview, Analyze & Cluster
        row = box.row(align=True)
        row.scale_x = 0.12
        row.label(text="④")
        row.scale_x = 1.0
        row.operator("mesh.preview_color_matches", icon='ZOOM_ALL', text="")
        row.operator("mesh.pc_analyze_colors", icon='FCURVE', text="Analyze")
        row.operator("mesh.cluster_color", text="Cluster", icon='IMAGE_RGB_ALPHA')

        # Step 5: Manage Results
        row = box.row(align=True)
        row.scale_x = 0.12
        row.label(text="⑤")
        row.scale_x = 1.0
        row.operator("mesh.pc_select_cluster", text="Select").attr_target = "cluster_id_color"
        row.operator("mesh.pc_invert_selection", text="", icon='UV_SYNC_SELECT')
        row.operator("mesh.pc_split_clusters", text="Split").attr_target = "cluster_id_color"
        row.operator("mesh.pc_delete_noise", text="Trash").attr_target = "cluster_id_color"
        box = layout.box()
        box.label(text="Boolean Logic", icon='MOD_BOOLEAN')

        # Step 1: Pick the closed mesh and the point cloud
        row = box.row(align=False)
        row.scale_x = 0.12
        row.label(text="①")
        row.scale_x = 1.0
        row.prop(scene, "pc_bool_container")

        row = box.row(align=False)
        row.scale_x = 0.12
        row.label(text="  ")
        row.scale_x = 1.0
        row.prop(scene, "pc_bool_cloud")

        # Step 2: Configure Accuracy
        row = box.row(align=False)
        row.scale_x = 0.12
        row.label(text="②")
        row.scale_x = 1.0
        row.prop(scene, "pc_bool_samples", text="")

        # Step 3: Cut the point cloud
        row = box.row(align=True)
        row.scale_x = 0.12
        row.label(text="③")
        row.scale_x = 1.0
        row.operator("mesh.pc_boolean_intersect", text="Intersect", icon='SELECT_INTERSECT')
        row.operator("mesh.pc_boolean_difference", text="Difference", icon='SELECT_DIFFERENCE')

        # Live progress while running, result line once finished
        if scene.pc_bool_running:
            note = f"{scene.pc_bool_progress * 100:.1f}%    {scene.pc_bool_rate:,.0f} pts/s"
        else:
            note = scene.pc_bool_status
        if note:
            col = box.column(align=True)
            col.scale_y = 0.55
            col.label(text=note)

# --- 4. REGISTRATION ---

classes = (MESH_OT_InvertSelection, MESH_OT_PickColor, MESH_OT_Heatmap, MESH_OT_AnalyzeColors, MESH_OT_SelectNoise,
           MESH_OT_SelectCluster, MESH_OT_DeleteNoise, MESH_OT_SplitClusters,
           MESH_OT_ClusterDist, MESH_OT_ClusterColor, MESH_OT_PreviewColorMatches,
           MESH_OT_BooleanIntersect, MESH_OT_BooleanDifference, MESH_PT_MainPanel)

def register():
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
    for cls in classes: bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    props = ['pc_radius', 'pc_min_points', 'pc_color_attr_enum', 'pc_color_tol', 'pc_material_mode',
             'pc_ref_color', 'pc_bool_container', 'pc_bool_cloud',
             'pc_bool_samples', 'pc_bool_progress', 'pc_bool_rate',
             'pc_bool_status', 'pc_bool_running']
    for prop in props:
        if hasattr(bpy.types.Scene, prop): delattr(bpy.types.Scene, prop)

if __name__ == "__main__":
    register()
