# Blender Addons

A collection of Blender add-ons for point cloud processing, geometry generation and
other workflow tools. Each add-on is self-contained and can be installed on its own —
there is no shared dependency between them.

| Add-on | Version | Description |
| --- | --- | --- |
| [Point Cloud Master Suite](#point-cloud-master-suite) | 1.9.0 | Generate, clean, segment, measure, grow, fill, boolean-cut and visualize point clouds |

More add-ons will be added to this repository over time.

---

## Installation

Two ways to get an add-on running. Both start with downloading the `.py` file from
this repository.

**Permanent install (recommended)**

1. In Blender, go to `Edit ▸ Preferences ▸ Add-ons`.
2. Click the dropdown arrow in the top-right corner and choose `Install from Disk…`
   (in Blender versions before 4.2 this is an `Install…` button).
3. Select the downloaded `.py` file.
4. Tick the checkbox next to the add-on name to enable it.

When updating from an older version, disable and remove the old add-on first, then
install the new file.

**Quick test (current session only)**

Open the `Scripting` workspace, load the `.py` file into the Text Editor and press
`Run Script` (`Alt + P`). The add-on stays active until you close Blender.

Most add-ons in this repository add a tab to the 3D Viewport sidebar. Press `N` in the
viewport if the sidebar is hidden.

---

# Point Cloud Master Suite

A toolkit for working with point clouds in Blender — photogrammetry or laser-scan data
brought in as `.ply` files (which Blender loads as a mesh with vertices but no faces),
or clouds generated from video right inside the add-on.

**Location:** `3D Viewport ▸ Sidebar (N) ▸ PC Tools`

**Requirements:** Developed and tested against Blender 5.0. The add-on manifest allows
3.6 and newer, but only 5.0 has been verified. Progress bars need Blender 4.0 or newer;
older versions show the progress as text instead. The **Visualize** section needs
Blender 5.0 or newer; everything else runs on 3.6 and up. No external Python packages
are needed — NumPy ships with Blender.

The sidebar tab has four sections, each with its own tools:

| Section | Tools | What they do |
| --- | --- | --- |
| **Generate** | Video Stack | Turn a video into a stacked 3D point cloud |
| **Select** | Spatial, Thickness, Spectrum | Find points by neighbourhood, by material thickness or by colour |
| **Modify** | Analyze Lattice, Rescale, Growth, Fill Volume, Boolean | Measure the grid a cloud sits on and build on it |
| **Visualize** | Visualize, Size by Attribute | Show a cloud as real points, coloured and sized by its stored values |

**Before you start:** some tools change the point cloud in place (Trash, Rescale, Grow,
Intersect, Difference). Duplicate your object (`Shift + D`, `Esc`) before a destructive
step if you want to keep the full cloud around. `Ctrl + Z` works for those tools. Tools
that create something new (Video Stack, Split, Fill Volume, Invert) always write a new
object and leave their inputs untouched.

---

## Progress and Cancel

Every tool in the suite reports its progress in its own panel while it runs, with a small
red **Cancel** button (✕) to the right of the progress bar. `Esc` in the viewport does the
same. A cancelled tool writes nothing, so the cloud is exactly as it was before you
pressed the button. When a tool finishes, the progress line is replaced by a short result,
for example `Removed 385,211 of 400,000 points`.

A tool's button is greyed out while that tool is running.

---

## Generate

### Video Stack

Turns a video into a point cloud: every pixel of every frame becomes one point, and the
frames are stacked along Z. Useful for visualising motion over time as a 3D volume.

1. **① Video.** Pick the file. Its resolution and length are read straight away and shown
   below the field, and the frame range is set to the whole clip.
2. **② Width / Height / Lock Aspect.** The number of points across and down each frame.
   The video is scaled to this before it is read, so this — not the resolution of the
   file — decides the point count. *Lock Aspect* keeps the height on the video's aspect
   ratio.
3. **③ Start / End / Frame Step.** Which frames to take. `End = 0` runs to the end of the
   clip; a *Frame Step* of 2 takes every second frame.
4. **④ Pixel Size / Z Distance / Skip Below.** *Pixel Size* is the spacing between points
   inside one frame (each frame is centred on the origin). *Z Distance* is the spacing
   between frames — negative stacks downwards. *Skip Below* drops pixels darker than the
   value, which is a quick way to throw away a black background.
5. **⑤ Generate.** Builds the cloud as one new object.

The panel shows the resulting point count before you press anything, and turns red above
20 million points. Each point carries its pixel colour in `Col`, its source frame in
`frame`, and its pixel position in `px` / `py`, so the rest of the suite can work on the
result immediately. Stacks of more than 100 million points are refused.

---

## Select

### Spatial

Groups points that sit close together, so isolated specks of scanner noise can be
separated from the solid body of the scan.

1. **① Radius / Min Points.** *Radius* is the distance below which two points count as
   neighbours. *Min Points* is how many connected points a group needs before it counts
   as a real cluster. Start with a radius roughly 2–3× the average spacing of your cloud.
2. **② Cluster.** Runs the clustering. Every point is written to an integer attribute
   called `cluster_id_dist`: a group index for points in a valid cluster, or `-1` for
   points that never reached the *Min Points* threshold — the noise.
3. **③ Select / Invert / Split / Trash.**
   - **Select** — selects the *noise* (everything marked `-1`).
   - **Invert** (arrows icon) — flips the selection, so you have the clustered points
     instead.
   - **Split** — moves the selected points into a new object named `<name>_split`. Colours,
     all other attributes and the object transform are kept.
   - **Trash** — deletes the selected points.

The usual cleanup is `Cluster ▸ Select ▸ Trash`, which throws away the loose specks and
keeps the body of the scan.

- A radius that is too large merges everything into one cluster; too small and
  everything becomes noise. If *Select* highlights the whole cloud, the radius is too
  small.
- Clustering is a flood fill over a KD-tree, so runtime grows with point count. On very
  large clouds start with a generous radius and refine.

### Thickness

Measures how thick the material is around every point, so thin sheets and thin struts
can be found in any orientation — even where they are joined to the rest of the cloud,
which clustering cannot detect.

1. **① Measure Up To / Voxel Size / Fill Gaps.**
   - *Measure Up To* is the largest thickness worth measuring; anything thicker is just
     reported as thick. Keep it a little above the thickest part you would still call thin
     — raising it costs time.
   - *Voxel Size* is the resolution. Leave it at `0` and Analyze picks about 3.5× the
     point spacing and writes it back here.
   - *Fill Gaps* plugs single-voxel holes caused by uneven sampling, which would
     otherwise make the parts around them look thin. Leave it on.
2. **② Analyze.** Measures the thickness in metres and stores it in the `pc_thickness`
   attribute. It works by fitting a ball inside the cloud at each point.
3. **③ Thin Below / Heatmap / Select / Invert.**
   - *Thin Below* is the cut: points in material thinner than this count as thin. Change
     it freely after Analyze — nothing is re-measured.
   - **Heatmap** (curve icon) writes a colour attribute `pc_thin_heat`, red where thin
     and blue where thick. Switch the viewport to attribute display to see it.
   - **Select** selects every thin point. **Invert** flips the selection.

A typical use is `Analyze ▸ Select`, then **Grow** (see Modify) to thicken the thin parts.

### Spectrum

Selects points by colour, matched in HSV rather than raw RGB so that lighting differences
can be ignored on request. Useful for pulling vegetation, sky or a particular material out
of a scan.

1. **① Attribute.** Pick the colour attribute to read. Point clouds from `.ply` usually
   carry one named `Col`. Only colour attributes are listed.
2. **② Reference colour.** Either click the swatch and choose a colour, or select some
   representative points in Edit Mode and press the eyedropper — that averages the colour
   of the selection and stores it as the reference.
3. **③ Colour Tolerance / Material Mode.** *Tolerance* is how far a point may deviate and
   still count as a match (`0` = exact, `1` = anything). *Material Mode (Ignore Shadows)*
   drops brightness from the comparison, so the same material matches whether it sits in
   sunlight or in shadow.
4. **④ Preview / Analyze / Cluster.**
   - **Preview** (magnifier) — reports how many points currently match, without changing
     anything. Use it to dial in the tolerance before committing.
   - **Analyze** — writes a float attribute `pc_color_score` between 0 and 1 describing
     how well each point matches. Handy for driving Geometry Nodes or shaders.
   - **Cluster** — writes the integer attribute `cluster_id_color`, set to `0` on matching
     points and `-1` everywhere else.
5. **⑤ Select / Invert / Split / Trash.** As in Spatial, except that **Select** grabs the
   *matching* points.

> **Watch out:** *Select* behaves opposite to the one in Spatial. There it grabs the
> noise; here it grabs the matches. Use **Invert** if you want the other set.

- Hue is compared as a wrapped distance, so reds either side of the 0/1 boundary still
  match each other.
- A `Heatmap` operator is also registered but has no button in the panel. It writes a
  colour attribute `pc_heatmap`, green where points match and blue where they do not.
  Run it from the search menu (`F3`, then type "Heatmap").

---

## Modify

Everything in Modify works on the **grid** a point cloud sits on, so the section starts
with the tool that measures it.

### Analyze Lattice

The button at the top of the Modify section. Select the cloud and press **Analyze
Lattice**: it works out the grid the points sit on — at whatever orientation that grid
has — and fills in **Pitch**, the spacing between neighbouring points along each of the
three grid directions.

The result line reports how well the points really sit on that grid. Check it before
building anything: a low percentage, or an offset that grows with distance, means one
grid does not describe this cloud and new points would land in the wrong places.

Analyze Lattice never moves a point. It only writes a `pc_lattice_residual` attribute
(how far each point is off the grid). *Pitch* can be corrected by hand afterwards.

### Rescale

Changes the spacing of a cloud that already sits on a grid.

1. **① New Pitch / Uniform.** *New Pitch* starts out as the measured Pitch. Type the
   spacing you want instead. With *Uniform* on, changing one value scales the other two in
   proportion, so the grid keeps its shape; turn it off to stretch the three directions
   differently. The factors are shown below the field.
2. **② Rescale.** Moves every point until the grid steps really measure *New Pitch*, then
   reads the result back into *Pitch*.

The stretch runs along the grid's own directions around the grid origin, so the grid keeps
its orientation and anything already grown on it still lines up. It moves the points, not
the object, so the object transform, modifiers and attributes are left alone. Distances
set elsewhere in the sidebar (Radius, Thin Below, …) are not converted.

### Growth

Adds new points around the selected ones, on the grid found by Analyze Lattice, and never
where a point already exists.

1. **① Mode / Steps.**
   - **Uniform** — grows one grid step in every direction (thicker and wider).
   - **Thicken** — grows only across the thin direction, so a sheet gets thicker and a
     strut gets fatter without spreading sideways.
   - **Extend** — grows only along the part, so a strut grows out at its tip and a sheet
     at its rim. Use it to carry a thin part on until it meets another one.
   - *Steps* is how many grid steps to grow in one press.
2. **② Grow.** Select the points to grow from in Edit Mode first — for example the thin
   parts found by Thickness. Afterwards only the new points are selected, so pressing again
   carries the growth on from the new front. New points inherit the attributes of the
   points they grew from.

### Fill Volume

Fills a closed mesh with mirrored copies of a point cloud, producing one large, seamless
cloud with exactly the original spacing.

1. **① Volume / Point Cloud / Mode / Tile Override.**
   - *Volume* is the closed mesh to fill. *Point Cloud* is the cloud to repeat. Both are
     read-only.
   - *Mode*:
     - **Fill and Trim** — keeps only the points inside the mesh, so the fill follows its
       shape.
     - **Cover Only** — mirrors the cloud until the mesh's bounding box is covered in
       whole tiles and keeps every point, so you can cut it with a Boolean yourself.
       Faster, since nothing is tested against the surface.
   - *Tile Override* is unticked by default, and the tile (the block of grid cells that
     gets repeated) is measured from the cloud. Tick it to type the tile size by hand;
     `0` in a direction still measures that direction. Unticking it resets the tile to
     `0, 0, 0`.
2. **② Fill Volume.** Builds the fill as a new object named `<cloud>_filled`.

The copies are laid out on the grid from Analyze Lattice (the panel shows which pitch it
will use, or reminds you to analyze first). Every second copy along each grid direction is
mirrored, and each seam is placed so the mirrored row lands exactly one point spacing from
the row it was mirrored off — no gap and no doubled points. The original cloud keeps its
exact position, and the result keeps the cloud's attributes, materials and modifiers.

The panel shows an estimate of the tile count and point count before you press anything.
There is no upper limit, but a fill of more than **100 million points** asks for
confirmation first — in a popup and with **Place Them / Cancel** buttons in the panel.
(Scripts that call the operator need to pass `allow_large=True` for such fills.)

### Boolean

Cuts a point cloud against a closed mesh, or builds its inverse. This is the point cloud
equivalent of a boolean modifier, which cannot operate on geometry that has no faces.

1. **① Closed Mesh / Point Cloud.** Pick both objects explicitly.
   - *Closed Mesh* is the container. Only objects with faces appear in this list. It is
     never modified.
   - *Point Cloud* is the object that gets cut. **For Intersect and Difference, this is
     the one that changes.**
   - Each dropdown hides whatever is already picked in the other, so you cannot pick the
     same object twice.
2. **② Accuracy.** How many ray directions vote on each point:
   - **Fast (1 ray)** — quickest, but tends to leave a scatter of stray points behind.
   - **Balanced (3 rays)** — the default, and the right choice almost always.
   - **Thorough (5 rays)** — for containers with messy, coplanar or non-watertight
     geometry.
3. **③ Intersect / Difference / Invert.**
   - **Intersect** keeps the points **inside** the closed mesh.
   - **Difference** keeps the points **outside** it.
   - **Invert** builds the **voids**: it imagines the mesh completely filled with points
     on the cloud's grid, and keeps only the grid positions the cloud does *not* occupy.

While Intersect or Difference runs, the progress bar also shows the throughput, for
example `523,992 pts/s`. `Ctrl + Z` undoes the cut.

#### Invert in detail

- **Run Analyze Lattice on the cloud first** — Invert uses that grid. If the points do not
  fit it, Invert stops and says so.
- **The mesh is optional.** Without one, Invert fills the box of grid rows the cloud spans
  (the panel notes this).
- A grid position counts as **occupied** when a cloud point lies within half a pitch of
  it, so slightly irregular clouds still work. Every other position inside the boundary
  becomes a new point — including holes of a single point. Use Spatial afterwards if you
  want to remove small ones.
- Every new point takes the **colour and all other attributes of the nearest original
  point**.
- The result is always a **new object** named `<cloud>_inverted`, with the cloud's
  transform. The cloud and the mesh are left untouched.
- The grid is fitted to the points before it is used, so the new points line up with the
  cloud's own points across the whole cloud. Centred grids (for example body-centred
  clouds read as a cubic grid) are handled: points are only placed at positions the cloud
  actually uses.
- Like Fill Volume, an Invert of more than 100 million points asks for confirmation first.

#### How the inside test works

Each point is classified by crossing-parity ray casting: a ray fired from the point
crosses a closed surface an odd number of times if and only if it started inside. A single
ray is fragile — one that clips an edge or a vertex, or slips through a small hole in the
container, miscounts by one and flips the answer. The *Accuracy* setting therefore has
several well-spread directions vote on each point, and voting stops early once a result is
settled. Points outside the container's bounding box are rejected without a ray cast at
all, so the cost tracks the number of points that genuinely overlap the container.

Fill Volume and Invert use the same test per grid line rather than per point: one ray
along a line of grid positions sees every crossing on that line, which makes filling large
volumes fast.

#### Notes and troubleshooting

- **Colours and other point attributes are preserved** on the surviving points.
- **The container should be watertight.** If it has holes or non-manifold edges, the
  add-on says so in the result line after the run.
- **Stray points left over?** Raise *Accuracy* to Thorough. If the container is badly
  non-watertight, closing it is the more fundamental fix.
- **Nothing was removed?** Check that the two objects actually overlap in world space,
  and that you used the button you meant — *Intersect* and *Difference* keep opposite
  sets.
- If a run would delete every single point, the add-on refuses and leaves the cloud
  untouched rather than emptying it silently.
- Object transforms are handled, so the two objects do not need matching scale, rotation
  or origin.

---

## Visualize

Shows a point cloud as real, round points in its own colours — in the viewport and in
Cycles and EEVEE renders — with one click. Requires **Blender 5.0 or newer**.

Visualize adds a Geometry Nodes modifier to the cloud; the mesh itself is never changed,
so every other tool in the suite keeps working on the cloud while it is shown. The
modifier is kept last in the stack and is switched off in Edit Mode, where you see the
bare vertices.

### Visualize

1. **Visualize.** Select one or more clouds and press the button. A viewport in Solid or
   Wireframe is switched to Material Preview so the colours show straight away. Pressing
   it again hides the points and keeps every setting; the **✕** next to it removes the
   modifier.
2. **Size.** Counted in point spacings, so `1` makes neighbouring points just touch,
   whether the cloud is a millimetre or a kilometre across. The spacing is measured when
   Visualize is turned on and shown below the field. The refresh button beside *Size*
   measures it again — use it after Rescale, Fill Volume or Growth changed the spacing.
3. **Color By.**
   - **True Color** (default) — the cloud's colour attribute (`Col`, or whichever you
     pick), shown unchanged. *Brightness* and *Saturation* adjust it.
   - **Heatmap** — colours the points by a stored value. Palettes: Turbo, Viridis, Blue to
     Red, Grayscale.
   - **ID Colors** — a colour of its own for every cluster id (`cluster_id_dist`,
     `cluster_id_color`). *Seed* reshuffles the colours.
   - **Solid** — one colour for every point, to judge the size alone. A cloud without
     colours starts here.
4. **Unlit / Shaded.** *Unlit* shows the exact colours without light or shadow; *Shaded*
   lights the points like any surface, for renders.
5. **Viewport Density.** Draws only a share of the points in the viewport, to keep huge
   clouds smooth to orbit. Renders always use every point.

### Heatmaps read stored values

The heatmap and the size influences only read attributes already on the cloud —
`pc_thickness`, `pc_color_score`, `pc_lattice_residual`, the cluster ids, `frame`, or
anything you imported. Nothing is recomputed: changing the range, curve or palette updates
the view at once. If the value you pick is missing, the panel tells you which tool makes
it, for example *run Select > Thickness > Analyze*.

Every value is mapped through the same remap block:

- **Min / Max** — the range of values used. **Auto** (and picking a value) sets it to the
  2nd–98th percentile, so a few outliers do not flatten the rest. The real range of the
  value is shown below. A range set by hand is left alone.
- **Linear / Smooth / Steps** — how values are spread over the range. *Steps* sets the
  number of bands.
- **Invert** (⇆) swaps the ends. Picking `pc_thickness` turns it on, so thin parts come
  out hot and large.

### Size by Attribute

A sub-panel under Visualize, closed until you need it.

- **Scale by** picks a stored value that scales the points, from **Low ×** to **High ×**
  of their size. *None* turns it off.
- **And by** appears once the first is on and adds a second value; the two multiply.
- **Hide Below** leaves out points scaled below a given multiplier, to show only part of
  the cloud.

### Under the hood

One node group (`PC Visualize`, using `PC Visualize Remap`) and two materials
(`PC Visualize Unlit` / `PC Visualize Shaded`) are shared by every cloud in the file. Each
cloud's settings live only in its own modifier, so the sidebar and the Modifier tab always
agree, and a setting changed in either place shows up in the other. The computed colour is
stored as `pc_viz_color`, which is the only attribute the materials read.

---

## Changelog

See the release notes on GitHub for details. In short:

- **1.9.0** — New **Visualize** section: one-click point display in Cycles and EEVEE,
  live heatmaps from stored values, ID colours, and point size driven by up to two
  attributes. Spectrum colour-attribute dropdown fix.
- **1.8.0** — Progress bar and Cancel button on every tool; Boolean **Invert**; Tile
  Override checkbox in Fill Volume; Split keeps attributes and transform.
- **1.7.0** — Seamless mirroring in Fill Volume fixed on all axes; Fill and Trim / Cover
  Only modes; point limit removed (confirmation above 100 million points); Modify section
  reordered with Analyze Lattice on top and Rescale as its own tool.
- **1.6.0** — Fill Volume.
- **1.5.0** — Pitch Rescale.
- **1.4.0** — Video Stack, Thickness, Analyze Lattice and Growth.
- **1.2.0** — First release: Spatial, Spectrum and Boolean.

---

## Feedback and contributions

Bug reports, feature ideas and general suggestions are very welcome — please open an issue
in this repository. If something behaves unexpectedly, include your Blender version and,
where relevant, roughly how large the point cloud was and whether the container mesh was
watertight. That makes issues much faster to reproduce.

Pull requests are welcome too.

---

## Licence

Released under the **MIT Licence**. In plain terms, you are free to:

- use the add-ons for anything, including commercial work,
- modify them and adapt them to your own pipeline,
- redistribute them, on their own or as part of something larger.

The one condition is that the copyright notice and licence text stay with the code in any
copy or substantial portion you pass on. The software comes with no warranty of any kind —
if it breaks something, that is on you, not on the author.

Beyond what the licence strictly requires: if you use this work, build on it, or ship
something derived from it, **please credit Korbinian Enzinger** and link back to this
repository. Attribution costs nothing and makes it worthwhile to keep publishing these
tools openly.

© Korbinian Enzinger. See the [LICENSE](LICENSE) file for the full text.
