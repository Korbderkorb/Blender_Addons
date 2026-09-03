# Blender Addons

A collection of Blender add-ons for point cloud processing, geometry generation and
other workflow tools. Each add-on is self-contained and can be installed on its own —
there is no shared dependency between them.

| Add-on | Version | Description |
| --- | --- | --- |
| [Point Cloud Master Suite](#point-cloud-master-suite) | 1.2.0 | Cluster, filter and boolean-cut large point clouds |

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

**Quick test (current session only)**

Open the `Scripting` workspace, load the `.py` file into the Text Editor and press
`Run Script` (`Alt + P`). The add-on stays active until you close Blender.

Most add-ons in this repository add a tab to the 3D Viewport sidebar. Press `N` in the
viewport if the sidebar is hidden.

---

# Point Cloud Master Suite

A toolkit for cleaning up and segmenting imported point clouds — typically photogrammetry
or laser-scan data brought in as `.ply` files, which Blender loads as a mesh with vertices
but no faces.

It has three independent sections: cluster by **position**, filter by **colour**, and cut
against a **closed mesh**. All three work directly on mesh vertices and preserve vertex
colours and any other point attributes carried in from the source file.

**Location:** `3D Viewport ▸ Sidebar (N) ▸ PC Tools`

**Requirements:** Developed and tested against Blender 5.0. The add-on manifest allows
3.6 and newer, but only 5.0 has been verified. No external Python packages are needed —
NumPy ships with Blender.

**A note before you start:** every tool here modifies the point cloud in place. Duplicate
your object (`Shift + D`, `Esc`) before a destructive step if you want to keep the full
cloud around. `Ctrl + Z` works for the boolean tools.

---

## Spatial Logic

Groups vertices that sit close together, so isolated specks of scanner noise can be
separated from the solid body of the scan.

### How to use it

1. **① Radius / Min Points.** *Radius* is the distance below which two points count as
   neighbours. *Min Points* is how many connected points a group needs before it counts
   as a real cluster. Start with a radius roughly 2–3× the average spacing of your cloud.
2. **② Cluster.** Runs the clustering. Every vertex is written to an integer attribute
   called `cluster_id_dist`: a group index for points in a valid cluster, or `-1` for
   points that never reached the *Min Points* threshold — the noise.
3. **③ Select / Invert / Split / Trash.**
   - **Select** — selects the *noise* (everything marked `-1`).
   - **Invert** — flips the selection, so you have the clustered points instead.
   - **Split** — moves the selected vertices out into a new object named `<name>_split`.
   - **Trash** — deletes the selected vertices.

The usual cleanup is: `Cluster ▸ Select ▸ Trash`, which throws away the loose specks and
keeps the body of the scan.

### Notes

- Clustering is a flood fill over a KD-tree, so runtime grows with point count. On very
  large clouds start with a generous radius and refine.
- A radius that is too large merges everything into one cluster; too small and everything
  becomes noise. If *Select* highlights the whole cloud, the radius is too small.

---

## Spectrum Logic

Selects points by colour, matched in HSV rather than raw RGB so that lighting differences
can be ignored on request. Useful for pulling vegetation, sky or a particular material out
of a scan.

### How to use it

1. **① Attribute.** Pick the colour attribute to read. Point clouds from `.ply` usually
   carry one named `Col`. The dropdown lists every point-domain attribute on the object.
2. **② Reference colour.** Either click the swatch and choose a colour, or select some
   representative vertices in Edit Mode and press the eyedropper button — that averages
   the colour of the selection and stores it as the reference.
3. **③ Colour Tolerance / Material Mode.** *Tolerance* is how far a point may deviate and
   still count as a match (`0` = exact, `1` = anything). *Material Mode (Ignore Shadows)*
   drops brightness from the comparison, so the same material matches whether it sits in
   sunlight or in shadow. Turn it on for surfaces that are lit unevenly.
4. **④ Preview / Analyze / Cluster.**
   - **Preview** (magnifier) — reports how many vertices currently match, without changing
     anything. Use it to dial in the tolerance before committing.
   - **Analyze** — writes a float attribute `pc_color_score` between 0 and 1 describing how
     well each point matches. Handy for driving effects in Geometry Nodes or shaders.
   - **Cluster** — writes the integer attribute `cluster_id_color`, set to `0` on matching
     points and `-1` everywhere else.
5. **⑤ Select / Invert / Split / Trash.**
   - **Select** — selects the *matching* points.
   - **Invert**, **Split**, **Trash** — as in Spatial Logic.

> **Watch out:** *Select* behaves opposite to the one in Spatial Logic. There it grabs the
> noise; here it grabs the matches. Use **Invert** if you want the other set.

### Notes

- Hue is compared as a wrapped distance, so reds either side of the 0/1 boundary still
  match each other.
- A `Heatmap` operator is also registered but has no button in the panel. It writes a
  colour attribute `pc_heatmap`, green where points match and blue where they do not.
  Run it from the search menu (`F3`, then type "Heatmap") and switch the viewport to
  attribute display to see the result.

---

## Boolean Logic

Cuts a point cloud against a closed mesh: keep the points inside it, or keep the points
outside it. This is the point cloud equivalent of a boolean modifier, which cannot operate
on geometry that has no faces.

### How to use it

1. **① Closed Mesh / Point Cloud.** Pick both objects explicitly.
   - *Closed Mesh* is the container. Only objects that actually have faces appear in this
     list. It is read-only and never modified.
   - *Point Cloud* is the object that gets cut. **This is the one that changes.**
   - Each dropdown hides whatever is already selected in the other, so you cannot pick the
     same object twice.
2. **② Accuracy.** How many ray directions vote on each point:
   - **Fast (1 ray)** — quickest, but tends to leave a scatter of stray points behind.
   - **Balanced (3 rays)** — the default, and the right choice almost always.
   - **Thorough (5 rays)** — for containers with messy, coplanar or non-watertight geometry.
3. **③ Intersect / Difference.**
   - **Intersect** keeps the points **inside** the closed mesh.
   - **Difference** keeps the points **outside** it.

While it runs, a small line under the buttons shows progress and throughput, for example
`42.4%    523,992 pts/s`. Press `Esc` to cancel — nothing is deleted if you do. When it
finishes, that line is replaced by the result, for example `Removed 385,211 of 400,000 points`.

`Ctrl + Z` undoes the cut.

### How it works

Each point is classified by crossing-parity ray casting: a ray fired from the point crosses
a closed surface an odd number of times if and only if it started inside. A single ray is
fragile — one that clips an edge or a vertex, or slips through a small hole in the
container, miscounts by one and flips the answer, which shows up as isolated stray points
surviving far away from the container. The *Accuracy* setting therefore has several
well-spread directions vote on each point, and voting stops early once a result is settled.

Points outside the container's bounding box are rejected without a ray cast at all, so the
cost tracks the number of points that genuinely overlap the container rather than the total
size of the cloud. Expect roughly half a million classified points per second on typical
geometry.

### Notes and troubleshooting

- **Vertex colours and other point attributes are preserved** on the surviving points.
- **The container should be watertight.** If it has holes or non-manifold edges, the
  add-on says so in the status bar after the run. Parity testing is only as reliable as the
  surface it is testing against.
- **Stray points left over?** Raise *Accuracy* to Thorough. If the container is badly
  non-watertight, closing it is the more fundamental fix.
- **Nothing was removed?** Check that the two objects actually overlap in world space, and
  that you used the button you meant — *Intersect* and *Difference* keep opposite sets.
- If a run would delete every single point, the add-on refuses and leaves the cloud
  untouched rather than emptying it silently.
- Object transforms are handled, so the two objects do not need matching scale, rotation
  or origin.

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
