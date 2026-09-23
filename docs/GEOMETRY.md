# The submap geometry convention

Several files point at this page — `davio_mapper/mapping.py`, `davio_mapper/online.py`,
`scripts/evaluate_surface.py`, `scripts/view_run.py` — because reading an archived submap
back into world points is the one place where a silent convention change moves every point
in the map. This page states the convention and what the older one was.

`map/map_index.json` carries `geometry_convention`, and everything that reads an archive
branches on it. Schema 3 writes `depth_only_scale`.

## What a submap archive holds

`map/submaps/NNNNNN.npz` is one mapping window: `window_frames` rectified frames, the DA3
depth for each, per-pixel colour and validity, and the **local** camera poses DA3 was
conditioned on. `map_index.json` records, per submap, its member frame ids, which of them
is the `center`, its `T_odom_submap` / `T_map_submap` node pose and its `scale`.

A submap's node is a similarity: a rotation, a translation, and one **depth scale**.

## `depth_only_scale` (schema 3, current)

**The residual scale multiplies depth only. Camera baselines are never rescaled.**

The camera poses inside a window come from the VIO filter and are already metric — that is
the entire point of conditioning DA3 on them. What is *not* metric is the depth the network
predicts, which carries a per-window scale error of roughly 10 % however long the window.
So the node's scale variable is a property of the depth, and applying it to the in-window
camera translations as well would be applying a correction to a quantity that did not need
it, stretching the window's own geometry.

Reading a point back is therefore: take the pixel's ray in its own frame, multiply the
depth by the node scale, place it with that frame's **unscaled** local pose, then apply the
node's rotation and translation.

`davio_mapper.mapping.world_points(archive, center, transform, pixel_step)` does this.

## `scaled_baseline` (legacy)

Earlier archives applied the node scale to the in-window camera translations too. Only the
centre frame is unaffected — it is the frame the node pose is expressed at, so its
translation is zero either way. Every other frame in the window lands in the wrong place,
by the scale error times its baseline from the centre.

`world_points(..., legacy_scaled_baseline=True)` reads them the old way. Anything that
opens an archive checks `map_index.json`'s `geometry_convention` and passes the flag; a
run whose index predates the field is treated as legacy.

The archives themselves were never rewritten. A legacy run read with the legacy convention
reproduces exactly what it reported, which is why the flag exists rather than a migration.

## What this means for a comparison

Two runs can only be compared on surface metrics if they were read under the same
convention, which is why every `surface*.json` carries `map_index_geometry`. Every run this
code writes uses the current convention.
