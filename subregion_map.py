#!/usr/bin/env python3
"""Clip a region out of a Craft parent map into its own child map.

A Craft `location` map (``metadata.map``) can hold many ``areas[]`` polygons.
When an overland map gets crowded, you can give one region its own zoomed-in
child map: this tool takes one area on a parent map, computes the window around
it (its bounding box plus optional padding), and re-projects that region plus a
chosen set of neighbouring regions into the child location's own ``metadata.map``.

The map lives in a fixed place (``metadata.map``) on every ``location`` file,
so this works on any Craft project regardless of the extra fields a project's
location schema adds. It operates on a single packed ``.cdf.json`` in place.

Example
-------
    python tools/subregion_map.py subregion-tool-example-cdf.json \
      --parent "Example Location" --clip "Area 1" \
      --include "Overlap Include in Area 1" "Interior Include in Area 1" \
                "Exterior include in area 1" \
      --padding 0.05 --dry-run

Behaviour / decisions
---------------------
- Background: the child map's ``background`` is left ``null``. The tool instead
  prints the crop rectangle (in parent-normalised coords) so you can crop the
  overland image externally and paste the URL in later.
- Partial-overlap areas are clipped to the crop window (Sutherland-Hodgman) so
  every vertex stays in 0..1 — Craft's importer rejects out-of-range vertices.
  Pass --no-clip to keep whole polygons instead (import will reject them).
- ``--boundary`` (default off) writes the clip area's own polygon (rescaled) as
  the child map's ``boundaryPolygon``.
- Parent ``points[]`` (pins/tokens) inside the padded window are carried over
  and rescaled (default on; disable with ``--no-points``).
- The parent map is never modified.

Known limitations
-----------------
- ``grid`` is not recomputed under rescale; it is left ``null`` unless the
  parent had one, in which case it is copied verbatim and may need manual
  adjustment (cell sizing shifts under rescale).
- The overlap test uses vertex-bounding-box intersection (an over-approximation),
  which matches the "full or part" intent; whole polygons are kept so true
  polygon-rectangle intersection is unnecessary.
- The background image itself is not cropped (by design) — only the crop rect
  is emitted.
"""

import argparse
import json
import math
import shutil
import sys
import uuid

DEFAULT_SCHEMA_VERSION = "file-designation-metadata:v7"


def eprint(*args):
    print(*args, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Loading / resolving
# --------------------------------------------------------------------------- #

def load_cdf(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def location_files(cdf):
    """All files with fileTypeKey 'location'."""
    return [f for f in cdf.get("files", []) if f.get("fileTypeKey") == "location"]


def resolve_location(cdf, token):
    """Resolve a location by content.name (preferred) then by referenceId.

    Returns the file dict, or None. Name matching is exact; if several
    locations share a name it is ambiguous and we error out.
    """
    locs = location_files(cdf)
    by_name = [f for f in locs if (f.get("content") or {}).get("name") == token]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise SystemExit(
            f"error: location name {token!r} is ambiguous "
            f"({len(by_name)} matches); use its referenceId instead"
        )
    for f in locs:
        if f.get("referenceId") == token:
            return f
    return None


def get_map(loc):
    return ((loc or {}).get("metadata") or {}).get("map")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def bbox(vertices):
    xs = [v["x"] for v in vertices]
    ys = [v["y"] for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def padded_window(clip_bbox, padding):
    """Expand the clip bbox per axis by `padding` (a fraction of each side),
    then clamp to [0, 1] since no image exists off-canvas.

    Returns (wmin, hmin, wmax, hmax).
    """
    minx, miny, maxx, maxy = clip_bbox
    pad_x = padding * (maxx - minx)
    pad_y = padding * (maxy - miny)
    wmin = max(0.0, minx - pad_x)
    hmin = max(0.0, miny - pad_y)
    wmax = min(1.0, maxx + pad_x)
    hmax = min(1.0, maxy + pad_y)
    return wmin, hmin, wmax, hmax


def window_padding_km(parent_map, km):
    """Convert an absolute padding in km into normalised pad_x, pad_y using the
    parent map's scale + aspect ratio. Returns (pad_x, pad_y).

    normalised-x 1.0 spans scale.widthMeasurement km; normalised-y 1.0 spans that
    scaled by the aspect ratio (height px / width px), so non-square maps stay
    isotropic in km.
    """
    scale = parent_map.get("scale") or {}
    wm = scale.get("widthMeasurement")
    if not wm:
        raise SystemExit(
            "error: --padding-km needs the parent map's scale.widthMeasurement"
        )
    pa = parent_map.get("aspectRatio") or {"width": 1, "height": 1}
    aw = pa.get("width", 1) or 1
    ah = pa.get("height", 1) or 1
    km_per_x = wm
    km_per_y = wm * (ah / aw)
    return km / km_per_x, km / km_per_y


def padded_window_abs(clip_bbox, pad_x, pad_y):
    """Like padded_window but with absolute per-axis pads (normalised units)."""
    minx, miny, maxx, maxy = clip_bbox
    wmin = max(0.0, minx - pad_x)
    hmin = max(0.0, miny - pad_y)
    wmax = min(1.0, maxx + pad_x)
    hmax = min(1.0, maxy + pad_y)
    return wmin, hmin, wmax, hmax


def make_rescaler(window):
    wmin, hmin, wmax, hmax = window
    w = wmax - wmin
    h = hmax - hmin
    if w <= 0 or h <= 0:
        raise SystemExit("error: padded window has zero area; check the clip area")

    def rescale_point(x, y):
        return (x - wmin) / w, (y - hmin) / h

    return rescale_point, w, h


def _clip_halfplane(verts, keep, intersect):
    """Sutherland-Hodgman clip of a polygon against one half-plane."""
    if not verts:
        return verts
    out = []
    S = verts[-1]
    s_in = keep(S)
    for E in verts:
        e_in = keep(E)
        if e_in:
            if not s_in:
                out.append(intersect(S, E))
            out.append(E)
        elif s_in:
            out.append(intersect(S, E))
        S, s_in = E, e_in
    return out


def clip_to_unit_square(verts):
    """Clip a polygon to the [0,1]x[0,1] box. Returns new vertex list (may be
    empty if the polygon lies fully outside). Craft's importer requires every
    vertex in range, so out-of-window neighbour polygons must be clipped, not
    left to "visual" clipping."""
    def lin(S, E, t):
        return {"x": S["x"] + (E["x"] - S["x"]) * t,
                "y": S["y"] + (E["y"] - S["y"]) * t}
    verts = _clip_halfplane(verts, lambda p: p["x"] >= 0.0,
                            lambda S, E: lin(S, E, (0.0 - S["x"]) / (E["x"] - S["x"])))
    verts = _clip_halfplane(verts, lambda p: p["x"] <= 1.0,
                            lambda S, E: lin(S, E, (1.0 - S["x"]) / (E["x"] - S["x"])))
    verts = _clip_halfplane(verts, lambda p: p["y"] >= 0.0,
                            lambda S, E: lin(S, E, (0.0 - S["y"]) / (E["y"] - S["y"])))
    verts = _clip_halfplane(verts, lambda p: p["y"] <= 1.0,
                            lambda S, E: lin(S, E, (1.0 - S["y"]) / (E["y"] - S["y"])))
    for p in verts:  # guard against tiny FP overshoot
        p["x"] = min(1.0, max(0.0, p["x"]))
        p["y"] = min(1.0, max(0.0, p["y"]))
    return verts


def bbox_intersects(b, window):
    """Axis-aligned bbox vs window intersection (touching counts)."""
    minx, miny, maxx, maxy = b
    wmin, hmin, wmax, hmax = window
    return not (maxx < wmin or minx > wmax or maxy < hmin or miny > hmax)


def point_in_window(x, y, window):
    wmin, hmin, wmax, hmax = window
    return wmin <= x <= wmax and hmin <= y <= hmax


def rescale_vertices(vertices, rescale_point):
    out = []
    for v in vertices:
        nx, ny = rescale_point(v["x"], v["y"])
        out.append({"x": nx, "y": ny})
    return out


def child_aspect(parent_map, w, h):
    """Child aspect = W*parentW : H*parentH, reduced when it comes out clean.

    A normalised rect's true proportions depend on the parent's aspect ratio,
    so ignoring it would stretch the child map.
    """
    pa = parent_map.get("aspectRatio") or {"width": 1, "height": 1}
    aw = w * pa.get("width", 1)
    ah = h * pa.get("height", 1)
    # Try to express as small integers when the ratio is clean.
    for scale in (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16):
        iw = aw * scale
        ih = ah * scale
        if _near_int(iw) and _near_int(ih):
            iw, ih = round(iw), round(ih)
            g = math.gcd(iw, ih) or 1
            if iw and ih:
                return {"width": iw // g, "height": ih // g}
    # Fall back to rounded floats.
    return {"width": round(aw, 6), "height": round(ah, 6)}


def _near_int(x, tol=1e-6):
    return abs(x - round(x)) < tol


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build_child_map(cdf, parent_map, clip_area, includes, window,
                    rescale_point, w, h, want_boundary, want_points,
                    want_clip=True):
    ref_index = {f.get("referenceId"): f for f in cdf.get("files", [])}
    parent_areas = parent_map.get("areas") or []

    kept_areas = []
    skipped = []

    for token in includes:
        loc = resolve_location(cdf, token)
        if loc is None:
            skipped.append((token, "location not found"))
            continue
        ref = loc.get("referenceId")
        matches = [a for a in parent_areas if a.get("fileReferenceId") == ref]
        if not matches:
            skipped.append((token, "not an area on the parent map"))
            continue
        added = False
        for area in matches:
            verts = area.get("vertices") or []
            if not verts:
                continue
            if not bbox_intersects(bbox(verts), window):
                continue
            rv = rescale_vertices(verts, rescale_point)
            if want_clip:
                rv = clip_to_unit_square(rv)
                if len(rv) < 3:  # polygon lies fully outside the window
                    continue
            kept_areas.append({
                "id": str(uuid.uuid4()),
                "fileReferenceId": ref,
                "vertices": rv,
                "style": area.get("style"),
                "fillImage": area.get("fillImage"),
            })
            added = True
        if not added:
            skipped.append((token, "no polygon overlaps the window"))

    # Points / pins inside the window.
    kept_points = []
    if want_points:
        for pt in (parent_map.get("points") or []):
            pos = pt.get("position") or {}
            if "x" not in pos or "y" not in pos:
                continue
            if not point_in_window(pos["x"], pos["y"], window):
                continue
            new_pt = dict(pt)
            nx, ny = rescale_point(pos["x"], pos["y"])
            new_pt["position"] = {"x": nx, "y": ny}
            new_pt["id"] = str(uuid.uuid4())
            kept_points.append(new_pt)

    boundary = None
    if want_boundary:
        boundary = rescale_vertices(clip_area.get("vertices") or [], rescale_point)
        if want_clip:
            boundary = clip_to_unit_square(boundary)

    # Scale (regional maps): the child covers a fraction W of the parent width.
    scale = None
    parent_scale = parent_map.get("scale")
    if parent_scale and parent_scale.get("widthMeasurement") is not None:
        scale = dict(parent_scale)
        scale["widthMeasurement"] = parent_scale["widthMeasurement"] * w

    child = {
        "aspectRatio": child_aspect(parent_map, w, h),
        "scale": scale,
        "background": None,
        "boundaryPolygon": boundary,
        "areas": kept_areas,
        "points": kept_points,
        "overlayImages": [],
        "grid": parent_map.get("grid"),  # best-effort copy; see module docstring
        "clusteringEnabled": True,
    }
    return child, kept_areas, kept_points, skipped


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Clip a region out of a Craft parent map into its own child map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("cdf", help="path to the packed .cdf.json (edited in place)")
    ap.add_argument("--parent", required=True,
                    help="name or referenceId of the parent map location")
    ap.add_argument("--clip", required=True,
                    help="name or referenceId of the area/location to zoom into")
    ap.add_argument("--include", nargs="*", default=[],
                    help="names or referenceIds of neighbour locations to carry over")
    ap.add_argument("--padding", type=float, default=0.05,
                    help="fraction of the clip bbox added on each side (default 0.05)")
    ap.add_argument("--padding-km", type=float, default=None,
                    help="absolute padding in km on each side, converted via the "
                         "parent map scale (isotropic; overrides --padding)")
    ap.add_argument("--boundary", action="store_true",
                    help="write the clip polygon as the child map's boundaryPolygon")
    ap.add_argument("--no-points", dest="points", action="store_false",
                    help="do not carry parent pins/tokens into the child map")
    ap.add_argument("--no-clip", dest="clip", action="store_false",
                    help="keep whole polygons unclipped (Craft's importer rejects "
                         "out-of-range vertices; default is to clip to the window)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print summary + crop rect, write nothing")
    args = ap.parse_args(argv)

    cdf = load_cdf(args.cdf)

    parent = resolve_location(cdf, args.parent)
    if parent is None:
        raise SystemExit(f"error: parent location {args.parent!r} not found")
    parent_map = get_map(parent)
    if not parent_map:
        raise SystemExit(
            f"error: parent {args.parent!r} has no metadata.map (not a map location)"
        )

    clip = resolve_location(cdf, args.clip)
    if clip is None:
        raise SystemExit(f"error: clip location {args.clip!r} not found")
    clip_ref = clip.get("referenceId")

    clip_areas = [a for a in (parent_map.get("areas") or [])
                  if a.get("fileReferenceId") == clip_ref]
    if not clip_areas:
        raise SystemExit(
            f"error: clip {args.clip!r} is not an area on the parent map "
            f"{args.parent!r}"
        )
    if len(clip_areas) > 1:
        eprint(f"warning: clip {args.clip!r} appears {len(clip_areas)} times on "
               f"the parent map; using the union of all their vertices for the bbox")
    clip_verts = [v for a in clip_areas for v in (a.get("vertices") or [])]
    if not clip_verts:
        raise SystemExit(f"error: clip area {args.clip!r} has no vertices")

    clip_bbox = bbox(clip_verts)
    if args.padding_km is not None:
        pad_x, pad_y = window_padding_km(parent_map, args.padding_km)
        window = padded_window_abs(clip_bbox, pad_x, pad_y)
    else:
        window = padded_window(clip_bbox, args.padding)
    rescale_point, w, h = make_rescaler(window)

    # For boundaryPolygon we use the first matching clip area's own polygon.
    child_map, kept_areas, kept_points, skipped = build_child_map(
        cdf, parent_map, clip_areas[0], args.include, window,
        rescale_point, w, h, args.boundary, args.points, args.clip,
    )

    # --- report ---------------------------------------------------------- #
    wmin, hmin, wmax, hmax = window
    print(f"parent : {(parent.get('content') or {}).get('name')!r}  "
          f"({parent.get('referenceId')})")
    print(f"clip   : {(clip.get('content') or {}).get('name')!r}  ({clip_ref})")
    print(f"clip bbox (parent coords): "
          f"x {clip_bbox[0]:.4f}..{clip_bbox[2]:.4f}  "
          f"y {clip_bbox[1]:.4f}..{clip_bbox[3]:.4f}")
    print("CROP RECTANGLE (parent-normalised; use to crop the overland image):")
    print(f"  x,y,w,h        = {wmin:.4f}, {hmin:.4f}, {w:.4f}, {h:.4f}")
    print(f"  minx,miny,maxx,maxy = {wmin:.4f}, {hmin:.4f}, {wmax:.4f}, {hmax:.4f}")
    print(f"child aspectRatio : {child_map['aspectRatio']}")
    if child_map["scale"]:
        print(f"child scale       : {child_map['scale']}")
    print(f"areas kept  : {len(kept_areas)}")
    for a in kept_areas:
        loc = next((f for f in cdf['files']
                    if f.get('referenceId') == a['fileReferenceId']), None)
        name = (loc.get('content') or {}).get('name') if loc else '?'
        print(f"    + {name!r}")
    print(f"points kept : {len(kept_points)}")
    if skipped:
        print(f"skipped     : {len(skipped)}")
        for token, why in skipped:
            print(f"    - {token!r}: {why}")
    print(f"boundaryPolygon : {'set' if child_map['boundaryPolygon'] else 'null'}")

    # --- write ----------------------------------------------------------- #
    clip.setdefault("metadata", {})
    schema_version = (
        clip["metadata"].get("schemaVersionId")
        or (parent.get("metadata") or {}).get("schemaVersionId")
        or DEFAULT_SCHEMA_VERSION
    )
    clip["metadata"]["schemaVersionId"] = schema_version
    clip["metadata"]["map"] = child_map

    if args.dry_run:
        print("\n[dry-run] no files written")
        return 0

    backup = args.cdf + ".bak"
    shutil.copyfile(args.cdf, backup)
    with open(args.cdf, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(cdf, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"\nwrote {args.cdf}  (backup: {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
