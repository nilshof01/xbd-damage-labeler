"""xBD Building Damage Labeler — Streamlit app.

Workflow:
  1. UNet runs on the pre-disaster image to detect building instances.
  2. Coloured borders are drawn on the post-disaster image.
  3. Annotator clicks a building:
       1st click -> severely_damaged (orange)
       2nd click -> destroyed (red)
       3rd click -> back to intact (green)
  4. Save & Next commits labels to Supabase and loads the next tile.

Run:
    streamlit run app.py

Environment variables (set in .env or Streamlit secrets):
    SUPABASE_URL, SUPABASE_KEY, XBD_DIR
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

import db
import data as xbd_data
import inference as inf

# ── Config ────────────────────────────────────────────────────────────────────

XBD_DIR      = xbd_data.get_xbd_dir()
DISPLAY_SIZE = 600   # px — both images shown at this width
DEVICE       = "cpu"

DAMAGE_CYCLE = ["intact", "severely_damaged", "destroyed"]

st.set_page_config(page_title="xBD Damage Labeler", layout="wide",
                   initial_sidebar_state="expanded")


# ── Data + model loading (cached) ─────────────────────────────────────────────

TILE_CACHE = Path(__file__).parent / "tile_order_cache.json"

@st.cache_resource
def load_tile_order():
    if TILE_CACHE.exists():
        with open(TILE_CACHE) as f:
            raw = json.load(f)
        return [xbd_data.Tile(
            **{k: Path(v) if k.endswith("_path") else v for k, v in r.items()}
        ) for r in raw]

    tiles = xbd_data.scan_tiles(XBD_DIR)
    ordered = xbd_data.stratify_tiles(tiles)
    with open(TILE_CACHE, "w") as f:
        json.dump([{k: str(v) if isinstance(v, Path) else v
                    for k, v in asdict(t).items()} for t in ordered], f)
    return ordered


@st.cache_resource
def load_model():
    with st.spinner("Loading UNet model (first run only)..."):
        return inf.load_model(device=DEVICE)


# ── Session state helpers ─────────────────────────────────────────────────────

def init_session():
    if "cursor" not in st.session_state:
        reviewed = db.get_reviewed_ids()
        tiles = load_tile_order()
        # sync tiles to Supabase (idempotent)
        for t in tiles:
            db.upsert_tile(t.id, t.event, t.country, t.pre_date, t.post_date)
        st.session_state.tiles         = tiles
        st.session_state.reviewed      = reviewed
        st.session_state.cursor        = _first_pending(tiles, reviewed)
        st.session_state.tile_cache_id = None   # which tile is in inference cache
        st.session_state.labeled       = {}     # {building_idx: damage_state}
        st.session_state.history       = []     # [(idx, old_state)] for undo
        st.session_state.last_click    = None


def _first_pending(tiles, reviewed):
    for i, t in enumerate(tiles):
        if t.id not in reviewed:
            return i
    return len(tiles)


def _advance():
    tiles, reviewed = st.session_state.tiles, st.session_state.reviewed
    for i in range(st.session_state.cursor + 1, len(tiles)):
        if tiles[i].id not in reviewed:
            st.session_state.cursor = i
            return
    st.session_state.cursor = len(tiles)  # done


def _load_tile_inference(tile):
    """Run inference if this is a new tile; cache result in session state."""
    if st.session_state.tile_cache_id == tile.id:
        return  # already cached
    model, n_out = load_model()
    pre_img, _ = xbd_data.load_image_pair(tile)
    with st.spinner("Detecting buildings on pre-disaster image..."):
        binary_mask = inf.predict_mask(pre_img, model, n_out, DEVICE)
        labeled, instances = inf.extract_instances(binary_mask)
    # Restore any previously saved labels for this tile
    saved = db.get_labels(tile.id)
    st.session_state.tile_cache_id = tile.id
    st.session_state.instances     = instances
    st.session_state.labeled       = dict(saved)
    st.session_state.history       = []
    st.session_state.last_click    = None
    st.session_state.labeled_arr   = labeled


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.title("xBD Damage Labeler")
    stats = db.get_stats()
    total = sum(stats.values())
    done  = stats["labeled"] + stats["skipped"]
    st.sidebar.progress(done / max(total, 1), text=f"{done}/{total} reviewed")
    st.sidebar.metric("Labeled",  stats["labeled"])
    st.sidebar.metric("Skipped",  stats["skipped"])
    st.sidebar.metric("Pending",  stats["pending"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
**Controls**
- **Click once** → 🟠 Severely damaged
- **Click twice** → 🔴 Destroyed
- **Click 3×** → 🟢 Reset to intact
""")
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
**Keyboard shortcuts**
- `S` = Skip
- `Enter` = Save & Next
""")


# ── Click handler ─────────────────────────────────────────────────────────────

def handle_click(cx: int, cy: int, display_w: int, display_h: int):
    labeled_arr = st.session_state.labeled_arr
    img_h, img_w = labeled_arr.shape

    # Scale display coords -> image coords
    ix = int(cx * img_w / display_w)
    iy = int(cy * img_h / display_h)
    ix = max(0, min(ix, img_w - 1))
    iy = max(0, min(iy, img_h - 1))

    building_idx = int(labeled_arr[iy, ix])
    if building_idx == 0:
        return  # clicked background

    old_state = st.session_state.labeled.get(building_idx, "intact")
    new_state = DAMAGE_CYCLE[(DAMAGE_CYCLE.index(old_state) + 1) % len(DAMAGE_CYCLE)]
    st.session_state.history.append((building_idx, old_state))
    st.session_state.labeled[building_idx] = new_state


# ── Main render ───────────────────────────────────────────────────────────────

def render_main():
    tiles  = st.session_state.tiles
    cursor = st.session_state.cursor

    if cursor >= len(tiles):
        st.success("🎉 All tiles reviewed!")
        return

    tile = tiles[cursor]
    _load_tile_inference(tile)

    # ── Top bar ───────────────────────────────────────────────────────────────
    st.markdown(f"### `{tile.id}` &nbsp;&nbsp; {tile.country} — {tile.event}")
    progress_text = (f"Tile {cursor + 1} / {len(tiles)}  |  "
                     f"{len(st.session_state.instances)} buildings detected")
    st.caption(progress_text)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 5])
    with c1:
        skip = st.button("⏭ Skip", use_container_width=True)
    with c2:
        undo_last = st.button("↩ Undo", use_container_width=True)
    with c3:
        undo_all = st.button("↩↩ All", use_container_width=True)
    with c4:
        save = st.button("✓ Save & Next", type="primary", use_container_width=True)

    if skip:
        db.set_tile_status(tile.id, "skipped")
        st.session_state.reviewed.add(tile.id)
        _advance()
        st.rerun()

    if undo_last and st.session_state.history:
        idx, old = st.session_state.history.pop()
        if old == "intact":
            st.session_state.labeled.pop(idx, None)
        else:
            st.session_state.labeled[idx] = old

    if undo_all:
        st.session_state.labeled  = {}
        st.session_state.history  = []

    if save:
        db.save_labels(tile.id, st.session_state.instances,
                       st.session_state.labeled)
        db.set_tile_status(tile.id, "labeled")
        st.session_state.reviewed.add(tile.id)
        _advance()
        st.rerun()

    # ── Images ────────────────────────────────────────────────────────────────
    pre_img, post_img = xbd_data.load_image_pair(tile)

    rendered_post = inf.draw_instance_borders(
        post_img,
        st.session_state.labeled_arr,
        st.session_state.labeled,
    )

    left, right = st.columns(2)

    with left:
        st.caption(f"**Pre-disaster** — {tile.pre_date}")
        st.image(pre_img.resize((DISPLAY_SIZE, DISPLAY_SIZE)))

    with right:
        st.caption(f"**Post-disaster** — {tile.post_date} &nbsp;|&nbsp; "
                   "click a building to label damage")
        coords = streamlit_image_coordinates(
            rendered_post.resize((DISPLAY_SIZE, DISPLAY_SIZE)),
            width=DISPLAY_SIZE,
            key=f"post_{tile.id}",
        )
        if coords and coords != st.session_state.last_click:
            st.session_state.last_click = coords
            handle_click(coords["x"], coords["y"], DISPLAY_SIZE, DISPLAY_SIZE)
            st.rerun()

    # ── Legend ────────────────────────────────────────────────────────────────
    st.markdown(
        "🟢 **Intact** &nbsp;&nbsp;&nbsp; "
        "🟠 **Severely damaged** &nbsp;&nbsp;&nbsp; "
        "🔴 **Destroyed**"
    )
    damaged = sum(1 for s in st.session_state.labeled.values()
                  if s != "intact")
    if damaged:
        st.info(f"{damaged} building(s) marked as damaged/destroyed")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    init_session()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
else:
    main()
