"""Supabase persistence layer for xBD damage labels."""
from __future__ import annotations

import os
from typing import Literal

import streamlit as st
from supabase import create_client, Client

DamageState = Literal["intact", "severely_damaged", "destroyed"]
TileStatus  = Literal["pending", "labeled", "skipped"]


@st.cache_resource
def get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


# ── Tile management ───────────────────────────────────────────────────────────

def upsert_tile(tile_id: str, event: str, country: str,
                pre_date: str, post_date: str) -> None:
    get_client().table("tiles").upsert(
        {"id": tile_id, "event": event, "country": country,
         "pre_date": pre_date, "post_date": post_date},
        on_conflict="id",
    ).execute()


def set_tile_status(tile_id: str, status: TileStatus) -> None:
    from datetime import datetime, timezone
    get_client().table("tiles").update(
        {"status": status,
         "labeled_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", tile_id).execute()


def get_reviewed_ids() -> set[str]:
    """Return set of tile_ids that are already labeled or skipped."""
    rows = _fetch_all("tiles", filters=None, columns="id,status")
    return {r["id"] for r in rows if r["status"] in ("labeled", "skipped")}


# ── Building labels ───────────────────────────────────────────────────────────

def save_labels(tile_id: str,
                instances: list[dict],
                states: dict[int, str]) -> None:
    """Upsert one row per building instance (including intact ones)."""
    rows = []
    for inst in instances:
        idx = inst["idx"]
        rows.append({
            "tile_id":      tile_id,
            "building_idx": idx,
            "damage_state": states.get(idx, "intact"),
            "centroid_x":   inst["centroid_x"],
            "centroid_y":   inst["centroid_y"],
            "area_px":      inst["area"],
        })
    if rows:
        get_client().table("building_labels").upsert(
            rows, on_conflict="tile_id,building_idx"
        ).execute()


def get_labels(tile_id: str) -> dict[int, str]:
    """Return {building_idx: damage_state} for a tile (for resuming)."""
    resp = get_client().table("building_labels") \
        .select("building_idx,damage_state") \
        .eq("tile_id", tile_id).execute()
    return {r["building_idx"]: r["damage_state"] for r in (resp.data or [])}


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    rows = _fetch_all("tiles", filters=None, columns="status")
    counts = {"pending": 0, "labeled": 0, "skipped": 0}
    for r in rows:
        counts[r.get("status", "pending")] += 1
    return counts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_all(table: str, filters, columns: str = "*") -> list[dict]:
    """Paginate through Supabase 1000-row limit."""
    results, offset = [], 0
    while True:
        q = get_client().table(table).select(columns).range(offset, offset + 999)
        resp = q.execute()
        batch = resp.data or []
        results.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return results
