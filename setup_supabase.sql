-- Run this once in the Supabase SQL editor to create the schema.

CREATE TABLE IF NOT EXISTS tiles (
    id          TEXT PRIMARY KEY,
    event       TEXT NOT NULL,
    country     TEXT,
    pre_date    TEXT,
    post_date   TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending | labeled | skipped
    labeled_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_tiles_status  ON tiles (status);
CREATE INDEX IF NOT EXISTS idx_tiles_event   ON tiles (event);

CREATE TABLE IF NOT EXISTS building_labels (
    id            BIGSERIAL PRIMARY KEY,
    tile_id       TEXT        NOT NULL REFERENCES tiles(id) ON DELETE CASCADE,
    building_idx  INTEGER     NOT NULL,
    damage_state  TEXT        NOT NULL,   -- intact | severely_damaged | destroyed
    centroid_x    INTEGER,
    centroid_y    INTEGER,
    area_px       INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tile_id, building_idx)
);

CREATE INDEX IF NOT EXISTS idx_bl_tile_id ON building_labels (tile_id);
