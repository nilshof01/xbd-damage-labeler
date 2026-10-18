# xBD Damage Labeler

Label building damage on xBD post-disaster imagery.  
The UNet model detects buildings on the pre-disaster image; borders are overlaid on the post-disaster image for you to click.

## How it works

1. UNet (`nilsho01/unet-buildings-autorun-v2`) runs on the **pre-disaster** image → building instance masks
2. Coloured borders are drawn on the **post-disaster** image
3. Click a building to cycle its damage state:
   - 🟢 Intact (default)
   - 🟠 Severely damaged (1 click)
   - 🔴 Destroyed (2 clicks)
   - back to intact (3 clicks)
4. **Save & Next** writes labels to Supabase and loads the next tile
5. Tiles are served stratified by disaster event (geographic diversity from the start)

---

## 1 — Supabase setup

1. Go to [supabase.com](https://supabase.com) → New project
2. Once created, go to **SQL Editor** and paste + run the contents of `setup_supabase.sql`
3. Go to **Project Settings → API**:
   - Copy **Project URL** → `SUPABASE_URL`
   - Copy **service_role** secret key → `SUPABASE_KEY`

---

## 2 — Local setup

```bash
# Clone / copy this folder, then:
cd xbd-damage-labeler

# Install dependencies
pip install -r requirements.txt

# Create your .env from the example
cp .env.example .env
# Edit .env and fill in SUPABASE_URL, SUPABASE_KEY, XBD_DIR

# Run
streamlit run app.py
```

The app loads environment variables from `.env` automatically via `python-dotenv`.  
On first run it downloads the UNet weights from HuggingFace (~80 MB) and caches them locally.

---

## 3 — Streamlit Secrets (for deployment / sharing)

If you deploy to [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub
2. In the Streamlit Cloud dashboard → **Secrets**:
```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your-service-role-key"
XBD_DIR      = "/path/to/xbd/train"
```
3. The xBD images must be accessible at `XBD_DIR` on the server.  
   For local-only use, just run `streamlit run app.py` locally.

---

## Output schema

**`tiles`** — one row per xBD tile  
| column | type | notes |
|--------|------|-------|
| id | text | e.g. `hurricane-matthew_00000123` |
| event | text | disaster event name |
| country | text | derived from event |
| pre_date / post_date | text | image capture dates |
| status | text | `pending` / `labeled` / `skipped` |

**`building_labels`** — one row per building instance  
| column | type | notes |
|--------|------|-------|
| tile_id | text | FK → tiles.id |
| building_idx | int | instance index from connected components |
| damage_state | text | `intact` / `severely_damaged` / `destroyed` |
| centroid_x/y | int | pixel coordinates in original 1024×1024 image |
| area_px | int | building footprint area in pixels |

---

## Notes

- Instance segmentation is approximate: connected components on the UNet binary mask.  
  Adjacent touching buildings may merge into one instance.
- The model was trained on pre-disaster clean imagery; post-disaster inference would be less reliable — which is exactly why we only run it on pre-disaster images.
- Labels survive page refresh: the app restores saved labels when you return to a tile.
