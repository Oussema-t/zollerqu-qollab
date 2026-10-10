"""Build data/champaign.json: Census 2020 block groups and OSRM driving times.

Run from the repo root: python tools/build_data.py
"""
import datetime
import json
import math
import os
import time

import requests

TIGER = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/"
         "MapServer/11/query")
OSRM = "https://router.project-osrm.org/table/v1/driving/"
HEADERS = {"User-Agent": "zollerqu-placement-hackathon/1.0 (data build script)"}
N_SITES, POOL, MIN_GAP, MAX_COORDS = 12, 40, 4.0, 100
HAV_KMH = 30.0
MIN_BLOCK_SITE = 0.5  # minutes; floor for a block whose centroid is its own site
OUT = os.path.join("data", "champaign.json")


def fetch_blocks():
    params = {"where": "STATE='17' AND COUNTY='019'",
              "outFields": "GEOID,CENTLAT,CENTLON,AREALAND,POP100,HU100",
              "returnGeometry": "false", "f": "json"}
    r = requests.get(TIGER, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    blocks = []
    for f in r.json()["features"]:
        a = f["attributes"]
        blocks.append({"geoid": a["GEOID"], "lat": float(a["CENTLAT"]),
                       "lon": float(a["CENTLON"]), "pop": int(a["POP100"]),
                       "area_m2": int(a["AREALAND"])})
    blocks.sort(key=lambda b: b["geoid"])
    return blocks


def haversine_min(p, q):
    (la1, lo1), (la2, lo2) = p, q
    a = (math.sin(math.radians(la2 - la1) / 2) ** 2 + math.cos(math.radians(la1))
         * math.cos(math.radians(la2)) * math.sin(math.radians(lo2 - lo1) / 2) ** 2)
    km = 2 * 6371.0 * math.asin(math.sqrt(a))
    return km / HAV_KMH * 60.0


def osrm_table(src, dst, state):
    """Minutes table len(src) x len(dst); sources and destinations are (lat, lon).
    Falls back to haversine for the whole table if OSRM fails."""
    coords = src + dst
    assert len(coords) <= MAX_COORDS
    url = OSRM + ";".join(f"{lo:.6f},{la:.6f}" for la, lo in coords)
    params = {"annotations": "duration",
              "sources": ";".join(map(str, range(len(src)))),
              "destinations": ";".join(str(len(src) + j) for j in range(len(dst)))}
    time.sleep(1.0)
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=120)
        j = r.json()
        if r.status_code != 200 or j.get("code") != "Ok" or not j.get("durations"):
            raise RuntimeError(f"OSRM status {r.status_code} code {j.get('code')}")
        return [[(v / 60.0 if v is not None else haversine_min(src[i], dst[k]))
                 for k, v in enumerate(row)] for i, row in enumerate(j["durations"])]
    except Exception as e:
        print("OSRM failed, using haversine:", e)
        state["osrm"] = False
        return [[haversine_min(p, q) for q in dst] for p in src]


def pick_sites(blocks, state):
    order = sorted(range(len(blocks)), key=lambda i: -blocks[i]["pop"])[:POOL]
    pts = [(blocks[i]["lat"], blocks[i]["lon"]) for i in order]
    t = osrm_table(pts, pts, state)
    chosen = []
    for k in range(len(order)):
        if all(max(t[k][c], t[c][k]) >= MIN_GAP and (t[k][c] + t[c][k]) / 2 >= MIN_GAP
               for c in chosen):
            chosen.append(k)
        if len(chosen) == N_SITES:
            break
    if len(chosen) < N_SITES:
        raise SystemExit(f"only {len(chosen)} sites satisfy the {MIN_GAP} min gap in the top {POOL}")
    return [order[k] for k in chosen]


def pick_warehouses(blocks):
    tot = sum(b["pop"] for b in blocks)
    clon = sum(b["pop"] * b["lon"] for b in blocks) / tot
    pop = [b for b in blocks if b["pop"] > 0]
    east = max(pop, key=lambda b: b["lon"] - clon)
    west = min(pop, key=lambda b: b["lon"] - clon)
    return [east, west]


def block_site_table(blocks, sites, state):
    chunk = MAX_COORDS - len(sites)
    out = []
    for s in range(0, len(blocks), chunk):
        part = [(b["lat"], b["lon"]) for b in blocks[s:s + chunk]]
        out += osrm_table(part, sites, state)
    return out


def main():
    state = {"osrm": True}
    now = datetime.date.today().isoformat()
    blocks = fetch_blocks()
    idx = pick_sites(blocks, state)
    sites = [{"id": k, "name": blocks[i]["geoid"], "lat": blocks[i]["lat"],
              "lon": blocks[i]["lon"]} for k, i in enumerate(idx)]
    spts = [(s["lat"], s["lon"]) for s in sites]
    wh = pick_warehouses(blocks)
    whs = [{"id": k, "lat": b["lat"], "lon": b["lon"]} for k, b in enumerate(wh)]

    t_ss = osrm_table(spts, spts, state)
    t_ss = [[0.0 if i == j else (t_ss[i][j] + t_ss[j][i]) / 2 for j in range(N_SITES)]
            for i in range(N_SITES)]
    t_ws = osrm_table([(w["lat"], w["lon"]) for w in whs], spts, state)
    t_bs = block_site_table(blocks, spts, state)

    rnd = lambda m: [[round(max(v, 0.0), 2) for v in row] for row in m]
    t_bs = [[max(v, MIN_BLOCK_SITE) for v in row] for row in t_bs]
    t_ws = [[max(v, MIN_BLOCK_SITE) for v in row] for row in t_ws]
    travel = ("OSRM demo server (router.project-osrm.org), driving durations, fetched "
              f"{now}; map data (c) OpenStreetMap contributors"
              if state["osrm"] else "haversine (OSRM unavailable)")
    data = {"county": "Champaign County, IL",
            "source": {"population": "US Census Bureau TIGERweb, Census 2020 Block Groups "
                                     f"(layer 11), fetched {now}", "travel": travel},
            "blocks": blocks, "sites": sites, "warehouses": whs,
            "t_block_site": rnd(t_bs), "t_site_site": rnd(t_ss), "t_wh_site": rnd(t_ws)}
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1)
        f.write("\n")

    print(f"blocks: {len(blocks)}  total pop: {sum(b['pop'] for b in blocks)}")
    for s, i in zip(sites, idx):
        print(f"site {s['id']:2d} {s['name']} pop {blocks[i]['pop']}")
    print("travel:", "OSRM" if state["osrm"] else "haversine")
    print(f"file: {OUT}, {os.path.getsize(OUT)} bytes")


if __name__ == "__main__":
    main()
