"""Data pipeline for the xG model.

Fetches StatsBomb open data, cleans it and builds the model features.
Knows nothing about the model itself.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# StatsBomb pitch is 120 x 80. The attacked goal is always at x = 120.
GOAL_X = 120.0
GOAL_Y = 40.0
POST_LOW = (120.0, 36.0)
POST_HIGH = (120.0, 44.0)

# In-play penalties are excluded from the model and given this fixed value
# for display, since their probability does not depend on any measured variable.
PENALTY_XG = 0.76

# (competition_id, season_id) pairs used to train the model.
COMPETITIONS = [
    (43, 106),    # FIFA World Cup 2022
    (43, 3),      # FIFA World Cup 2018
    (55, 282),    # UEFA Euro 2024
    (55, 43),     # UEFA Euro 2020
    (223, 282),   # Copa America 2024
    (1267, 107),  # African Cup of Nations 2023
]

COMPETITION_NAMES = {
    (43, 106): "FIFA World Cup 2022",
    (43, 3): "FIFA World Cup 2018",
    (55, 282): "UEFA Euro 2024",
    (55, 43): "UEFA Euro 2020",
    (223, 282): "Copa America 2024",
    (1267, 107): "African Cup of Nations 2023",
}

MODEL_INPUTS = [
    "distance",
    "angle",
    "under_pressure",
    "defenseurs_cone",
    "gardien_dans_cone",
    "gardien_dist_but",
    "adversaire_proche",
    "body_part",
    "shot_type",
    "technique",
]

# Categorical variables and the reference category dropped by get_dummies.
# The most frequent value is used, so coefficients read against a common case.
CATEGORICALS = {
    "shot_type": "Open Play",
    "body_part": "Right Foot",
    "technique": "Normal",
}


# downloading


def match_ids(competition_id, season_id):
    """Return the list of match ids for one competition and season."""
    url = f"{BASE_URL}/matches/{competition_id}/{season_id}.json"
    return pd.read_json(url)["match_id"].tolist()


def match_list(competition_id, season_id):
    """Return one row per match with a readable label, oldest first."""
    url = f"{BASE_URL}/matches/{competition_id}/{season_id}.json"
    matches = pd.read_json(url)

    listing = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["match_date"],
            "home": pd.json_normalize(matches["home_team"])["home_team_name"],
            "away": pd.json_normalize(matches["away_team"])["away_team_name"],
            "home_score": matches["home_score"],
            "away_score": matches["away_score"],
        }
    )

    listing["label"] = (
        listing["home"]
        + "  "
        + listing["home_score"].astype(str)
        + " - "
        + listing["away_score"].astype(str)
        + "  "
        + listing["away"]
    )

    return listing.sort_values("date").reset_index(drop=True)


def all_match_ids(competitions=None):
    """Return every match id across the given competitions."""
    competitions = COMPETITIONS if competitions is None else competitions
    ids = []
    for competition_id, season_id in competitions:
        ids.extend(match_ids(competition_id, season_id))
    return ids


def get_shots(match_id):
    """Return the raw shot events of one match, untouched."""
    url = f"{BASE_URL}/events/{match_id}.json"
    events = pd.read_json(url)
    events["type_name"] = pd.json_normalize(events["type"])["name"]
    shots = events[events["type_name"] == "Shot"].copy()
    shots["match_id"] = match_id
    return shots


def get_shots_many(ids, workers=8):
    """Download several matches in parallel and stack the results.

    Matches that fail are skipped rather than aborting the whole run.
    """

    def try_one(match_id):
        try:
            return get_shots(match_id)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(try_one, ids))

    kept = [part for part in parts if part is not None]
    return pd.concat(kept, ignore_index=True)


#- geometry


def _side(point, start, end):
    """Signed area telling which side of the line start-end the point sits on."""
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def in_triangle(point, a, b, c):
    """True when the point lies inside the triangle a-b-c."""
    d1 = _side(point, a, b)
    d2 = _side(point, b, c)
    d3 = _side(point, c, a)
    has_negative = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_positive = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_negative and has_positive)


def _shot_angle(x, y):
    """Angle in degrees subtended by the two posts at the shooting position."""
    ux = POST_LOW[0] - x
    uy = POST_LOW[1] - y
    vx = POST_HIGH[0] - x
    vy = POST_HIGH[1] - y
    cosine = (ux * vx + uy * vy) / (np.hypot(ux, uy) * np.hypot(vx, vy))
    # Rounding can push the cosine a hair outside [-1, 1], which arccos rejects.
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _freeze_frame_features(row):
    """Opponent positions at the moment of the shot, summarised in four numbers."""
    frame = row["shot"].get("freeze_frame") or []
    shooter = (row["x"], row["y"])

    defenders = 0
    keeper_in_cone = 0
    keeper_x, keeper_y = np.nan, np.nan
    nearest = np.nan

    for player in frame:
        if player["teammate"]:
            continue

        location = player["location"]
        inside = in_triangle(location, shooter, POST_LOW, POST_HIGH)

        if player["position"]["name"] == "Goalkeeper":
            keeper_x, keeper_y = location
            keeper_in_cone = int(inside)
        elif inside:
            defenders += 1

        distance = np.hypot(location[0] - shooter[0], location[1] - shooter[1])
        if np.isnan(nearest) or distance < nearest:
            nearest = distance

    return pd.Series(
        {
            "defenseurs_cone": defenders,
            "gardien_dans_cone": keeper_in_cone,
            "gardien_dist_but": np.hypot(GOAL_X - keeper_x, GOAL_Y - keeper_y),
            "adversaire_proche": nearest,
        }
    )


#preparation


def prepare(shots, drop_penalties=True):
    """Turn raw shot events into a modelling table.

    Penalty shootouts are always removed. In-play penalties are removed for
    training and kept for display, which is what drop_penalties controls.
    """
    shots = shots[shots["period"] != 5].reset_index(drop=True)

    details = pd.json_normalize(shots["shot"])
    shots["shot_type"] = details["type.name"]
    shots["body_part"] = details["body_part.name"]
    shots["technique"] = details["technique.name"]
    shots["outcome"] = details["outcome.name"]
    shots["xg_statsbomb"] = details["statsbomb_xg"]

    shots["is_penalty"] = (shots["shot_type"] == "Penalty").astype(int)
    if drop_penalties:
        shots = shots[shots["is_penalty"] == 0].reset_index(drop=True)

    shots["x"] = shots["location"].str[0]
    shots["y"] = shots["location"].str[1]
    shots["distance"] = np.hypot(GOAL_X - shots["x"], GOAL_Y - shots["y"])
    shots["angle"] = _shot_angle(shots["x"], shots["y"])

    shots["under_pressure"] = shots["under_pressure"].fillna(0).astype(int)
    shots["is_goal"] = (shots["outcome"] == "Goal").astype(int)
    shots["team_name"] = pd.json_normalize(shots["team"])["name"]
    shots["player_name"] = pd.json_normalize(shots["player"])["name"]

    freeze = shots.apply(_freeze_frame_features, axis=1)
    for column in freeze.columns:
        shots[column] = freeze[column]

    return shots


# features


def _reference_first(series, reference):
    """Order the categories so get_dummies drops the reference we chose."""
    others = sorted(value for value in series.dropna().unique() if value != reference)
    return pd.Categorical(series, categories=[reference] + others)


def build_features(df, columns=None, keeper_median=None):
    """One-hot encode the categorical variables and return the model matrix.

    Passing columns forces the exact set and order the model was trained on,
    which protects against a category missing from a small batch of shots.
    """
    tmp = df.copy()

    if keeper_median is not None:
        tmp["gardien_dist_but"] = tmp["gardien_dist_but"].fillna(keeper_median)

    for column, reference in CATEGORICALS.items():
        tmp[column] = _reference_first(tmp[column], reference)

    X = pd.get_dummies(tmp[MODEL_INPUTS], columns=list(CATEGORICALS), drop_first=True)

    if columns is not None:
        X = X.reindex(columns=columns, fill_value=False)

    return X
