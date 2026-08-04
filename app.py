"""Streamlit application: pick a match, see its shot map and expected goals.

Run from the project root:

    streamlit run app.py

The model is loaded from models/xg_model.joblib and never retrained here.
Run train.py first if that file is missing.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import streamlit as st
from mplsoccer import VerticalPitch

import pipeline

ARTEFACT = Path("models") / "xg_model.joblib"

PITCH_COLOR = "#1b2838"
HOME_COLOR = "#ef476f"
AWAY_COLOR = "#06d6a0"

st.set_page_config(page_title="xG Football", page_icon="⚽", layout="centered")


@st.cache_resource
def load_model():
    """Loaded once per server, not once per interaction."""
    return joblib.load(ARTEFACT)


@st.cache_data(show_spinner=False)
def load_match_list(competition_id, season_id):
    return pipeline.match_list(competition_id, season_id)


@st.cache_data(show_spinner=False)
def load_shots(match_id):
    raw = pipeline.get_shots(match_id)
    return pipeline.prepare(raw, drop_penalties=False)


def add_xg(shots, artefact):
    """Model probability for open play, a fixed value for penalties."""
    shots = shots.copy()
    shots["xg"] = pipeline.PENALTY_XG

    open_play = shots[shots["is_penalty"] == 0]
    if not open_play.empty:
        features = pipeline.build_features(
            open_play,
            columns=artefact["columns"],
            keeper_median=artefact["keeper_median"],
        )
        shots.loc[open_play.index, "xg"] = artefact["model"].predict_proba(features)[:, 1]

    return shots


def shot_map(shots, home, away):
    pitch = VerticalPitch(
        pitch_type="statsbomb",
        half=True,
        pitch_color=PITCH_COLOR,
        line_color="white",
        linewidth=1.5,
    )
    fig, ax = pitch.draw(figsize=(7, 7))
    fig.set_facecolor(PITCH_COLOR)

    for team, color in ((home, HOME_COLOR), (away, AWAY_COLOR)):
        side = shots[shots["team_name"] == team]
        if side.empty:
            continue
        pitch.scatter(
            side["x"],
            side["y"],
            s=side["xg"] * 600 + 25,
            color=color,
            edgecolors="white",
            linewidth=0.5,
            alpha=0.85,
            ax=ax,
            label=f"{team}   {side['xg'].sum():.2f} xG   {int(side['is_goal'].sum())} goal(s)",
        )

    scored = shots[shots["is_goal"] == 1]
    if not scored.empty:
        pitch.scatter(
            scored["x"],
            scored["y"],
            s=scored["xg"] * 600 + 25,
            facecolors="none",
            edgecolors="white",
            linewidth=2.2,
            ax=ax,
        )

    ax.legend(
        loc="lower center",
        labelcolor="white",
        facecolor=PITCH_COLOR,
        edgecolor="none",
        fontsize=9,
    )
    return fig


st.title("Expected goals")
st.caption(
    "Shot maps built from StatsBomb open data. Marker size is proportional to "
    "the chance of scoring. A white ring means the shot went in."
)

if not ARTEFACT.exists():
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

artefact = load_model()

competition = st.selectbox(
    "Competition",
    options=pipeline.COMPETITIONS,
    format_func=lambda key: pipeline.COMPETITION_NAMES[key],
)

matches = load_match_list(*competition)
row = st.selectbox(
    "Match",
    options=matches.index,
    format_func=lambda i: f"{matches.at[i, 'date']}   {matches.at[i, 'label']}",
)

selected = matches.loc[row]

with st.spinner("Loading match events..."):
    shots = add_xg(load_shots(int(selected["match_id"])), artefact)

home_xg = shots.loc[shots["team_name"] == selected["home"], "xg"].sum()
away_xg = shots.loc[shots["team_name"] == selected["away"], "xg"].sum()

left, right = st.columns(2)
left.metric(f"{selected['home']} xG", f"{home_xg:.2f}", f"{int(selected['home_score'])} scored")
right.metric(f"{selected['away']} xG", f"{away_xg:.2f}", f"{int(selected['away_score'])} scored")

st.pyplot(shot_map(shots, selected["home"], selected["away"]))
plt.close("all")

st.subheader("Every shot")
table = (
    shots[["minute", "team_name", "player_name", "distance", "angle", "xg", "is_goal", "is_penalty"]]
    .sort_values("xg", ascending=False)
    .rename(
        columns={
            "minute": "Min",
            "team_name": "Team",
            "player_name": "Player",
            "distance": "Distance",
            "angle": "Angle",
            "xg": "xG",
            "is_goal": "Goal",
            "is_penalty": "Penalty",
        }
    )
)
table["Goal"] = table["Goal"].astype(bool)
table["Penalty"] = table["Penalty"].astype(bool)
st.dataframe(
    table.style.format({"Distance": "{:.1f}", "Angle": "{:.1f}", "xG": "{:.3f}"}),
    hide_index=True,
    use_container_width=True,
)

st.caption("Penalties are excluded from the model and shown at a fixed 0.76. Data by StatsBomb.")
