from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")

st.set_page_config(
    page_title="Mental Health in Tech Survey",
    page_icon="🧠",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "survey.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)

    for column, value in {
        "state": "Unknown",
        "self_employed": "Unknown",
        "work_interfere": "Unknown",
        "comments": "No comment",
    }.items():
        if column in data.columns:
            data[column] = data[column].fillna(value)

    data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
    data = data[(data["Age"] >= 18) & (data["Age"] <= 100)].copy()

    def clean_gender(value):
        if pd.isna(value):
            return "Other"
        value = str(value).strip().lower()
        if value in {"male", "m", "man", "cis male", "cis man", "male-ish"}:
            return "Male"
        if value in {"female", "f", "woman", "cis female", "cis woman"}:
            return "Female"
        return "Other"

    data["Gender_Clean"] = data["Gender"].apply(clean_gender)
    data["Age_Group"] = pd.cut(
        data["Age"],
        bins=[17, 24, 34, 44, 54, 100],
        labels=["18–24", "25–34", "35–44", "45–54", "55+"],
    )
    return data


def count_plot(frame, x=None, y=None, hue=None, title="", order=None):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.countplot(
        data=frame,
        x=x,
        y=y,
        hue=hue,
        order=order,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title() if x else "")
    ax.set_ylabel("Number of Respondents")
    plt.xticks(rotation=20)
    fig.tight_layout()
    return fig


try:
    data = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load survey.csv: {exc}")
    st.stop()

st.sidebar.header("Filters")

def multi_filter(label, column):
    options = sorted(data[column].dropna().astype(str).unique().tolist())
    return st.sidebar.multiselect(label, options, default=options)

gender_filter = multi_filter("Gender", "Gender_Clean")
treatment_filter = multi_filter("Treatment", "treatment")
remote_filter = multi_filter("Remote work", "remote_work")

filtered = data[
    data["Gender_Clean"].isin(gender_filter)
    & data["treatment"].isin(treatment_filter)
    & data["remote_work"].isin(remote_filter)
].copy()

st.title("Mental Health in Tech Survey")
st.caption(
    "Interactive exploratory data analysis dashboard. "
    "Charts describe patterns in this dataset and do not establish causation."
)

total = len(filtered)
yes_count = int((filtered["treatment"] == "Yes").sum())
no_count = int((filtered["treatment"] == "No").sum())
tech_count = int((filtered["tech_company"] == "Yes").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Respondents", f"{total:,}")
c2.metric("Treatment: Yes", f"{yes_count:,}")
c3.metric("Treatment: No", f"{no_count:,}")
c4.metric("Technology companies", f"{tech_count:,}")

st.divider()

if filtered.empty:
    st.warning("No records match the selected filters. Please broaden your selections.")
    st.stop()

left, right = st.columns(2)

with left:
    st.subheader("Gender distribution")
    fig = count_plot(
        filtered,
        x="Gender_Clean",
        title="Gender Distribution",
        order=filtered["Gender_Clean"].value_counts().index.tolist(),
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.subheader("Treatment distribution")
    fig = count_plot(
        filtered,
        x="treatment",
        title="Mental Health Treatment",
        order=filtered["treatment"].value_counts().index.tolist(),
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

left, right = st.columns(2)

with left:
    st.subheader("Family history vs treatment")
    fig = count_plot(
        filtered,
        x="family_history",
        hue="treatment",
        title="Family History vs Treatment",
        order=filtered["family_history"].value_counts().index.tolist(),
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.subheader("Work interference vs treatment")
    fig = count_plot(
        filtered,
        x="work_interfere",
        hue="treatment",
        title="Work Interference vs Treatment",
        order=filtered["work_interfere"].value_counts().index.tolist(),
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

left, right = st.columns(2)

with left:
    st.subheader("Benefits vs treatment")
    fig = count_plot(
        filtered,
        x="benefits",
        hue="treatment",
        title="Benefits vs Treatment",
        order=filtered["benefits"].value_counts().index.tolist(),
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.subheader("Remote work vs treatment")
    fig = count_plot(
        filtered,
        x="remote_work",
        hue="treatment",
        title="Remote Work vs Treatment",
        order=filtered["remote_work"].value_counts().index.tolist(),
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.subheader("Treatment rate by family history")
rate_table = (
    filtered.groupby("family_history")["treatment"]
    .apply(lambda values: (values == "Yes").mean() * 100)
    .round(2)
    .reset_index(name="Treatment rate (%)")
    .sort_values("Treatment rate (%)", ascending=False)
)
st.dataframe(rate_table, use_container_width=True, hide_index=True)

st.subheader("Filtered data preview")
st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered CSV",
    data=csv_bytes,
    file_name="filtered_mental_health_survey.csv",
    mime="text/csv",
)
