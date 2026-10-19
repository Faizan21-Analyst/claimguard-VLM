"""ClaimGuard reviewer screen.

The UI only displays what the API returns and sends the reviewer's action back.
It never decides anything, never reads files or the database, and never holds the Groq key.

Run:  streamlit run app/streamlit_app.py     (the API must be running too)
"""
from datetime import date
from typing import get_args

import streamlit as st

from claimguard.schemas.visual import Component
from claimguard.ui.api_client import ApiClient, ApiError
from claimguard.ui.formatting import (
    REVIEW_ACTION_LABELS,
    WEATHER_ATTRIBUTION,
    citation_role,
    citation_title,
    decision_label,
    describe_weather,
    finding_rows,
    format_time,
    history_rows,
    last_review_action,
    policy_basis,
    sort_citations,
    verdict_recap,
)
from claimguard.ui.settings import load_ui_settings

st.set_page_config(page_title="ClaimGuard AI", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    settings = load_ui_settings()
    return ApiClient(settings.api_url, settings.timeout_s)


# Only cheap read calls are cached. The analysis is NEVER cached: it is slow, costs money
# and writes an audit record, so it must only run when the reviewer clicks the button.
@st.cache_data(ttl=10, show_spinner=False)
def cached_claim_ids() -> list[str]:
    return get_client().list_claims()


@st.cache_data(ttl=30, show_spinner=False)
def cached_claim(claim_id: str):
    return get_client().get_claim(claim_id)


@st.cache_data(ttl=30, show_spinner=False)
def cached_image_names(claim_id: str) -> list[str]:
    return get_client().list_images(claim_id)


@st.cache_data(ttl=60, show_spinner=False)
def cached_image(claim_id: str, name: str) -> bytes:
    return get_client().get_image_bytes(claim_id, name)


# ----- "Try your own claim": same pattern, separate API routes -----
@st.cache_data(ttl=5, show_spinner=False)
def cached_user_claim_ids() -> list[str]:
    return get_client().list_user_claims()


@st.cache_data(ttl=30, show_spinner=False)
def cached_user_claim(claim_id: str):
    return get_client().get_user_claim(claim_id)


@st.cache_data(ttl=30, show_spinner=False)
def cached_user_image_names(claim_id: str) -> list[str]:
    return get_client().list_user_images(claim_id)


@st.cache_data(ttl=60, show_spinner=False)
def cached_user_image(claim_id: str, name: str) -> bytes:
    return get_client().get_user_image_bytes(claim_id, name)


@st.cache_data(ttl=60, show_spinner=False)
def cached_policies() -> list[str]:
    return get_client().list_policies()


NEW_CLAIM = "__new__"
AUTO_POLICY = "auto"   # the API's word for "search every policy document"
CLAIM_TYPES = ["hail", "wind", "water", "fire"]  # the perils the decision rules can reason about
COMPONENTS = list(get_args(Component))


def fail(exc: ApiError) -> None:
    """Show a friendly message and stop this run."""
    st.error(str(exc))
    st.stop()


def render_new_claim_form() -> None:
    """The "Try your own claim" form. On success it saves the claim, then the next run analyzes it."""
    st.subheader("Try your own claim")
    st.caption(
        "Upload photos of the damage and tell us when and where it happened. Your photos are kept apart from "
        "the demo claims, location data hidden in photos is removed, and you can delete the claim afterwards. "
        "The weather for that day and place is checked on the internet. You do not need to pick a policy: your "
        "claim is checked against all policy documents automatically, and the result shows which document, "
        "section and sentence it relied on."
    )

    with st.form("new_claim_form"):
        left, right = st.columns(2)
        claimed = left.selectbox("Claimed damage", CLAIM_TYPES)
        component = right.selectbox("Part of the home", COMPONENTS, format_func=lambda c: "not sure" if c == "unclear" else c.replace("_", " "))
        incident = left.date_input("Incident date", value=None, max_value=date.today())
        location = st.text_input("Where did it happen? (city and country, e.g. Pune, India)")
        description = st.text_area("What happened? (optional)")
        photos = st.file_uploader("Photos (JPEG or PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("Submit and analyze", type="primary")

    if not submitted:
        return

    problems = []
    if not photos:
        problems.append("Please add at least one photo.")
    if claimed in ("hail", "wind") and (incident is None or not location.strip()):
        problems.append(
            "For hail and wind claims, please give the incident date and the location so the weather can be checked."
        )
    if problems:
        st.error(" ".join(problems))
        return

    try:
        with st.spinner("Saving your photos..."):
            claim = get_client().create_user_claim(
                AUTO_POLICY,
                claimed,
                component,
                incident.isoformat() if incident else "",
                location.strip(),
                description.strip(),
                [(p.name, p.getvalue(), p.type or "image/jpeg") for p in photos],
            )
    except ApiError as exc:
        st.error(str(exc))
        return

    cached_user_claim_ids.clear()
    st.session_state["pending_own_choice"] = claim.claim_id   # select it on the next run
    st.session_state["auto_analyze"] = claim.claim_id          # and analyze it straight away
    st.rerun()


st.title("🛡️ ClaimGuard AI")
st.caption("Evidence-based claim review. The system advises; a person decides.")

flash = st.session_state.pop("flash", None)
if flash:
    st.success(flash)

# ---------------- sidebar: mode, pick a claim, run the analysis ----------------
MODE_DEMO, MODE_OWN = "Demo claims", "Try your own claim"
own_mode = st.sidebar.radio("Mode", [MODE_DEMO, MODE_OWN]) == MODE_OWN

if own_mode:
    try:
        my_ids = cached_user_claim_ids()
    except ApiError as exc:
        fail(exc)

    options = [NEW_CLAIM, *my_ids]
    if "pending_own_choice" in st.session_state:  # set by the form or the delete button before a rerun
        st.session_state["own_claim_choice"] = st.session_state.pop("pending_own_choice")
    if st.session_state.get("own_claim_choice") not in options:
        st.session_state["own_claim_choice"] = NEW_CLAIM

    claim_id = st.sidebar.selectbox(
        "My claims",
        options,
        key="own_claim_choice",
        format_func=lambda value: "➕ New claim" if value == NEW_CLAIM else value,
    )
    if claim_id == NEW_CLAIM:
        render_new_claim_form()
        st.stop()

    load_claim, load_image_names, load_image = cached_user_claim, cached_user_image_names, cached_user_image
    run_analysis = lambda cid: get_client().analyze_user_claim(cid)
    load_history = lambda cid: get_client().user_history(cid)
else:
    try:
        claim_ids = cached_claim_ids()
    except ApiError as exc:
        fail(exc)

    if not claim_ids:
        st.info("The API has no claims to show.")
        st.stop()

    claim_id = st.sidebar.selectbox("Claim", claim_ids)
    load_claim, load_image_names, load_image = cached_claim, cached_image_names, cached_image
    run_analysis = lambda cid: get_client().analyze(cid)
    load_history = lambda cid: get_client().history(cid)

select_key = f"selected_assessment_{claim_id}"
auto_analyze = st.session_state.pop("auto_analyze", None) == claim_id  # a claim that was just submitted

if st.sidebar.button("Run analysis", type="primary") or auto_analyze:
    with st.spinner("Analyzing the claim. This can take up to a minute..."):
        try:
            new_record = run_analysis(claim_id)
        except ApiError as exc:
            fail(exc)
    st.session_state[select_key] = new_record.id  # show the new result straight away

if own_mode:
    with st.sidebar.expander("Privacy"):
        st.caption(
            "Delete removes your photos and the form data from the server. "
            "The audit record of the assessment (decision and reasons, no photos) is kept."
        )
        if st.button("Delete my photos and claim"):
            try:
                get_client().delete_user_claim(claim_id)
            except ApiError as exc:
                fail(exc)
            for cached in (cached_user_claim_ids, cached_user_claim, cached_user_image_names, cached_user_image):
                cached.clear()
            st.session_state["pending_own_choice"] = NEW_CLAIM
            st.session_state["flash"] = "Your claim and photos were deleted."
            st.rerun()

# ---------------- claim card + photos ----------------
try:
    claim = load_claim(claim_id)
    image_names = load_image_names(claim_id)
except ApiError as exc:
    fail(exc)

st.subheader(f"Claim {claim.claim_id}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Claimed damage", claim.claimed_damage_type.replace("_", " "))
c2.metric("Part of the home", claim.component.replace("_", " "))
c3.metric("Policy", "Matched automatically" if claim.policy_id == AUTO_POLICY else claim.policy_id)
c4.metric("Incident date", str(claim.incident_date) if claim.incident_date else "Not given")
if claim.description:
    st.write(claim.description)
if claim.location:
    st.caption(f"📍 Location: {claim.location}")

st.markdown("**Photos**")
if not image_names:
    st.warning("This claim has no photos yet.")
else:
    columns = st.columns(3)
    for index, name in enumerate(image_names):
        try:
            columns[index % 3].image(load_image(claim_id, name), caption=name)
        except ApiError as exc:
            columns[index % 3].warning(f"Could not load {name}: {exc}")

# ---------------- pick which assessment to show ----------------
try:
    history = load_history(claim_id)  # newest first; a cheap local call
except ApiError as exc:
    fail(exc)

if not history:
    st.info("This claim has not been analyzed yet. Use **Run analysis** in the sidebar.")
    st.stop()

by_id = {record.id: record for record in history}
if st.session_state.get(select_key) not in by_id:
    st.session_state[select_key] = history[0].id

st.divider()
chosen_id = st.selectbox(
    "Assessment to view",
    list(by_id),
    key=select_key,
    format_func=lambda i: (
        f"#{i} · {format_time(by_id[i].created_at)} · {decision_label(by_id[i].assessment.decision)}"
    ),
)
record = by_id[chosen_id]
assessment = record.assessment

# ---------------- the result ----------------
st.header(decision_label(assessment.decision))
st.caption(
    f"Assessment #{record.id} · {format_time(record.created_at)} · "
    f"model: {assessment.model_version or 'unknown'} · {last_review_action(record)}"
)

st.markdown("**Why (the official record):**")
for reason in assessment.reasons:
    st.markdown(f"- {reason.replace('$', chr(92) + '$')}")  # policy quotes can contain $ amounts

if assessment.missing_evidence:
    st.warning("**Missing evidence:** " + ", ".join(assessment.missing_evidence))

if assessment.summary:
    st.info(assessment.summary)
    st.caption("AI-written summary. The reasons above are the official record.")

# ---------------- evidence: photos -> policy -> weather -> verdict, top to bottom ----------------
policy_name_fallback = claim.policy_id if claim.policy_id != AUTO_POLICY else ""


def safe(text: str) -> str:
    return text.replace("$", chr(92) + "$")  # Streamlit would read two $ signs as a formula


st.markdown("### Photo findings")
if assessment.visual:
    st.table(finding_rows(assessment.visual))
    st.caption(
        "\"Model confidence\" is how sure the vision model says it is. A model can be confidently wrong, "
        "so the decision rules never use it."
    )
else:
    st.write("No photos were analyzed.")

st.markdown("### What the decision is based on")

st.markdown("**Policy documents**")
basis = policy_basis(assessment.policy_citations)
if basis:
    st.caption("The clauses that grant or exclude cover for this damage, with the exact words of the document.")
    for clause in basis:
        st.markdown(f"**{citation_role(clause)}** · {citation_title(clause, policy_name_fallback)}")
        if clause.quote:
            st.markdown(f"> “{safe(clause.quote)}”")
else:
    st.write("No clause that grants or excludes cover for this damage was found.")

if assessment.policy_citations:
    with st.expander(f"All policy clauses that were checked ({len(assessment.policy_citations)})"):
        for clause in sort_citations(assessment.policy_citations):
            st.markdown(f"**{citation_role(clause)}** · {citation_title(clause, policy_name_fallback)}")
            st.markdown(safe(" ".join(clause.text.split())))

st.markdown("**Weather**")
st.write(describe_weather(assessment.weather))
if assessment.weather is not None and assessment.weather.details:
    st.caption(WEATHER_ATTRIBUTION)

st.markdown("### Final verdict")
for label, line in verdict_recap(assessment, policy_name_fallback):
    st.markdown(f"- **{label}:** {safe(line)}")
st.caption("A person makes the final call: use the reviewer decision below.")

# ---------------- reviewer panel ----------------
st.divider()
st.subheader("Reviewer decision")

with st.form(f"review_form_{record.id}"):
    action = st.radio("Your decision", list(REVIEW_ACTION_LABELS), format_func=REVIEW_ACTION_LABELS.get)
    note = st.text_area("Note (required for an override)")
    submitted = st.form_submit_button("Save review")

if submitted:
    if action == "override" and not note.strip():
        st.error("An override needs a note explaining why.")
    else:
        try:
            get_client().review(record.id, action, note)
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.session_state["flash"] = "Review saved."
            st.rerun()

if record.reviews:
    st.markdown("**Reviews on this assessment:**")
    for item in record.reviews:
        extra = f" · {item.note}" if item.note else ""
        st.markdown(f"- {format_time(item.created_at)} · {item.action.replace('_', ' ')}{extra}")

# ---------------- history ----------------
with st.expander(f"All assessments for this claim ({len(history)})"):
    st.table(history_rows(history))