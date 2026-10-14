import json

from claimguard.agent.perils import WEATHER_PERILS
from claimguard.agent.tools import (
    EVENT_NAMES,
    build_policy_query,
    weather_check_tool,
)
from claimguard.schemas.claim import Claim


def make_claim(peril="hail", incident_date="2026-06-10", component="roof_shingles"):
    return Claim(
        claim_id="claim_test",
        policy_id="policy_A",
        claimed_damage_type=peril,
        component=component,
        incident_date=incident_date,
    )


def write_weather(claim_dir, records):
    external = claim_dir / "external"
    external.mkdir(parents=True, exist_ok=True)
    (external / "weather.json").write_text(json.dumps(records), encoding="utf-8")


def test_build_policy_query_normal_case():
    assert build_policy_query(make_claim()) == "hail damage to roof shingles"


def test_build_policy_query_unclear_component():
    assert build_policy_query(make_claim(component="unclear")) == "hail damage"


def test_every_weather_peril_has_event_names():
    # Guards against adding a peril to one list and forgetting the other.
    assert set(EVENT_NAMES) == set(WEATHER_PERILS)


def test_matching_hail_record_is_found(tmp_path):
    write_weather(
        tmp_path,
        [{"event": "hailstorm", "date": "2026-06-10",
          "location": "Demo City", "source": "synthetic"}],
    )

    result = weather_check_tool(make_claim(), tmp_path)

    assert result is not None
    assert result.found is True
    assert result.event == "hailstorm"
    assert result.date == "2026-06-10"


def test_matching_wind_record_is_found(tmp_path):
    write_weather(tmp_path, [{"event": "windstorm", "date": "2026-06-22"}])

    result = weather_check_tool(make_claim("wind", "2026-06-22"), tmp_path)

    assert result is not None and result.found is True


def test_different_date_is_not_found(tmp_path):
    write_weather(tmp_path, [{"event": "hailstorm", "date": "2026-06-11"}])

    result = weather_check_tool(make_claim(), tmp_path)

    assert result is not None
    assert result.found is False
    assert result.source == "no_match"


def test_missing_weather_file(tmp_path):
    result = weather_check_tool(make_claim(), tmp_path)

    assert result is not None
    assert result.found is False
    assert result.source == "missing"


def test_invalid_json_is_not_found(tmp_path):
    (tmp_path / "external").mkdir()
    (tmp_path / "external" / "weather.json").write_text("{not json", encoding="utf-8")

    result = weather_check_tool(make_claim(), tmp_path)

    assert result is not None
    assert result.found is False
    assert result.source == "invalid"


def test_water_claim_skips_weather(tmp_path):
    assert weather_check_tool(make_claim(peril="water"), tmp_path) is None


def test_claim_without_incident_date_skips_weather(tmp_path):
    assert weather_check_tool(make_claim(incident_date=None), tmp_path) is None


def test_heavy_rain_does_not_match_hail(tmp_path):
    write_weather(tmp_path, [{"event": "heavy_rain", "date": "2026-06-10"}])

    result = weather_check_tool(make_claim(), tmp_path)

    assert result is not None
    assert result.found is False