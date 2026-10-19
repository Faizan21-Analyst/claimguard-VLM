"""Live weather check. Open-Meteo is replaced by a fake network, so these tests never go online."""
from datetime import date, timedelta

import httpx
import pytest

from claimguard.agent.weather_live import (
    ARCHIVE_URL,
    FORECAST_URL,
    OpenMeteoClient,
    make_live_weather_tool,
)
from claimguard.schemas.claim import Claim

GEO_MUMBAI = {"results": [{"name": "Mumbai", "latitude": 19.07, "longitude": 72.88,
                           "admin1": "Maharashtra", "country": "India", "country_code": "IN"}]}


def daily(day, gust=None, rain=None, code=None):
    return {"daily": {"time": [day.isoformat()], "wind_gusts_10m_max": [gust],
                      "precipitation_sum": [rain], "weather_code": [code]}}


class FakeInternet:
    """Routes geocoding and weather requests and remembers every URL that was called."""

    def __init__(self, geo=None, weather=None, status=200, geo_by_name=None):
        self.geo = GEO_MUMBAI if geo is None else geo
        self.geo_by_name = geo_by_name
        self.weather = weather
        self.status = status
        self.urls: list[str] = []

    def client(self) -> OpenMeteoClient:
        return OpenMeteoClient(http=httpx.Client(transport=httpx.MockTransport(self._handle)))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        if self.status != 200:
            return httpx.Response(self.status)
        if request.url.host == "geocoding-api.open-meteo.com":
            if self.geo_by_name is not None:
                return httpx.Response(200, json=self.geo_by_name.get(request.url.params["name"], {}))
            return httpx.Response(200, json=self.geo)
        return httpx.Response(200, json=self.weather)


TODAY = date(2026, 10, 19)
DAY = date(2026, 6, 10)


def claim(peril="wind", when=DAY, location="Mumbai, India") -> Claim:
    return Claim(claim_id="c1", policy_id="policy_A", claimed_damage_type=peril,
                 component="roof_shingles", incident_date=when, location=location)


def tool_for(net: FakeInternet, **kwargs):
    return make_live_weather_tool(client=net.client(), today=TODAY, **kwargs)


def test_strong_gust_confirms_a_wind_claim():
    result = tool_for(FakeInternet(weather=daily(DAY, gust=74.0, rain=12.3, code=95)))(claim("wind"), None)

    assert result.found is True
    assert result.event == "wind"
    assert result.source == "open-meteo"
    assert result.location == "Mumbai, Maharashtra, India"
    assert "74 km/h" in result.details and "12.3 mm" in result.details


def test_weak_wind_does_not_confirm_and_explains_the_threshold():
    result = tool_for(FakeInternet(weather=daily(DAY, gust=35.0, code=3)))(claim("wind"), None)

    assert result.found is False
    assert result.event is None
    assert "at least 60 km/h" in result.details


def test_wind_threshold_is_configurable():
    net = FakeInternet(weather=daily(DAY, gust=45.0))
    assert tool_for(net, wind_gust_kmh=40.0)(claim("wind"), None).found is True


@pytest.mark.parametrize("code, expected", [(96, True), (99, True), (95, False), (3, False), (None, False)])
def test_hail_is_confirmed_only_by_hail_weather_codes(code, expected):
    result = tool_for(FakeInternet(weather=daily(DAY, gust=80.0, code=code)))(claim("hail"), None)

    assert result.found is expected
    assert result.event == ("hail" if expected else None)


def test_a_strong_gust_does_not_confirm_a_hail_claim():
    result = tool_for(FakeInternet(weather=daily(DAY, gust=120.0, code=3)))(claim("hail"), None)
    assert result.found is False


@pytest.mark.parametrize("peril", ["water", "fire"])
def test_weather_does_not_apply_to_water_or_fire(peril):
    net = FakeInternet(weather=daily(DAY, gust=100.0))
    assert tool_for(net)(claim(peril), None) is None
    assert net.urls == []  # no pointless network call


def test_no_incident_date_means_not_applicable():
    assert tool_for(FakeInternet())(claim("wind", when=None), None) is None


def test_missing_location_is_reported_not_guessed():
    result = tool_for(FakeInternet())(claim("wind", location="  "), None)
    assert (result.found, result.source) == (False, "no_location")


def test_future_incident_date_is_refused():
    net = FakeInternet()
    result = tool_for(net)(claim("wind", when=TODAY + timedelta(days=3)), None)

    assert (result.found, result.source) == (False, "future_date")
    assert net.urls == []


def test_unknown_place_is_reported():
    result = tool_for(FakeInternet(geo={}))(claim("wind", location="Nowhereville"), None)
    assert (result.found, result.source) == (False, "location_not_found")


@pytest.mark.parametrize("status", [429, 500, 503])
def test_service_errors_never_crash_the_claim(status):
    result = tool_for(FakeInternet(status=status))(claim("wind"), None)
    assert (result.found, result.source) == (False, "weather_service_error")


def test_day_without_data_is_reported():
    result = tool_for(FakeInternet(weather=daily(DAY)))(claim("wind"), None)   # every value is null
    assert (result.found, result.source) == (False, "no_data")


def test_city_and_country_text_falls_back_to_the_city_name():
    net = FakeInternet(
        weather=daily(DAY, gust=80.0),
        geo_by_name={
            "Pune, India": {},   # the geocoder does not understand "city, country" as one name
            "Pune": {"results": [
                {"name": "Pune", "latitude": 18.5, "longitude": 73.8, "admin1": "Maharashtra", "country": "India"},
            ]},
        },
    )
    result = tool_for(net)(claim("wind", location="Pune, India"), None)

    assert result.found is True and result.location == "Pune, Maharashtra, India"


def test_country_hint_picks_the_right_city():
    net = FakeInternet(
        weather=daily(DAY, gust=80.0),
        geo_by_name={
            "Springfield, USA": {},
            "Springfield": {"results": [
                {"name": "Springfield", "latitude": 1.0, "longitude": 1.0, "admin1": "Somewhere", "country": "Canada"},
                {"name": "Springfield", "latitude": 39.8, "longitude": -89.6, "admin1": "Illinois",
                 "country": "United States", "country_code": "US"},
            ]},
        },
    )
    result = tool_for(net)(claim("wind", location="Springfield, United States"), None)

    assert result.location == "Springfield, Illinois, United States"


def test_old_dates_use_the_archive_and_recent_dates_use_the_forecast_api():
    old = FakeInternet(weather=daily(DAY, gust=80.0))
    tool_for(old)(claim("wind", when=DAY), None)
    assert any(url.startswith(ARCHIVE_URL) for url in old.urls)

    recent_day = TODAY - timedelta(days=1)
    recent = FakeInternet(weather=daily(recent_day, gust=80.0))
    tool_for(recent)(claim("wind", when=recent_day), None)
    assert any(url.startswith(FORECAST_URL) for url in recent.urls)


def test_the_same_lookup_is_only_fetched_once():
    net = FakeInternet(weather=daily(DAY, gust=80.0))
    tool = tool_for(net)

    tool(claim("wind"), None)
    calls_after_first = len(net.urls)
    tool(claim("wind"), None)

    assert len(net.urls) == calls_after_first
