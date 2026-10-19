"""Live weather check for user-submitted claims, using Open-Meteo (free, no API key).

Same job as `weather_check_tool` (which reads a mock JSON file for the demo claims), but the
evidence comes from the internet: place name -> coordinates -> that day's weather.

Honest limits (shown to the reviewer, never hidden):
- The archive is reanalysis data on a 9-25 km grid, not a weather-station report. It can miss a
  small, local storm, so "no match" means "not confirmed", never "it did not happen".
- Open-Meteo has no hail sensor. Hail counts as confirmed only when the day's weather code is a
  thunderstorm-with-hail code (96/99 by default). Most hail claims will therefore go to a human.
- Any failure (no internet, unknown place, no data) becomes found=False with a reason, so the
  claim goes to human review instead of crashing or being guessed.

Weather data by Open-Meteo.com (CC BY 4.0, free for non-commercial use).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx

from claimguard.agent.perils import WEATHER_PERILS
from claimguard.schemas.claim import Claim
from claimguard.schemas.evidence import WeatherEvidence

logger = logging.getLogger(__name__)

SOURCE = "open-meteo"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARIABLES = "weather_code,wind_gusts_10m_max,precipitation_sum"
ARCHIVE_DELAY_DAYS = 3  # the archive ends about 2 days ago; newer days come from the forecast API

# Only the codes people need to read in a reviewer's note.
WMO_TEXT = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
    51: "drizzle", 53: "drizzle", 55: "drizzle", 61: "rain", 63: "rain", 65: "heavy rain",
    71: "snow", 73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers",
    82: "violent rain showers", 95: "thunderstorm", 96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


class WeatherServiceError(Exception):
    """The weather service could not be reached or answered with something unusable."""


@dataclass(frozen=True)
class Place:
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class DayWeather:
    gust_kmh: float | None
    rain_mm: float | None
    code: int | None


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class OpenMeteoClient:
    """Two small calls (geocoding, daily weather) with an in-memory cache."""

    def __init__(self, http: httpx.Client | None = None, timeout: float = 10.0):
        self._http = http or httpx.Client(timeout=timeout)
        self._places: dict[str, Place | None] = {}
        self._days: dict[tuple[float, float, date], DayWeather | None] = {}

    def _get(self, url: str, params: dict) -> dict:
        try:
            response = self._http.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WeatherServiceError(f"weather request failed: {exc.__class__.__name__}") from exc
        if not isinstance(data, dict):
            raise WeatherServiceError("weather service returned an unexpected response")
        return data

    def geocode(self, text: str) -> Place | None:
        """'Mumbai, India' -> a Place. Tries the full text, then the first part with the rest as a hint."""
        text = " ".join(text.split())
        key = text.lower()
        if key in self._places:
            return self._places[key]

        parts = [p.strip() for p in text.split(",") if p.strip()]
        queries = [text] + ([parts[0]] if len(parts) > 1 else [])
        hint = {t for t in re.split(r"[\s,]+", " ".join(parts[1:]).lower()) if len(t) > 1}

        place = None
        for query in queries:
            params = {"name": query, "count": 10, "language": "en", "format": "json"}
            results = self._get(GEOCODE_URL, params).get("results") or []
            if not results:
                continue
            chosen = results[0]
            if hint and query != text:
                for r in results:
                    where = " ".join(str(r.get(k, "")) for k in ("admin1", "admin2", "country", "country_code")).lower()
                    if any(token in where for token in hint):
                        chosen = r
                        break
            label = ", ".join(str(chosen[k]) for k in ("name", "admin1", "country") if chosen.get(k))
            place = Place(label, float(chosen["latitude"]), float(chosen["longitude"]))
            break

        self._places[key] = place
        return place

    def day(self, place: Place, day: date, today: date | None = None) -> DayWeather | None:
        """Weather summary for one calendar day (local time), or None if the service has no data."""
        key = (round(place.latitude, 3), round(place.longitude, 3), day)
        if key in self._days:
            return self._days[key]

        today = today or date.today()
        url = ARCHIVE_URL if day <= today - timedelta(days=ARCHIVE_DELAY_DAYS) else FORECAST_URL
        data = self._get(url, {
            "latitude": place.latitude, "longitude": place.longitude,
            "start_date": day.isoformat(), "end_date": day.isoformat(),
            "daily": DAILY_VARIABLES, "timezone": "auto", "wind_speed_unit": "kmh",
        })

        daily = data.get("daily")
        weather = None
        if isinstance(daily, dict) and isinstance(daily.get("time"), list) and day.isoformat() in daily["time"]:
            i = daily["time"].index(day.isoformat())

            def at(name):
                values = daily.get(name)
                return values[i] if isinstance(values, list) and i < len(values) else None

            code = _number(at("weather_code"))
            weather = DayWeather(_number(at("wind_gusts_10m_max")), _number(at("precipitation_sum")),
                                 int(code) if code is not None else None)
            if weather == DayWeather(None, None, None):
                weather = None

        self._days[key] = weather
        return weather


def _describe(place: Place, day: date, w: DayWeather, peril: str, wind_gust_kmh: float,
              hail_codes: tuple[int, ...]) -> str:
    gust = "n/a" if w.gust_kmh is None else f"{w.gust_kmh:.0f} km/h"
    rain = "n/a" if w.rain_mm is None else f"{w.rain_mm:.1f} mm"
    sky = "n/a" if w.code is None else f"{WMO_TEXT.get(w.code, 'code ' + str(w.code))} (code {w.code})"
    rule = (f"wind event needs a gust of at least {wind_gust_kmh:.0f} km/h" if peril == "wind"
            else f"hail needs weather code {'/'.join(map(str, hail_codes))}")
    return f"{place.name} on {day.isoformat()}: max gust {gust}, rain {rain}, weather {sky}. ({rule})"


def make_live_weather_tool(
    client: OpenMeteoClient | None = None,
    wind_gust_kmh: float = 60.0,
    hail_codes: tuple[int, ...] = (96, 99),
    timeout: float = 10.0,
    today: date | None = None,
) -> Callable[[Claim, Path | str | None], WeatherEvidence | None]:
    """Build a drop-in replacement for weather_check_tool: (claim, claim_dir) -> WeatherEvidence | None."""
    client = client or OpenMeteoClient(timeout=timeout)

    def live_weather_tool(claim: Claim, claim_dir: Path | str | None = None) -> WeatherEvidence | None:
        peril = claim.claimed_damage_type
        if peril not in WEATHER_PERILS or claim.incident_date is None:
            return None  # a weather check does not apply (same rule as the demo tool)

        iso = claim.incident_date.isoformat()
        if not (claim.location or "").strip():
            return WeatherEvidence(found=False, date=iso, source="no_location")
        if claim.incident_date > (today or date.today()):
            return WeatherEvidence(found=False, date=iso, location=claim.location, source="future_date")

        try:
            place = client.geocode(claim.location)
            if place is None:
                return WeatherEvidence(found=False, date=iso, location=claim.location, source="location_not_found")
            weather = client.day(place, claim.incident_date, today)
        except WeatherServiceError:
            logger.warning("Live weather lookup failed for claim %s", claim.claim_id, exc_info=True)
            return WeatherEvidence(found=False, date=iso, location=claim.location, source="weather_service_error")

        if weather is None:
            return WeatherEvidence(found=False, date=iso, location=place.name, source="no_data")

        if peril == "wind":
            confirmed = weather.gust_kmh is not None and weather.gust_kmh >= wind_gust_kmh
        else:  # hail
            confirmed = weather.code in hail_codes

        return WeatherEvidence(
            event=peril if confirmed else None,
            date=iso,
            location=place.name,
            found=confirmed,
            source=SOURCE,
            details=_describe(place, claim.incident_date, weather, peril, wind_gust_kmh, hail_codes),
        )

    return live_weather_tool
