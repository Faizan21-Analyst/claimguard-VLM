"""Thin HTTP client for the ClaimGuard API.

Every failure becomes a small exception whose message is safe to show to a reviewer.
Server error text is never copied into a message.
"""
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from claimguard.api.models import AssessmentRecord
from claimguard.schemas.claim import Claim


class ApiError(Exception):
    """Any API failure. `str(error)` is always safe to display."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ApiUnreachable(ApiError):
    """Connection refused, DNS failure or timeout."""


class ClaimNotFound(ApiError):
    """The API answered 404."""


class ServiceUnavailable(ApiError):
    """The API answered 502 or 503: the assessment service or the image model behind it is down."""


class InvalidInput(ApiError):
    """The submitted claim form was refused. The message was written by our API to be shown to the user."""


def _segment(value: str) -> str:
    """Percent-encode one URL path segment (so spaces and slashes cannot change the route)."""
    return quote(value, safe="")


class ApiClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    # ---------- internals ----------

    def _request(self, method: str, url: str, show_detail: bool = False, **kwargs) -> httpx.Response:
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiUnreachable(
                "The request timed out. An analysis may still have finished: "
                "check the assessment history in a moment."
            ) from exc
        except httpx.TransportError as exc:
            raise ApiUnreachable(
                f"Cannot reach the ClaimGuard API at {self.base_url}. "
                "Start it with: uvicorn claimguard.api.main:app"
            ) from exc

        status = response.status_code
        if show_detail and status in (413, 422):
            # Only for the claim form: our API's own validation texts are meant for the user.
            # Anything that is not a plain string (framework error lists) stays generic.
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            raise InvalidInput(detail if isinstance(detail, str) else "The submitted claim was not valid.", status)
        if status == 404:
            raise ClaimNotFound("That item was not found.", status)
        if status == 502:
            raise ServiceUnavailable(
                "The assessment service is temporarily unavailable. Please try again.", status
            )
        if status == 503:
            raise ServiceUnavailable(
                "The image-analysis model is not available right now (its free GPU time may be used up "
                "for today). Please try again later.",
                status,
            )
        if status >= 400:
            raise ApiError(f"The API returned an error (status {status}).", status)
        return response

    @staticmethod
    def _json(response: httpx.Response):
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError("The API returned an unexpected response.") from exc

    @staticmethod
    def _parse(model, data):
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise ApiError("The API returned an unexpected response.") from exc

    # ---------- public methods ----------

    def list_claims(self) -> list[str]:
        data = self._json(self._request("GET", "/claims"))
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ApiError("The API returned an unexpected response.")
        return data

    def get_claim(self, claim_id: str) -> Claim:
        data = self._json(self._request("GET", f"/claims/{_segment(claim_id)}"))
        return self._parse(Claim, data)

    def list_images(self, claim_id: str) -> list[str]:
        data = self._json(self._request("GET", f"/claims/{_segment(claim_id)}/images"))
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ApiError("The API returned an unexpected response.")
        return data

    def get_image_bytes(self, claim_id: str, filename: str) -> bytes:
        response = self._request(
            "GET", f"/claims/{_segment(claim_id)}/images/{_segment(filename)}"
        )
        return response.content

    def analyze(self, claim_id: str) -> AssessmentRecord:
        data = self._json(self._request("POST", f"/claims/{_segment(claim_id)}/analyze"))
        return self._parse(AssessmentRecord, data)

    # ---------- "Try your own claim" ----------

    @staticmethod
    def _string_list(data) -> list[str]:
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ApiError("The API returned an unexpected response.")
        return data

    def list_policies(self) -> list[str]:
        return self._string_list(self._json(self._request("GET", "/policies")))

    def create_user_claim(
        self,
        policy_id: str,
        claimed_damage_type: str,
        component: str,
        incident_date: str,
        location: str,
        description: str,
        photos: list[tuple[str, bytes, str]],
    ) -> Claim:
        """photos: (file name, bytes, mime type). incident_date: 'YYYY-MM-DD' or ''."""
        response = self._request(
            "POST",
            "/user-claims",
            show_detail=True,
            data={
                "policy_id": policy_id,
                "claimed_damage_type": claimed_damage_type,
                "component": component,
                "incident_date": incident_date,
                "location": location,
                "description": description,
            },
            files=[("photos", (name, content, mime)) for name, content, mime in photos],
        )
        return self._parse(Claim, self._json(response))

    def list_user_claims(self) -> list[str]:
        return self._string_list(self._json(self._request("GET", "/user-claims")))

    def get_user_claim(self, claim_id: str) -> Claim:
        return self._parse(Claim, self._json(self._request("GET", f"/user-claims/{_segment(claim_id)}")))

    def list_user_images(self, claim_id: str) -> list[str]:
        return self._string_list(self._json(self._request("GET", f"/user-claims/{_segment(claim_id)}/images")))

    def get_user_image_bytes(self, claim_id: str, filename: str) -> bytes:
        return self._request(
            "GET", f"/user-claims/{_segment(claim_id)}/images/{_segment(filename)}"
        ).content

    def analyze_user_claim(self, claim_id: str) -> AssessmentRecord:
        data = self._json(self._request("POST", f"/user-claims/{_segment(claim_id)}/analyze"))
        return self._parse(AssessmentRecord, data)

    def user_history(self, claim_id: str) -> list[AssessmentRecord]:
        data = self._json(self._request("GET", f"/user-claims/{_segment(claim_id)}/assessments"))
        if not isinstance(data, list):
            raise ApiError("The API returned an unexpected response.")
        return [self._parse(AssessmentRecord, item) for item in data]

    def delete_user_claim(self, claim_id: str) -> None:
        self._request("DELETE", f"/user-claims/{_segment(claim_id)}")

    # ---------- demo claims (continued) ----------

    def history(self, claim_id: str) -> list[AssessmentRecord]:
        data = self._json(self._request("GET", f"/claims/{_segment(claim_id)}/assessments"))
        if not isinstance(data, list):
            raise ApiError("The API returned an unexpected response.")
        return [self._parse(AssessmentRecord, item) for item in data]

    def review(self, assessment_id: int, action: str, note: str = "") -> AssessmentRecord:
        data = self._json(
            self._request(
                "POST",
                f"/assessments/{int(assessment_id)}/review",
                json={"action": action, "note": note},
            )
        )
        return self._parse(AssessmentRecord, data)