
import json
from pathlib import Path

import pymupdf

from claimguard.schemas.claim import Claim


# Project root: the parent directory of scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICIES_DIR = PROJECT_ROOT / "data" / "synthetic" / "policies"
CLAIMS_DIR = PROJECT_ROOT / "data" / "synthetic" / "claims"
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"


POLICIES = {
    "policy_A.pdf": {
        "title": "Synthetic Home Insurance Policy A",
        "clauses": [
            (
                "1.1 Policy scope",
                "This policy provides coverage for eligible sudden "
                "physical damage to the insured residential property, "
                "subject to the terms and exclusions below.",
            ),
            (
                "4.2 Hail and windstorm",
                "Direct physical damage caused by hail or windstorms "
                "to roof shingles, gutters, and other insured exterior "
                "components is covered, subject to verification.",
            ),
            (
                "4.3 Roof damage evidence",
                "The claimant must provide reasonable evidence of "
                "the reported damage. Weather records may support "
                "the assessment but do not independently prove damage.",
            ),
            (
                "6.1 Wear and tear exclusion",
                "Gradual deterioration, aging, corrosion, and normal "
                "wear and tear are excluded. Pre-existing deterioration "
                "is not covered as sudden storm damage.",
            ),
            (
                "7.1 Claim investigation",
                "Claims may require photographs, inspection findings, "
                "and relevant incident information before a decision.",
            ),
        ],
    },
    "policy_B.pdf": {
        "title": "Synthetic Home Insurance Policy B",
        "clauses": [
            (
                "1.1 Policy scope",
                "This policy covers eligible sudden physical damage "
                "to insured property, subject to the specific "
                "coverage terms and exclusions below.",
            ),
            (
                "3.1 Sudden water damage",
                "Sudden and accidental water damage to insured "
                "interior property, including ceilings, is covered "
                "when the event is not otherwise excluded.",
            ),
            (
                "4.1 Hail damage to roofs",
                "Direct damage to roof shingles caused by hail is "
                "excluded under this policy, even when a hailstorm "
                "is confirmed by external weather records.",
            ),
            (
                "4.2 Other storm damage",
                "Damage not specifically covered by this policy "
                "must be reviewed against the applicable exclusions "
                "and coverage terms.",
            ),
            (
                "7.1 Claim investigation",
                "Photographs, incident information, and relevant "
                "supporting evidence may be requested during review.",
            ),
        ],
    },
    "policy_C.pdf": {
        "title": "Synthetic Home Insurance Policy C",
        "clauses": [
            (
                "1.1 Policy scope",
                "This policy provides the specific fire coverage "
                "described in this document. Other causes of loss "
                "are not granted coverage by this clause.",
            ),
            (
                "3.1 Fire damage",
                "Direct physical damage caused by an accidental "
                "fire to insured property is covered, subject "
                "to the terms and exclusions of this policy.",
            ),
            (
                "3.2 Fire investigation",
                "Fire-related claims require evidence of the "
                "incident and the resulting physical damage.",
            ),
            (
                "7.1 Claim investigation",
                "The insurer may request photographs, inspection "
                "findings, and supporting incident documentation.",
            ),
        ],
    },
}


def create_policy_pdf(
    filename: str,
    title: str,
    clauses: list[tuple[str, str]],
) -> None:
    """Create a searchable PDF with numbered clauses on separate paragraphs."""
    output_path = POLICIES_DIR / filename

    document = pymupdf.open()
    page = document.new_page()

    page_width = page.rect.width
    margin = 55
    y = 50

    page.insert_text(
        (margin, y),
        title,
        fontsize=16,
        fontname="helv",
    )
    y += 35

    for clause_id, clause_text in clauses:
        # Keep the clause number at the beginning of the paragraph.
        paragraph = f"{clause_id} {clause_text}"

        # Start a new page if the next paragraph would exceed the page.
        if y + 75 > page.rect.height - 45:
            page = document.new_page()
            y = 50

        rect = pymupdf.Rect(
            margin,
            y,
            page_width - margin,
            min(y + 70, page.rect.height - 35),
        )

        unused_height = page.insert_textbox(
            rect,
            paragraph,
            fontsize=10.5,
            fontname="helv",
            lineheight=1.3,
        )

        # Defensive fallback if a paragraph does not fit.
        if unused_height < 0:
            page = document.new_page()
            y = 50
            rect = pymupdf.Rect(
                margin,
                y,
                page_width - margin,
                page.rect.height - 40,
            )
            page.insert_textbox(
                rect,
                paragraph,
                fontsize=10.5,
                fontname="helv",
                lineheight=1.3,
            )
            y = 130
        else:
            y += 78

    document.save(output_path)
    document.close()


CLAIMS = [
    {
        "claim": Claim(
            claim_id="claim_001",
            policy_id="policy_A",
            claimed_damage_type="hail",
            component="roof_shingles",
            incident_date="2026-06-10",
            description="Hailstorm reportedly damaged roof shingles.",
        ),
        "weather": [
            {
                "event": "hailstorm",
                "date": "2026-06-10",
                "location": "Demo City",
                "source": "synthetic",
            }
        ],
        "expected_decision": "supported",
    },
    {
        "claim": Claim(
            claim_id="claim_002",
            policy_id="policy_A",
            claimed_damage_type="hail",
            component="roof_shingles",
            incident_date="2026-06-12",
            description="Roof shingle damage reported after suspected hail.",
        ),
        "weather": [],
        "expected_decision": "needs_review",
    },
    {
        "claim": Claim(
            claim_id="claim_003",
            policy_id="policy_B",
            claimed_damage_type="hail",
            component="roof_shingles",
            incident_date="2026-06-15",
            description="Hail damage reportedly affected roof shingles.",
        ),
        "weather": [
            {
                "event": "hailstorm",
                "date": "2026-06-15",
                "location": "Demo City",
                "source": "synthetic",
            }
        ],
        "expected_decision": "possible_exclusion",
    },
    {
        "claim": Claim(
            claim_id="claim_004",
            policy_id="policy_C",
            claimed_damage_type="hail",
            component="roof_shingles",
            incident_date="2026-06-18",
            description="Hail damage reported; policy contains fire coverage only.",
        ),
        "weather": [
            {
                "event": "hailstorm",
                "date": "2026-06-18",
                "location": "Demo City",
                "source": "synthetic",
            }
        ],
        "expected_decision": "insufficient_evidence",
    },
    {
        "claim": Claim(
            claim_id="claim_005",
            policy_id="policy_B",
            claimed_damage_type="water",
            component="ceiling",
            incident_date="2026-06-20",
            description="A sudden water leak reportedly damaged the ceiling.",
        ),
        "weather": [
            {
                "event": "heavy_rain",
                "date": "2026-06-20",
                "location": "Demo City",
                "source": "synthetic",
            }
        ],
        "expected_decision": "supported",
    },
    {
        "claim": Claim(
            claim_id="claim_006",
            policy_id="policy_A",
            claimed_damage_type="wind",
            component="roof_shingles",
            incident_date="2026-06-22",
            description="Strong winds reportedly damaged roof shingles.",
        ),
        "weather": [
            {
                "event": "windstorm",
                "date": "2026-06-22",
                "location": "Demo City",
                "source": "synthetic",
            }
        ],
        "expected_decision": "supported",
    },
]


def main() -> None:
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all three searchable policy PDFs.
    for filename, policy in POLICIES.items():
        create_policy_pdf(
            filename=filename,
            title=policy["title"],
            clauses=policy["clauses"],
        )
        print(f"Created policy: {POLICIES_DIR / filename}")

    expected_decisions = {}

    # Generate schema-validated claims and synthetic weather records.
    for item in CLAIMS:
        claim = item["claim"]
        claim_dir = CLAIMS_DIR / claim.claim_id

        images_dir = claim_dir / "images"
        external_dir = claim_dir / "external"

        images_dir.mkdir(parents=True, exist_ok=True)
        external_dir.mkdir(parents=True, exist_ok=True)

        claim_path = claim_dir / "claim.json"
        claim_path.write_text(
            claim.model_dump_json(indent=2),
            encoding="utf-8",
        )

        weather_path = external_dir / "weather.json"
        weather_path.write_text(
            json.dumps(item["weather"], indent=2),
            encoding="utf-8",
        )

        expected_decisions[claim.claim_id] = item["expected_decision"]

        print(f"Created claim: {claim_path}")
        print(f"Created weather: {weather_path}")

    manifest_path = DATA_DIR / "expected_decisions.json"
    manifest_path.write_text(
        json.dumps(expected_decisions, indent=2),
        encoding="utf-8",
    )

    print(f"\nCreated expected decisions: {manifest_path}")
    print("Demo data generation complete.")


if __name__ == "__main__":
    main()
