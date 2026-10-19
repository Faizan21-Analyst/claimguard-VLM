"""Builds the policy library used by "Try your own claim".

Four documents that together behave like one real insurer's policy pack:

  * a base homeowners policy (definitions, property, perils, exclusions, conditions)
  * a storm and roof endorsement   (hail and wind terms)
  * a water damage endorsement     (escape of water, flood, leakage)
  * a claims guide                 (what to send, how a claim is reviewed)

A person who submits a claim does not know which document answers their question, so the system searches
all of them and cites the document, the section number, the heading and the exact sentence.

The wording follows what real home policies say. One constraint is deliberate: the decision rules in
agent/fusion.py are keyword based, so a clause that names hail, wind, water or fire either grants cover or
excludes it, never both. Exclusions that are about something else (wear and tear, flood, neglect...) are
worded without those four words so they cannot be mistaken for an exclusion of the claimed peril.
tests/test_policy_library.py checks this, so an edit that breaks it fails a test instead of a claim.

Plain ASCII on purpose: the PDF text then matches the quotes shown in the app character for character.
"""
from pathlib import Path

import pymupdf

# ---------------------------------------------------------------------------------------------------------
# Page layout. Header and footer live outside the band that ingest reads (rag/ingest.py: PAGE_*_MARGIN).
# ---------------------------------------------------------------------------------------------------------
PAGE_W, PAGE_H = 595, 842
MARGIN_X = 60
TEXT_W = PAGE_W - 2 * MARGIN_X
BODY_TOP = 84          # first body line on a page
BODY_BOTTOM = 782      # a clause never runs below this (ingest reads down to y = 792)
LINE = 13.5
PARA_GAP = 5
INDENT = 16

REGULAR, BOLD, ITALIC = "helv", "hebo", "heit"
GREY = (0.35, 0.35, 0.35)
RULE = (0.75, 0.75, 0.75)


def C(number: str, heading: str, *paragraphs: str) -> tuple[str, str, tuple[str, ...]]:
    """One numbered clause: ("3.4", "Hail", (paragraph, ...))."""
    return number, heading, paragraphs


DOCUMENTS: list[dict] = [
    {
        "file": "homeshield_homeowners_hs100.pdf",
        "title": "HomeShield Homeowners Policy (Form HS-100)",
        "form": "HS-100",
        "edition": "Edition 03/2026",
        "notice": (
            "Please read this policy carefully. It describes what we insure, what we do not insure and what we "
            "need from you when you make a claim. Words in quotation marks have the meaning given in Section 1."
        ),
        "sections": [
            ("SECTION 1  DEFINITIONS AND GENERAL TERMS", [
                C("1.1", "About this policy",
                  'This policy is an agreement between you and HomeShield Insurance Company ("we", "us", "our"). '
                  "It applies to the dwelling named in your Declarations. The policy consists of this form, your "
                  "Declarations page and any endorsements that are listed there."),
                C("1.2", "Definitions",
                  '"You" and "your" mean the person named in the Declarations and a spouse or partner who lives '
                  'with that person. "Dwelling" means the house at the address shown in the Declarations, '
                  "including attached garages and permanently attached fixtures.",
                  '"Sudden" means happening abruptly and without warning. "Accidental" means not expected or '
                  'intended by you. "Occurrence" means one event, or a series of events that come from the same '
                  "cause."),
                C("1.3", "Policy period",
                  "This policy applies to loss that starts during the policy period shown in the Declarations. "
                  "A loss that began before the policy period started is not insured under this policy."),
            ]),
            ("SECTION 2  PROPERTY WE INSURE", [
                C("2.1", "Dwelling",
                  "We cover the dwelling, including walls, floors, ceilings, the roof structure and roof "
                  "coverings, and fixtures that are permanently attached to the building."),
                C("2.2", "Other structures",
                  "We cover other structures on the residence premises that are separated from the dwelling, such "
                  "as detached garages, fences and garden sheds, up to 10% of the dwelling limit shown in the "
                  "Declarations."),
                C("2.3", "Personal property",
                  "We cover personal property that you own and keep in the dwelling, up to the contents limit "
                  "shown in the Declarations. Jewelry, watches and furs are limited to $1,500 in total unless "
                  "they are scheduled."),
                C("2.4", "Additional living expenses",
                  "If a loss that we insure makes the dwelling unfit to live in, we cover the reasonable increase "
                  "in your living costs while it is repaired, for up to 12 months."),
            ]),
            ("SECTION 3  PERILS WE INSURE AGAINST", [
                C("3.1", "What this section covers",
                  "We cover sudden and accidental direct physical loss to the property described in Section 2 "
                  "when it is caused by one of the perils named in clauses 3.2 to 3.6. Loss that is excluded in "
                  "Section 4 is not insured, even if it is caused by one of these perils."),
                C("3.2", "Fire and lightning",
                  "Direct physical loss to the property caused by fire or lightning is covered. This includes "
                  "damage from smoke and heat, and the reasonable cost of putting out the fire."),
                C("3.3", "Windstorm",
                  "Direct physical loss to the dwelling caused by windstorm is covered. This includes loss to roof "
                  "coverings, gutters, fascia, siding and attached fences.",
                  "Rain, sleet or snow that enters the dwelling is covered only if a windstorm first damages the "
                  "roof or a wall and the damage creates an opening through which the rain, sleet or snow enters."),
                C("3.4", "Hail",
                  "Direct physical loss to the dwelling caused by hail is covered. This includes bruising, "
                  "cracking or puncture of roof shingles and other roof coverings, and damage to gutters, siding, "
                  "skylights and windows."),
                C("3.5", "Sudden escape of water",
                  "Sudden and accidental discharge or overflow of water from a plumbing, heating or "
                  "air-conditioning system, or from a household appliance, is covered. We also cover the "
                  "reasonable cost of opening and repairing the part of the building that must be removed to "
                  "reach the system."),
                C("3.6", "Other named perils",
                  "We cover direct physical loss caused by explosion, riot, vandalism, theft, falling objects, "
                  "and the weight of ice, snow or sleet on a building. Damage to the inside of a building caused "
                  "by falling objects is covered only if an object first damages the roof or an outside wall."),
            ]),
            ("SECTION 4  WHAT WE DO NOT COVER", [
                C("4.1", "Wear and tear",
                  "We do not cover loss caused by wear and tear, marring, gradual deterioration, rust, corrosion, "
                  "rot, mold, or an inherent defect in the property. Deterioration of roof coverings that "
                  "develops over time is not covered even if it is first noticed after a storm."),
                C("4.2", "Flood and ground movement",
                  "We do not cover loss caused by flood, groundwater seepage, storm surge, backup of sewers or "
                  "drains, landslide, earthquake or sinking of the ground."),
                C("4.3", "Neglect",
                  "We do not cover loss that is made worse because you failed to take reasonable steps to protect "
                  "the property from further damage after a loss."),
                C("4.4", "Intentional loss",
                  "We do not cover loss that is caused by an act that you or a person living with you committed "
                  "or arranged with the intent to cause a loss."),
                C("4.5", "Vacancy",
                  "We do not cover vandalism or breakage of glass if the dwelling has been vacant for more than "
                  "60 consecutive days before the loss."),
                C("4.6", "War and nuclear hazards",
                  "We do not cover loss caused by war, military action, or nuclear reaction or radiation."),
            ]),
            ("SECTION 5  WHAT TO DO AFTER A LOSS", [
                C("5.1", "Your duties after a loss",
                  "If you have a loss, you must do the following:",
                  "(a) Give us prompt notice, and in any case within 60 days of discovering the loss.",
                  "(b) Protect the property from further damage and keep records of what you spend on repairs.",
                  "(c) Take clear photographs of the damage and keep damaged items until we have inspected them.",
                  "(d) Tell us the date, the place and the cause of the loss, as far as you know them.",
                  "(e) Give us the reasonable access, documents and information we ask for."),
                C("5.2", "Evidence we may ask for",
                  "We may ask for photographs, an inspection report, repair estimates, receipts and weather "
                  "records for the date of loss. Weather records support our assessment of a claim but do not by "
                  "themselves prove that the damage was caused by a particular event."),
                C("5.3", "How we settle a loss",
                  "We settle covered loss to the dwelling on a replacement cost basis if you insure it to at "
                  "least 80% of its replacement cost. Otherwise we pay the actual cash value, which is the "
                  "replacement cost less a deduction for age and condition."),
                C("5.4", "Deductible",
                  "Your deductible applies to each occurrence and is subtracted from the amount we pay. The "
                  "deductible amount is shown in the Declarations."),
            ]),
            ("SECTION 6  GENERAL CONDITIONS", [
                C("6.1", "Fraud and misrepresentation",
                  "We provide no coverage for a person who has intentionally hidden or misrepresented a material "
                  "fact or circumstance, or has engaged in fraudulent conduct, in connection with this insurance "
                  "or a claim."),
                C("6.2", "Cancellation",
                  "You may cancel this policy at any time by notifying us. We may cancel it for non-payment of "
                  "premium with 10 days written notice, or for any other reason with 30 days written notice."),
                C("6.3", "Complaints and disputes",
                  "If you disagree with a claim decision, you may ask us to review it. A claim decision made with "
                  "the help of automated tools is always reviewed by a claims professional, and you may ask for "
                  "the reasons in writing. You may also contact your state insurance regulator."),
            ]),
        ],
    },
    {
        "file": "homeshield_storm_roof_endorsement_hse12.pdf",
        "title": "HomeShield Storm and Roof Endorsement (Form HS-E12)",
        "form": "HS-E12",
        "edition": "Edition 03/2026",
        "notice": (
            "This endorsement changes your HomeShield Homeowners Policy. Please keep it with your policy."
        ),
        "sections": [
            ("SECTION 1  ABOUT THIS ENDORSEMENT", [
                C("1.1", "How this endorsement works",
                  "This endorsement changes the HomeShield Homeowners Policy (Form HS-100). It applies when it is "
                  "listed in your Declarations. Where it conflicts with the policy form, this endorsement "
                  "applies. All other terms stay the same."),
            ]),
            ("SECTION 2  HAIL AND WINDSTORM TERMS", [
                C("2.1", "Separate deductible for hail and windstorm",
                  "A separate deductible of 1% of the dwelling limit applies to each loss caused by hail or "
                  "windstorm. This deductible replaces the standard deductible for those losses."),
                C("2.2", "Roof coverings older than 15 years",
                  "Loss to roof coverings caused by hail or windstorm is settled on an actual cash value basis if "
                  "the roof covering was more than 15 years old on the date of loss. We pay the cost to repair or "
                  "replace the roof covering less a deduction for age and condition."),
                C("2.3", "What counts as physical damage to a roof",
                  "Physical damage from hail includes a fractured or bruised shingle mat, a puncture, and loss of "
                  "granules that exposes the mat. Physical damage from windstorm includes shingles that are "
                  "lifted, creased, torn or missing. Damage of this kind is covered when it affects the ability "
                  "of the roof to keep out rain."),
                C("2.4", "Cosmetic damage",
                  "Cosmetic damage to metal roof panels, gutters and flashing, meaning denting or surface marring "
                  "that does not let rain in and does not shorten the life of the part, is not covered."),
                C("2.5", "Checking the date and place of a storm",
                  "To support a claim for hail or windstorm we compare the date and place of loss with "
                  "independent weather data. If the weather data does not show a hail or windstorm event near the "
                  "dwelling on the date of loss, we refer the claim for further review before we decide it."),
                C("2.6", "Temporary repairs",
                  "After a loss caused by hail or windstorm, we cover the reasonable cost of temporary repairs, "
                  "such as covering a damaged roof with a tarpaulin, up to $2,500. Keep your receipts and "
                  "photographs of the repair."),
                C("2.7", "Roof age statement",
                  "You must tell us the approximate age of the roof coverings when you apply for this "
                  "endorsement. An incorrect statement may affect how we settle a loss caused by hail or "
                  "windstorm."),
            ]),
            ("SECTION 3  CLAIMS FOR STORM LOSS", [
                C("3.1", "Information we need",
                  "For a claim for hail or windstorm damage, please tell us the date the storm happened and the "
                  "address where the damage occurred. Add photographs of the damage to the roof and to any other "
                  "part of the dwelling that was affected."),
            ]),
        ],
    },
    {
        "file": "homeshield_water_damage_endorsement_hse21.pdf",
        "title": "HomeShield Water Damage Endorsement (Form HS-E21)",
        "form": "HS-E21",
        "edition": "Edition 03/2026",
        "notice": (
            "This endorsement changes your HomeShield Homeowners Policy. Please keep it with your policy."
        ),
        "sections": [
            ("SECTION 1  ABOUT THIS ENDORSEMENT", [
                C("1.1", "How this endorsement works",
                  "This endorsement changes the HomeShield Homeowners Policy (Form HS-100). It applies when it is "
                  "listed in your Declarations. Where it conflicts with the policy form, this endorsement "
                  "applies. All other terms stay the same."),
            ]),
            ("SECTION 2  WATER DAMAGE TERMS", [
                C("2.1", "Sudden and accidental escape of water",
                  "We cover sudden and accidental escape of water from fixed pipes, tanks and household "
                  "appliances, and the resulting damage to ceilings, walls, floors and insured contents."),
                C("2.2", "Finding the source of a leak",
                  "We cover the reasonable cost of finding the source of a covered escape of water, including "
                  "opening walls or floors, and the cost of closing them again."),
                C("2.3", "Gradual leakage",
                  "Continuous or repeated seepage or leakage from a plumbing system, a roof or an appliance over "
                  "a period of 14 days or more is not covered."),
                C("2.4", "Flood and sewer backup",
                  "Flood, surface run-off, groundwater, storm surge and backup through sewers or drains are not "
                  "covered unless a flood endorsement is listed in your Declarations."),
                C("2.5", "Frozen plumbing",
                  "We cover damage caused by the freezing of a plumbing system if you kept the dwelling heated, "
                  "or if you shut off the supply and drained the system before leaving it unheated."),
                C("2.6", "Evidence for a water claim",
                  "For a water claim, we may ask for photographs of the damaged area and of the source of the "
                  "leak, a plumber's report, and repair invoices."),
                C("2.7", "Stopping the damage",
                  "If it is safe to do so, turn off the supply to stop the escape of water and move your "
                  "belongings out of the affected area. We cover the reasonable cost of emergency measures to "
                  "prevent further damage."),
            ]),
        ],
    },
    {
        "file": "homeshield_claims_guide_hsg01.pdf",
        "title": "HomeShield Claims Guide (Form HS-G01)",
        "form": "HS-G01",
        "edition": "Edition 03/2026",
        "notice": (
            "This guide explains how to make a claim and how we review it. It does not change the terms of "
            "your policy."
        ),
        "sections": [
            ("SECTION 1  MAKING A CLAIM", [
                C("1.1", "How to report a claim",
                  "You can report a claim online, by telephone or through your agent. Please report a loss as "
                  "soon as you can. Delay can make it harder to establish the cause of the damage."),
                C("1.2", "What to include",
                  "Include your policy number, the date and place of the loss, a short description of what "
                  "happened, and clear photographs of the damage taken from several angles and distances."),
                C("1.3", "Storm claims",
                  "For hail and windstorm claims, give the date of the storm and the address of the home. We use "
                  "these details to check independent weather data for that day and place."),
                C("1.4", "Water claims",
                  "For water claims, stop the source of the leak if you can do so safely, and photograph the "
                  "damage before you clean up."),
                C("1.5", "Fire claims",
                  "For fire claims, contact the fire service first. Tell us as soon as it is safe to do so, and "
                  "do not throw away damaged items until we have seen them."),
            ]),
            ("SECTION 2  HOW WE REVIEW A CLAIM", [
                C("2.1", "What we look at",
                  "We review the photographs you send, the wording of your policy and, for storm claims, "
                  "independent weather data. We may also ask for an inspection by an adjuster."),
                C("2.2", "Use of automated tools",
                  "We may use automated tools to help prepare a first assessment. These tools only advise. A "
                  "claims professional makes every decision, and no claim is denied by an automated tool alone."),
                C("2.3", "Time to decide",
                  "We aim to acknowledge a claim within 3 business days and to decide it within 30 days of "
                  "receiving the information we need."),
                C("2.4", "If you disagree",
                  "If you disagree with our decision, you can ask us to review it, send us more evidence, or "
                  "contact your state insurance regulator."),
            ]),
        ],
    },
]


# ---------------------------------------------------------------------------------------------------------
# Layout engine: draws each clause as one unbroken block so a clause never splits across two pages.
# ---------------------------------------------------------------------------------------------------------

def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if pymupdf.get_text_length(trial, fontname=font, fontsize=size) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


class _Writer:
    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.page = None
        self.y = BODY_TOP
        self.new_page()

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = BODY_TOP

    def ensure(self, height: float) -> None:
        if self.y + height > BODY_BOTTOM:
            self.new_page()

    def line(self, text: str, font: str, size: float, x: float = MARGIN_X, color=(0, 0, 0)) -> None:
        self.page.insert_text((x, self.y + size), text, fontname=font, fontsize=size, color=color)
        self.y += LINE if size <= 11 else size + 6

    def rule(self) -> None:
        self.page.draw_line((MARGIN_X, self.y), (PAGE_W - MARGIN_X, self.y), color=RULE, width=0.6)
        self.y += 8

    def cover(self, title: str, subtitle: str, notice: str) -> None:
        for part in _wrap(title, BOLD, 16, TEXT_W):
            self.line(part, BOLD, 16)
        self.y += 2
        self.line(subtitle, REGULAR, 10, color=GREY)
        self.y += 2
        self.rule()
        for part in _wrap(notice, ITALIC, 9.5, TEXT_W):
            self.line(part, ITALIC, 9.5, color=GREY)
        self.y += 10

    def banner_height(self) -> float:
        return 14 + 17 + 8

    def banner(self, text: str) -> None:
        self.y += 14
        self.line(text, BOLD, 11)
        self.rule()

    @staticmethod
    def clause_lines(clause: tuple[str, str, tuple[str, ...]]) -> list[tuple[str, bool, bool]]:
        """(text, is_bold, indented) for every line the clause will draw, so its height is known in advance."""
        number, heading, paragraphs = clause
        out: list[tuple[str, bool, bool]] = [(f"{number} {heading}", True, False)]
        for paragraph in paragraphs:
            indented = paragraph.startswith("(")
            width = TEXT_W - (INDENT if indented else 0)
            out.extend((part, False, indented) for part in _wrap(paragraph, REGULAR, 10, width))
        return out

    def clause_height(self, clause) -> float:
        return len(self.clause_lines(clause)) * LINE + PARA_GAP * len(clause[2]) + 8

    def clause(self, clause) -> None:
        for text, bold, indented in self.clause_lines(clause):
            self.line(text, BOLD if bold else REGULAR, 10.5 if bold else 10, x=MARGIN_X + (INDENT if indented else 0))
        self.y += PARA_GAP * len(clause[2]) + 8

    def finish(self, doc_meta: dict, path: Path) -> None:
        total = len(self.doc)
        for number, page in enumerate(self.doc, start=1):
            page.insert_text((MARGIN_X, 38), "HomeShield Insurance Company", fontname=REGULAR, fontsize=8, color=GREY)
            page.insert_text((PAGE_W - MARGIN_X - 60, 38), f"Form {doc_meta['form']}", fontname=REGULAR, fontsize=8, color=GREY)
            page.draw_line((MARGIN_X, 46), (PAGE_W - MARGIN_X, 46), color=RULE, width=0.6)
            page.draw_line((MARGIN_X, 806), (PAGE_W - MARGIN_X, 806), color=RULE, width=0.6)
            page.insert_text((MARGIN_X, 820), f"Form {doc_meta['form']}  |  {doc_meta['edition']}", fontname=REGULAR, fontsize=8, color=GREY)
            page.insert_text((PAGE_W - MARGIN_X - 52, 820), f"Page {number} of {total}", fontname=REGULAR, fontsize=8, color=GREY)
        self.doc.set_metadata({
            "title": doc_meta["title"],
            "author": "HomeShield Insurance Company",
            "subject": "Synthetic policy wording for the ClaimGuard demo. Not a real insurance contract.",
        })
        self.doc.save(path)
        self.doc.close()


def build_pdf(document: dict, path: Path) -> None:
    writer = _Writer()
    writer.cover(document["title"], f"HomeShield Insurance Company  |  {document['edition']}", document["notice"])
    for banner, clauses in document["sections"]:
        # Keep a section banner together with its first clause.
        writer.ensure(writer.banner_height() + writer.clause_height(clauses[0]))
        writer.banner(banner)
        for clause in clauses:
            writer.ensure(writer.clause_height(clause))
            writer.clause(clause)
    writer.finish(document, path)


def build_library(out_dir: Path | str) -> list[Path]:
    """Write every document of the library as a searchable PDF. Safe to run again: files are replaced."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for document in DOCUMENTS:
        path = out_dir / document["file"]
        build_pdf(document, path)
        paths.append(path)
    return paths
