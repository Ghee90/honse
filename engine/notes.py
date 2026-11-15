"""Multilingual (Finnish, Swedish, English) dealer note parser following RULES.md."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional

from spread.types import Note, TimeWindow

HOUR, DAY = 60, 24 * 60
DAYS_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# --- Boolean car / stop flags ---
PLATES_POS = re.compile(
    r"ilman\s+kilpi|ei\s+kilpi|kilpiä\s+ei|puuttuu\s+kilv|kilvet\s+puuttu|kilvet\s+ir+ot|rekisterikilvet\s+ir+ot"
    r"|ei\s+rekkar\w*|ilman\s+rekkar\w*|rekkar\w*\s+puuttu|rekkar\w*\s+ir+ot"
    r"|koek?ilv|siirtokilv|tarrakilv|rekisteröimät|kilvet\s+mukaan|tuo\s+kilv|ota\s+kilv"
    r"|utan\s+skylt|inga\s+skylt|saknar\s+skylt|skyltar\w*\s+borttag|prövningsskylt|provskylt|transportskylt|oregistrerad|medtag\s+skylt|ta\s+med\s+skylt"
    r"|no\s+plates|without\s+plates|missing\s+plates|plates\s+removed|plates\s+detached|unregistered"
    r"|trade\s+plates|transfer\s+plates|dealer\s+plates|bring\s+plates|needs?\s+plates",
    re.IGNORECASE,
)
PLATES_NEG = re.compile(
    r"kilvet\s+(?:on|paikallaan|kiinni|autossa)|ei\s+tarvitse\s+kilpi"
    r"|skyltar\s+finns|har\s+skyltar|behöver\s+inte\s+skylt"
    r"|plates\s+(?:on|attached|included)|has\s+plates|no\s+plates\s+needed",
    re.IGNORECASE,
)

MANUAL_POS = re.compile(
    r"manuaali|käsivaih|keppivaih"
    r"|manuell|växelspak"
    r"|\bmanual\b|stick\s*shift|clutch",
    re.IGNORECASE,
)
MANUAL_NEG = re.compile(
    r"ei\s+manuaali|(?<!ei\s)automaatti"
    r"|inte\s+manuell|(?<!inte\s)(?<!ej\s)\bautomat\b"
    r"|not\s+manual|(?<!not\s)\bautomatic\b",
    re.IGNORECASE,
)

SUMMER_TYRES_POS = re.compile(
    r"kesärenka|kesäpyör|kesägum|ei\s+talvirenka|talvirenka\w*\s+ei\b"
    r"|sommardäck|inga\s+vinterdäck|vinterdäck\w*\s+(?:inte|ej)\b"
    r"|s(?:um+|mu)mer\s+t[yi]re|no\s+winter\s+t[yi]re|winter\s+t[yi]res?\s+not\b",
    re.IGNORECASE,
)
# Spare summer tyres carried inside the trunk/boot do NOT mean the car drives on summer tyres!
SUMMER_TYRES_TRUNK = re.compile(
    r"takakont\w*|kontissa|tavaratil\w*|kyydissä|menee\s+mukana|mukaan\s+autoon"
    r"|bakluck\w*|bagage\w*|följer\s+med"
    r"|\btrunk\b|\bboot\b|go(?:es)?\s+with\s+the\s+car|in\s+the\s+back",
    re.IGNORECASE,
)

PAYMENT_WAIT_POS = re.compile(
    r"maksu|maksettu|maksaa\s+ensin|suoritukse|rahat\s+tilill"
    r"|betalning|betald|betalar\s+först"
    r"|payment|paid|pays?\s+first|funds\s+clear|bank\s+transfer",
    re.IGNORECASE,
)

# --- Correction & negation helpers ---
CORRECTION_SPLIT = re.compile(
    r"\b(?:eikun?|ei\s+kun|korjaus|anteeksi|sori|oikeastaan|eli\s+siis"
    r"|rättelse|ursäkta|förlåt|nej\s+vänta|menar"
    r"|sorry|correction|actually|wait,?\s+no|rather|i\s+mean)\b\s*:?",
    re.IGNORECASE,
)
DOT_BEFORE_CORRECTION = re.compile(
    r"[.;!]\s+(?=(?:eikun?|ei\s+kun|korjaus|anteeksi|sori|oikeastaan|eli\s+siis|rättelse|ursäkta|förlåt|nej\s+vänta|menar|sorry|correction|actually|wait|rather|i\s+mean)\b)",
    re.IGNORECASE,
)
REF_NOISE_RE = re.compile(
    r"\b(?:viite|tilaus\w*|order\s*n[ro]\w*|ref(?:erence)?)\s*[:#]?\s*\d+\b",
    re.IGNORECASE,
)

NEGATED_TIME = re.compile(
    r"\b(?:ei|not|inte|ej)\s+(?!ennen\b|före\b|before\b|earlier\b|aikaisemmin\b|tidigare\b)(?:klo\.?\s*|kl\.?\s*|at\s+)?\d{1,2}(?:[.:]\d{2})?(?:\s*(?:am|pm))?",
    re.IGNORECASE,
)

# Sentence splitter that preserves decimal clock times ("13.30") and abbreviations ("viim.", "huom.", "klo.", "ma.")
ABBREV_DOT_GUARD = re.compile(
    r"\b(viim|huom|klo|kl|ma|ti|ke|to|pe|la|su|mån|tis|ons|tors|tor|fre|lör|sön|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|no|nr)\.",
    re.IGNORECASE,
)
CLAUSE_SPLIT = re.compile(
    r"\.(?!\d)\s+|[;!\n/]+|\b(?:ja|sekä|mutta|och|men|and|but)\b",
    re.IGNORECASE,
)

# --- Delivery vs Pickup clause markers ---
DELIVERY_KW = re.compile(
    r"toimit\w*|perill[äe]|perile|vastaanot\w*|luovutus\s+asiakkaalle|kohde|kohtee\w*|määränpää\w*"
    r"|leverer\w*|leverans\w*|fram+e?m?\b|mottag\w*|på\s+plats|vara\s+där"
    r"|desti\w*|deliver\w*|arriv\w*|drop\s*-?off|receiv\w*|be\s+there",
    re.IGNORECASE,
)
PICKUP_KW = re.compile(
    r"nou[td]\w*|haettav\w*|hakea|luovutus|valmis|maksu\w*|maksaa|auton\s+saa|huolto\s+kesken"
    r"|hämt\w*|upphämt\w*|klar|betal\w*|verkstad\w*"
    r"|pick\s*u+p\w*|collect\w*|\bready\b|handover|release|workshop",
    re.IGNORECASE,
)
# Customer picking up AT destination ("asiakas tulee hakemaan, perillä viim...") is delivery, not driver pickup!
CUSTOMER_FETCH_NEG = re.compile(r"asiakas\s+(?:tulee\s+)?hak\w*|kund\w*\s+hämt\w*|customer\s+(?:will\s+)?pick\w*(?:\s+it\s+up)?", re.IGNORECASE)

AFTER_KW = re.compile(
    r"jälkeen|alkaen|aikaisintaan|ei\s+ennen|\bvasta\b|\bvalmis\b|auton\s+saa|voi\s+noutaa|noudettavissa"
    r"|efter|från\s+och\s+med|och\s+framåt|tidigast|inte\s+före|ej\s+före|\bförst\b|\bklar\b|kan\s+hämtas"
    r"|\bafter\b|\bfrom\b|onwards|earliest|not\s+before|no\s+earlier|\bready\b",
    re.IGNORECASE,
)
BEFORE_KW = re.compile(
    r"ennen|mennessä|viimeistään|\bviim\b|asti|saakka|sulke\w*|kiinni"
    r"|före|senast|innan|till\s+kl|stänger"
    r"|\bbefore\b|\bby\b|latest|until|no\s+later|closes?\b",
    re.IGNORECASE,
)
CLOSES_DAILY_KW = re.compile(
    r"sulke\w*|kiinni|stänger|closes?\b|auki\w*|öppet|opening\s+hours|vastaanot\w*|mottagning|receiving",
    re.IGNORECASE,
)
CLOSED_AFTER_RE = re.compile(
    r"(?:kiinni|sulke\w*|stänger|closed?)\s+(?:klo\.?\s*|kl\.?\s*|at\s+|after\s+)?(\d{1,2})(?:[.:](\d{2}))?\s*(?:jälkeen|efter)?",
    re.IGNORECASE,
)

# Relative days & weekdays (RULES.md: weekday name = 2..6 days after arrival)
DAY_PLUS_2 = re.compile(r"\b(?:ylihuomenna|i\s+övermorgon|övermorgon|day\s+after\s+tomorrow)\b", re.IGNORECASE)
DAY_PLUS_1 = re.compile(r"\b(?:huomenna|huom\b|i\s+morgon|imorgon|tomorrow|tmrw|tmw)\b", re.IGNORECASE)
DAY_PLUS_0 = re.compile(r"\b(?:tänään|idag|i\s+dag|today)\b", re.IGNORECASE)

WEEKDAY_PATTERNS_COMMON: list[tuple[int, re.Pattern[str]]] = [
    (0, re.compile(r"\b(?:maanantai\w*|\bma\b(?!\s*[-–]\s*pe)|måndag\w*|\bmån\b|monday|\bmon\b(?!\s*[-–]\s*fri))\b", re.IGNORECASE)),
    (1, re.compile(r"\b(?:tiistai\w*|\bti\b|tisdag\w*|\btis\b|tuesday|\btues?\b)\b", re.IGNORECASE)),
    (2, re.compile(r"\b(?:keskiviikko\w*|\bke\b|onsdag\w*|\bons\b|wednesday|\bwed\b)\b", re.IGNORECASE)),
    (3, re.compile(r"\b(?:to?rstai\w*|torsdag\w*|\btors?\b|thursday|\bthurs?\b|\bthu\b)\b", re.IGNORECASE)),
    (4, re.compile(r"\b(?:perjantai\w*|\bpe\b|fredag\w*|\bfre\b|friday|\bfri\b)\b", re.IGNORECASE)),
    (5, re.compile(r"\b(?:lauantai\w*|\bla\b|lördag\w*|\blör\b|saturday|\bsat\b)\b", re.IGNORECASE)),
    (6, re.compile(r"\b(?:sunnuntai\w*|\bsu\b|söndag\w*|\bsön\b|sunday|\bsun\b)\b", re.IGNORECASE)),
]
FI_THU_SHORT = re.compile(r"\bto\b", re.IGNORECASE)

# Word numbers -> working hours 9..17 ("two" = 14:00, "nine" = 09:00)
FI_TEEN_RE = re.compile(r"^(yhde\w*toista|yhte\w*toista|yksitoista|kahde\w*toista|kahte\w*toista|kaksitoista)$", re.IGNORECASE)

WORD_NUMS: dict[str, int] = {
    "yksitoista": 11, "yhdentoista": 11, "yhdeltätoista": 11, "yhdestätoista": 11, "yhteentoista": 11,
    "kaksitoista": 12, "kahdentoista": 12, "kahdeltatoista": 12, "kahdestatoista": 12, "kahteentoista": 12, "puolipäivä": 12,
    "yksi": 1, "yhde": 1, "yhte": 1,
    "kaksi": 2, "kahde": 2, "kahte": 2,
    "kolme": 3,
    "neljä": 4,
    "viisi": 5, "viide": 5, "viiteen": 5,
    "kuusi": 6, "kuude": 6, "kuute": 6,
    "seitsemä": 7, "seiska": 7,
    "kahdeksa": 8, "kasi": 8,
    "yhdeksä": 9, "ysi": 9,
    "kymmene": 10, "kympi": 10,
    # SV
    "elva": 11, "tolv": 12,
    "ett": 1, "två": 2, "tre": 3, "fyra": 4, "fem": 5, "sex": 6, "sju": 7, "åtta": 8, "nio": 9, "tio": 10,
    # EN
    "eleven": 11, "twelve": 12, "noon": 12,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

RANGE_RE = re.compile(
    r"\b(\d{1,2})(?:[.:](\d{2}))?\s*(am|pm)?\s*[-–—]\s*(\d{1,2})(?:[.:](\d{2}))?\s*(am|pm)?\b",
    re.IGNORECASE,
)
FOUR_DIGIT_CLOCK_RE = re.compile(
    r"\b(?:klo|kello|kl|at|before|after|until|by|ennen|jälkeen|aikaisintaan|viimeistään|viim|före|senast|efter|tidigast)\s+([01]\d|2[0-3])([0-5]\d)\b",
    re.IGNORECASE,
)
DIGIT_CLOCK_RE = re.compile(
    r"(?:(?:klo\b|kello\b|kl\b)\s*(\d{1,2})(?:[.:](\d{2}))?\b)"
    r"|\b(\d{1,2})[.:](\d{2})\s*(am|pm)?\b"
    r"|\b(\d{1,2})\s*(am|pm)\b"
    r"|(?:(?:ennen|mennessä|asti|saakka|jälkeen|alkaen|aikaisintaan|före|senast|efter|tidigast|before|after|until|by)\s+(\d{1,2})\b)"
    r"|\b(\d{1,2})\s+(?:mennessä|asti|saakka|alkaen|jälkeen)\b",
    re.IGNORECASE,
)
HALF_WORD_RE = re.compile(
    r"\b(?:(puoli|halv|half)\s+(?!past\b)([a-zåäö]+)|half\s+past\s+([a-z]+))\b",
    re.IGNORECASE,
)


def _work_hour(n: int) -> int:
    """Converts spoken hour 1..12 to working-day 24h clock (RULES.md: 9 to 17, 'two' is 14:00)."""
    if 1 <= n <= 7:
        return n + 12
    return n


def _ampm_hour(hh: int, ampm: Optional[str], work_fallback: bool = True) -> int:
    if ampm:
        ap = ampm.lower()
        if ap == "pm" and hh < 12:
            return hh + 12
        if ap == "am" and hh == 12:
            return 0
        return hh
    if work_fallback and 1 <= hh <= 6:
        return hh + 12
    return hh


def _match_word_num(token: str) -> Optional[int]:
    low = token.lower()
    if FI_TEEN_RE.match(low):
        return 11 if low.startswith(("yhd", "yht", "yks")) else 12
    for stem, val in sorted(WORD_NUMS.items(), key=lambda kv: -len(kv[0])):
        if low.startswith(stem):
            return val
    return None


def fmt_min(m: int) -> str:
    d, r = divmod(int(m), DAY)
    return f"{DAYS_SHORT[d % 7]} {r // 60:02d}:{r % 60:02d}"


@dataclass
class ParsedNote:
    needs_plates: bool = False
    manual_gearbox: bool = False
    summer_tyres: bool = False
    wait_payment: bool = False
    pickup_after: Optional[int] = None
    pickup_before: Optional[int] = None
    pickup_hours: Optional[tuple[int, int]] = None
    delivery_after: Optional[int] = None
    delivery_before: Optional[int] = None
    delivery_hours: Optional[tuple[int, int]] = None
    summary: list[str] = field(default_factory=list)

    @property
    def has_constraints(self) -> bool:
        return bool(
            self.needs_plates
            or self.manual_gearbox
            or self.summer_tyres
            or self.wait_payment
            or self.pickup_after is not None
            or self.pickup_before is not None
            or self.pickup_hours is not None
            or self.delivery_before is not None
            or self.delivery_hours is not None
        )


def _resolve_day(clause: str, arrived_day: int, default_day: int, lang: str = "") -> tuple[int, bool]:
    if DAY_PLUS_2.search(clause):
        return arrived_day + 2, True
    if DAY_PLUS_1.search(clause):
        return arrived_day + 1, True
    if DAY_PLUS_0.search(clause):
        return arrived_day, True
    for wd, pat in WEEKDAY_PATTERNS_COMMON:
        if pat.search(clause):
            for delta in range(2, 7):
                if (arrived_day + delta) % 7 == wd:
                    return arrived_day + delta, True
    if (lang == "fi" or re.search(r"\bto\s+klo\b", clause, re.IGNORECASE)) and FI_THU_SHORT.search(clause):
        for delta in range(2, 7):
            if (arrived_day + delta) % 7 == 3:
                return arrived_day + delta, True
    return default_day, False


def parse_note(
    note: Optional[Note],
    arrived_at: int,
    default_pickup_from: int,
    default_delivery_to: int,
) -> ParsedNote:
    res = ParsedNote()
    if not note or not note.get("text"):
        return res

    raw_text = note["text"]
    lang = (note.get("language") or "").lower()

    parts_corr = [p.strip() for p in CORRECTION_SPLIT.split(raw_text) if p.strip()]
    effective_text = parts_corr[-1] if len(parts_corr) > 1 else raw_text

    if (PLATES_POS.search(effective_text) or PLATES_POS.search(raw_text)) and not PLATES_NEG.search(effective_text):
        res.needs_plates = True
        res.summary.append("Needs Plates")

    if (MANUAL_POS.search(effective_text) or MANUAL_POS.search(raw_text)) and not MANUAL_NEG.search(effective_text):
        res.manual_gearbox = True
        res.summary.append("Manual Gearbox")

    if SUMMER_TYRES_POS.search(raw_text) and not SUMMER_TYRES_TRUNK.search(raw_text):
        res.summer_tyres = True
        res.summary.append("Summer Tyres")

    if PAYMENT_WAIT_POS.search(raw_text):
        res.wait_payment = True
        res.summary.append("Wait Payment")

    arrived_day = arrived_at // DAY
    pickup_day_default = default_pickup_from // DAY

    # Strip dots from known abbreviations ("viim.", "klo.", "ma.") and keep corrections in the same clause
    norm_text = REF_NOISE_RE.sub(" ", ABBREV_DOT_GUARD.sub(r"\1", DOT_BEFORE_CORRECTION.sub(", ", raw_text)))
    clauses = [c.strip() for c in CLAUSE_SPLIT.split(norm_text) if c.strip()]

    for raw_clause in clauses:
        sub = [p.strip() for p in CORRECTION_SPLIT.split(raw_clause) if p.strip()]
        override_day: Optional[int] = None
        if len(sub) > 1 and (DIGIT_CLOCK_RE.search(sub[-1]) or HALF_WORD_RE.search(sub[-1]) or RANGE_RE.search(sub[-1]) or FOUR_DIGIT_CLOCK_RE.search(sub[-1])):
            prefix = ""
            if (DELIVERY_KW.search(sub[0]) or CUSTOMER_FETCH_NEG.search(sub[0])) and not (DELIVERY_KW.search(sub[-1]) or CUSTOMER_FETCH_NEG.search(sub[-1])):
                prefix += "toimitus "
            for dpat in (DAY_PLUS_2, DAY_PLUS_1, DAY_PLUS_0):
                dm = dpat.search(sub[0])
                if dm and not (DAY_PLUS_2.search(sub[-1]) or DAY_PLUS_1.search(sub[-1]) or DAY_PLUS_0.search(sub[-1])):
                    prefix += dm.group(0) + " "
                    break
            else:
                for _, wpat in WEEKDAY_PATTERNS_COMMON:
                    wm = wpat.search(sub[0])
                    if wm and not any(wp.search(sub[-1]) for _, wp in WEEKDAY_PATTERNS_COMMON):
                        prefix += wm.group(0) + " "
                        break
            if AFTER_KW.search(sub[0]) and not (AFTER_KW.search(sub[-1]) or BEFORE_KW.search(sub[-1])):
                prefix += "alkaen "
            elif BEFORE_KW.search(sub[0]) and not (AFTER_KW.search(sub[-1]) or BEFORE_KW.search(sub[-1])):
                prefix += "mennessä "
            clause = prefix + sub[-1]
        elif len(sub) > 1:
            corr_d, corr_exp = _resolve_day(sub[-1], arrived_day, arrived_day, lang=lang)
            if corr_exp:
                override_day = corr_d
                clause = sub[0]
            else:
                clause = raw_clause
        else:
            clause = raw_clause

        clause_clean = NEGATED_TIME.sub(" ", clause)
        is_cust_fetch = bool(CUSTOMER_FETCH_NEG.search(clause_clean))
        clause_for_role = CUSTOMER_FETCH_NEG.sub(" ", clause_clean)

        has_del = bool(DELIVERY_KW.search(clause_for_role)) or is_cust_fetch
        has_pick = bool(PICKUP_KW.search(clause_for_role)) and not is_cust_fetch
        target = "delivery" if (has_del and not has_pick) else "pickup"

        base_day = max(arrived_day, pickup_day_default)
        day_idx, explicit_day = _resolve_day(clause_clean, arrived_day, base_day, lang=lang)
        if override_day is not None:
            day_idx, explicit_day = override_day, True

        # 0. Explicit "closed after HH:MM" ("kiinni klo 15 jälkeen" -> closes at 15:00!)
        cm_close = CLOSED_AFTER_RE.search(clause_clean)
        if cm_close and not explicit_day and not RANGE_RE.search(clause_clean):
            ch = _ampm_hour(int(cm_close.group(1)), None)
            cmin = int(cm_close.group(2) or 0)
            closes_tod = ch * HOUR + cmin
            if 10 * HOUR <= closes_tod <= 23 * HOUR:
                if target == "delivery":
                    cur_o = res.delivery_hours[0] if res.delivery_hours else 9 * HOUR
                    res.delivery_hours = (cur_o, closes_tod)
                else:
                    cur_o = res.pickup_hours[0] if res.pickup_hours else 9 * HOUR
                    res.pickup_hours = (cur_o, closes_tod)
                continue

        # 1. Check hour ranges e.g. "9-16", "klo 8.30-15.00", "9am-4:30pm"
        for m in RANGE_RE.finditer(clause_clean):
            # Ignore order/reference numbers or date ranges if out of bounds
            h1, m1, ap1 = int(m.group(1)), int(m.group(2) or 0), m.group(3)
            h2, m2, ap2 = int(m.group(4)), int(m.group(5) or 0), m.group(6)
            if h1 <= 24 and h2 <= 24 and m1 < 60 and m2 < 60:
                h1 = _ampm_hour(h1, ap1, work_fallback=False)
                h2 = _ampm_hour(h2, ap2, work_fallback=False)
                if not ap1 and not ap2:
                    if h1 < 7 and h2 <= 7:
                        h1, h2 = _work_hour(h1), _work_hour(h2)
                    elif h1 >= 7 and h2 <= 6:
                        h2 += 12
                opens, closes = h1 * HOUR + m1, h2 * HOUR + m2
                if 5 * HOUR <= opens < closes <= 24 * HOUR:
                    if target == "delivery":
                        res.delivery_hours = (opens, closes)
                    else:
                        res.pickup_hours = (opens, closes)
        clause_no_range = RANGE_RE.sub(" ", clause_clean)

        # Collect extracted (char_start, char_end, minute_of_day)
        extracted: list[tuple[int, int, int]] = []

        for m4 in FOUR_DIGIT_CLOCK_RE.finditer(clause_no_range):
            hh4, mm4 = int(m4.group(1)), int(m4.group(2))
            extracted.append((m4.start(), m4.end(), hh4 * HOUR + mm4))
        clause_no_4d = FOUR_DIGIT_CLOCK_RE.sub(lambda m: " " * (m.end() - m.start()), clause_no_range)

        for m in DIGIT_CLOCK_RE.finditer(clause_no_4d):
            if m.group(1) is not None:
                hh, mm, ampm = int(m.group(1)), int(m.group(2) or 0), None
            elif m.group(3) is not None:
                hh, mm, ampm = int(m.group(3)), int(m.group(4) or 0), m.group(5)
            elif m.group(6) is not None:
                hh, mm, ampm = int(m.group(6)), 0, m.group(7)
            elif m.group(8) is not None:
                hh, mm, ampm = int(m.group(8)), 0, None
            else:
                hh, mm, ampm = int(m.group(9)), 0, None
            hh = _ampm_hour(hh, ampm)
            if 0 <= hh <= 24 and 0 <= mm < 60:
                extracted.append((m.start(), m.end(), hh * HOUR + mm))

        used_spans: list[tuple[int, int]] = []
        for m in HALF_WORD_RE.finditer(clause_no_range):
            if m.group(2):
                wn = _match_word_num(m.group(2))
                if wn is not None:
                    hr = _work_hour(wn) - 1
                    extracted.append((m.start(), m.end(), hr * HOUR + 30))
                    used_spans.append((m.start(), m.end()))
            elif m.group(3):
                wn = _match_word_num(m.group(3))
                if wn is not None:
                    hr = _work_hour(wn)
                    extracted.append((m.start(), m.end(), hr * HOUR + 30))
                    used_spans.append((m.start(), m.end()))

        for wm in re.finditer(r"\b([a-zåäö]{3,18})\b", clause_no_range, re.IGNORECASE):
            if any(s <= wm.start() < e for s, e in used_spans):
                continue
            wn = _match_word_num(wm.group(1))
            if wn is None:
                continue
            around = clause_no_range[max(0, wm.start() - 25) : min(len(clause_no_range), wm.end() + 25)]
            if AFTER_KW.search(around) or BEFORE_KW.search(around) or re.search(r"\b(?:klo|kl|at)\b", around, re.IGNORECASE):
                extracted.append((wm.start(), wm.end(), _work_hour(wn) * HOUR))

        for s_pos, e_pos, tod in extracted:
            abs_t = day_idx * DAY + tod
            ctx = clause_no_range[max(0, s_pos - 30) : min(len(clause_no_range), e_pos + 28)]
            is_after = bool(AFTER_KW.search(ctx) or AFTER_KW.search(clause_no_range))
            is_before = bool(BEFORE_KW.search(ctx) or BEFORE_KW.search(clause_no_range))

            # If no explicit day is named and the clause states a general closing/until time ("We close at 16:30",
            # "vastaanotto klo 15 asti", "Nouto ennen klo 16"), treat it as daily closing hours!
            if not explicit_day and is_before and not is_after and 11 * HOUR <= tod <= 22 * HOUR:
                if target == "delivery" and CLOSES_DAILY_KW.search(clause_no_range):
                    cur_o = res.delivery_hours[0] if res.delivery_hours else 0
                    res.delivery_hours = (cur_o, tod)
                    continue
                if target == "pickup":
                    cur_o = res.pickup_hours[0] if res.pickup_hours else 0
                    res.pickup_hours = (cur_o, tod)
                    continue

            if target == "delivery":
                if is_before or not is_after:
                    res.delivery_before = min(res.delivery_before, abs_t) if res.delivery_before is not None else abs_t
                else:
                    res.delivery_after = max(res.delivery_after, abs_t) if res.delivery_after is not None else abs_t
            else:
                if is_after or (res.wait_payment and not is_before):
                    res.pickup_after = max(res.pickup_after, abs_t) if res.pickup_after is not None else abs_t
                elif is_before:
                    res.pickup_before = min(res.pickup_before, abs_t) if res.pickup_before is not None else abs_t

        if not extracted and target == "pickup" and day_idx > pickup_day_default and AFTER_KW.search(clause_clean):
            abs_t = day_idx * DAY + 9 * HOUR
            res.pickup_after = max(res.pickup_after, abs_t) if res.pickup_after is not None else abs_t

    if res.pickup_after is not None:
        res.summary.append(f"Pickup >= {fmt_min(res.pickup_after)}")
    if res.pickup_before is not None:
        res.summary.append(f"Pickup <= {fmt_min(res.pickup_before)}")
    if res.pickup_hours is not None:
        o, c = res.pickup_hours
        res.summary.append(f"Pickup hrs {o//60:02d}:{o%60:02d}-{c//60:02d}:{c%60:02d}")
    if res.delivery_before is not None:
        res.summary.append(f"Due <= {fmt_min(res.delivery_before)}")
    if res.delivery_hours is not None:
        o, c = res.delivery_hours
        res.summary.append(f"Deliv hrs {o//60:02d}:{o%60:02d}-{c//60:02d}:{c%60:02d}")

    return res


def apply_note_to_window(parsed: ParsedNote, base: TimeWindow, kind: str) -> TimeWindow:
    start, end = base["from"], base["to"]
    hours = parsed.pickup_hours if kind == "pickup" else parsed.delivery_hours
    if hours is not None:
        opens, closes = hours
        if opens > 0:
            start = max(start, (start // DAY) * DAY + opens)
        if closes > 0:
            end = min(end, (end // DAY) * DAY + closes)
    if kind == "pickup":
        if parsed.pickup_after is not None:
            start = max(start, parsed.pickup_after)
            if hours is not None and hours[0] > 0:
                start = max(start, (start // DAY) * DAY + hours[0])
        if parsed.pickup_before is not None:
            pb = parsed.pickup_before
            if pb < start and (pb % DAY) > (start % DAY):
                pb = (start // DAY) * DAY + (pb % DAY)
            end = min(end, pb)
    else:
        if parsed.delivery_after is not None:
            start = max(start, parsed.delivery_after)
        if parsed.delivery_before is not None:
            end = min(end, parsed.delivery_before)
    return {"from": start, "to": end} if start <= end else base
