import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
import re

VALID_COUNTRIES = {
    "AFG","ALB","DZA","AND","AGO","ATG","ARG","ARM","AUS","AUT","AZE","BHS","BHR",
    "BGD","BRB","BLR","BEL","BLZ","BEN","BTN","BOL","BIH","BWA","BRA","BRN","BGR",
    "BFA","BDI","CPV","KHM","CMR","CAN","CAF","TCD","CHL","CHN","COL","COM","COD",
    "COG","CRI","CIV","HRV","CUB","CYP","CZE","DNK","DJI","DOM","ECU","EGY","SLV",
    "GNQ","ERI","EST","SWZ","ETH","FJI","FIN","FRA","GAB","GMB","GEO","DEU","GHA",
    "GRC","GRD","GTM","GIN","GNB","GUY","HTI","HND","HUN","ISL","IND","IDN","IRN",
    "IRQ","IRL","ISR","ITA","JAM","JPN","JOR","KAZ","KEN","KIR","PRK","KOR","KWT",
    "KGZ","LAO","LVA","LBN","LSO","LBR","LBY","LIE","LTU","LUX","MDG","MWI","MYS",
    "MDV","MLI","MLT","MHL","MRT","MUS","MEX","FSM","MDA","MCO","MNG","MNE","MAR",
    "MOZ","MMR","NAM","NRU","NPL","NLD","NZL","NIC","NER","NGA","MKD","NOR","OMN",
    "PAK","PLW","PAN","PNG","PRY","PER","PHL","POL","PRT","QAT","ROU","RUS","RWA",
    "KNA","LCA","VCT","WSM","SMR","STP","SAU","SEN","SRB","SLE","SGP","SVK","SVN",
    "SLB","SOM","ZAF","SSD","ESP","LKA","SDN","SUR","SWE","CHE","SYR","TWN","TJK",
    "TZA","THA","TLS","TGO","TON","TTO","TUN","TUR","TKM","TUV","UGA","UKR","ARE",
    "GBR","USA","URY","UZB","VUT","VEN","VNM","YEM","ZMB","ZWE","UNO","XXA","XXB",
    "XXC","XXX","XOM","EUE"
}


def _parse_date_mrz(ymd: str) -> datetime | None:
    """Parse MRZ date format YYMMDD."""
    try:
        return datetime.strptime(ymd.strip(), "%y%m%d")
    except Exception:
        return None


def _parse_date_visual(raw: str) -> datetime | None:
    """
    Parse date formats visuels :
    DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY, YYYY-MM-DD, YYYYMMDD.
    """
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    # Format YYYYMMDD brut
    if re.fullmatch(r"\d{8}", raw):
        try:
            return datetime.strptime(raw, "%Y%m%d")
        except ValueError:
            pass
    # Format YYMMDD (MRZ dans champ visuel)
    if re.fullmatch(r"\d{6}", raw):
        return _parse_date_mrz(raw)
    return None


# ─────────────────────────────────────────────
# CAS 1 — MRZ trouvée : vérification complète
# ─────────────────────────────────────────────

def _analyze_with_mrz(fields: dict, mrz_type: str) -> dict:
    checks = {}
    details = {}
    now = datetime.now()

    # 1. MRZ présente
    checks["mrz_detected"] = True
    details["mrz_detected"] = f"MRZ type : {mrz_type}"

    # 2. Pays valide
    country = fields.get("country", "").strip().upper()
    checks["valid_country"] = country in VALID_COUNTRIES
    details["valid_country"] = (
        f"Pays : {country}" if checks["valid_country"]
        else f"Code pays invalide : '{country}'"
    )

    # 3. Nationalité valide
    nationality = fields.get("nationality", "").strip().upper()
    checks["valid_nationality"] = nationality in VALID_COUNTRIES or nationality == "<<"
    details["valid_nationality"] = (
        f"Nationalité : {nationality}" if checks["valid_nationality"]
        else f"Nationalité invalide : '{nationality}'"
    )

    # 4. Date de naissance
    dob_str = fields.get("date_of_birth", "").strip()
    dob = _parse_date_mrz(dob_str)
    if dob:
        age = (now - dob).days / 365.25
        checks["valid_dob"] = 0 < age < 120
        details["valid_dob"] = (
            f"Naissance : {dob_str} (~{int(age)} ans)"
            if checks["valid_dob"]
            else f"Date naissance incohérente : {dob_str}"
        )
    else:
        checks["valid_dob"] = False
        details["valid_dob"] = f"Date naissance illisible : '{dob_str}'"

    # 5. Date expiration
    exp_str = fields.get("expiration_date", "").strip()
    exp = _parse_date_mrz(exp_str)
    if exp:
        days_diff = (exp - now).days
        checks["valid_expiry"] = days_diff > -1825
        details["valid_expiry"] = (
            f"Expiration : {exp_str} ({'valide' if days_diff >= 0 else f'expiré depuis {abs(days_diff)}j'})"
        )
    else:
        checks["valid_expiry"] = False
        details["valid_expiry"] = f"Date expiration illisible : '{exp_str}'"

    # 6. Sexe
    sex = fields.get("sex", "").strip().upper()
    checks["valid_sex"] = sex in ("M", "F", "<", "")
    details["valid_sex"] = f"Sexe : {sex}" if checks["valid_sex"] else f"Sexe invalide : '{sex}'"

    # 7. N° document
    number = fields.get("number", "").strip()
    checks["valid_doc_number"] = bool(number) and number.replace("<", "") != ""
    details["valid_doc_number"] = (
        f"N° document : {number}" if checks["valid_doc_number"]
        else "Numéro document vide ou illisible"
    )

    # 8. Nom
    surname = fields.get("surname", "").strip()
    checks["valid_surname"] = bool(surname) and surname.replace("<", "").replace(" ", "") != ""
    details["valid_surname"] = (
        f"Nom : {surname}" if checks["valid_surname"]
        else "Nom vide ou illisible"
    )

    # 9. DOB < expiry
    if dob and exp:
        checks["dob_before_expiry"] = dob < exp
        details["dob_before_expiry"] = (
            "Naissance avant expiration : OK"
            if checks["dob_before_expiry"]
            else "Erreur : naissance après expiration"
        )
    else:
        checks["dob_before_expiry"] = False
        details["dob_before_expiry"] = "Impossible de comparer les dates"

    passed = sum(1 for v in checks.values() if v)
    score  = round(passed / len(checks), 4)

    return {
        "score":   score,
        "checks":  checks,
        "details": details,
        "error":   None,
        "mode":    "mrz",
    }


# ─────────────────────────────────────────────
# CAS 2 — Pas de MRZ : vérification visuelle
# ─────────────────────────────────────────────

def _analyze_without_mrz(fields: dict) -> dict:
    """
    Vérifie la cohérence logique des champs extraits visuellement.
    Checks allégés et pondération différente car extraction moins fiable.
    """
    checks = {}
    details = {}
    now = datetime.now()

    # 1. Pas de MRZ — on le note mais ce n'est pas un échec
    checks["mrz_detected"] = False
    details["mrz_detected"] = "Aucune MRZ — vérification visuelle des champs"

    # 2. Nom présent
    surname = fields.get("surname", "").strip()
    checks["name_found"] = bool(surname) and len(surname) >= 2
    details["name_found"] = (
        f"Nom extrait : {surname}" if checks["name_found"]
        else "Nom non extrait"
    )

    # 3. Date de naissance cohérente
    dob_raw = fields.get("date_of_birth", "")
    dob = _parse_date_visual(dob_raw)
    if dob:
        age = (now - dob).days / 365.25
        checks["valid_dob"] = 0 < age < 120
        details["valid_dob"] = (
            f"Naissance : {dob_raw} (~{int(age)} ans)"
            if checks["valid_dob"]
            else f"Date naissance incohérente : {dob_raw}"
        )
    else:
        checks["valid_dob"] = False
        details["valid_dob"] = (
            f"Date naissance non extraite ou illisible : '{dob_raw}'"
        )

    # 4. Date expiration cohérente
    exp_raw = fields.get("expiration_date", "")
    exp = _parse_date_visual(exp_raw)
    if exp:
        days_diff = (exp - now).days
        checks["valid_expiry"] = days_diff > -1825
        details["valid_expiry"] = (
            f"Expiration : {exp_raw} ({'valide' if days_diff >= 0 else f'expiré depuis {abs(days_diff)}j'})"
        )
    else:
        checks["valid_expiry"] = False
        details["valid_expiry"] = f"Date expiration non extraite : '{exp_raw}'"

    # 5. Numéro document
    number = fields.get("number", "").strip()
    checks["doc_number_found"] = bool(number) and len(number) >= 4
    details["doc_number_found"] = (
        f"N° document : {number}" if checks["doc_number_found"]
        else "Numéro document non extrait"
    )

    # 6. Sexe cohérent (si présent)
    sex = fields.get("sex", "").strip().upper()
    if sex:
        checks["valid_sex"] = sex in ("M", "F")
        details["valid_sex"] = (
            f"Sexe : {sex}" if checks["valid_sex"]
            else f"Sexe invalide : '{sex}'"
        )
    # Si absent : on ne pénalise pas (champ souvent absent sur CIN)

    # 7. Cohérence DOB < expiry
    if dob and exp:
        checks["dob_before_expiry"] = dob < exp
        details["dob_before_expiry"] = (
            "Naissance avant expiration : OK"
            if checks["dob_before_expiry"]
            else "Erreur : naissance après expiration"
        )

    # Score : les checks manqués à cause de l'absence d'extraction
    # ne pénalisent qu'à 50% (extraction visuelle moins fiable que MRZ)
    WEIGHTS = {
        "mrz_detected":    0.0,   # neutre, informatif seulement
        "name_found":      0.25,
        "valid_dob":       0.25,
        "valid_expiry":    0.20,
        "doc_number_found":0.15,
        "valid_sex":       0.10,
        "dob_before_expiry":0.05,
    }

    weighted_score = 0.0
    total_weight   = 0.0
    for key, w in WEIGHTS.items():
        if key in checks and w > 0:
            total_weight   += w
            weighted_score += w * (1.0 if checks[key] else 0.0)

    score = round(weighted_score / total_weight, 4) if total_weight > 0 else 0.5

    # On plafonne à 0.80 max (incertitude inhérente au mode visuel)
    score = round(min(score, 0.80), 4)

    return {
        "score":   score,
        "checks":  checks,
        "details": details,
        "error":   None,
        "mode":    "visual",   # ← flag frontend
        "note":    "Vérification visuelle — MRZ absente (CIN sans MRZ ou document non reconnu)",
    }


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

def analyze(ocr_result: dict) -> dict:
    """
    Aiguilleur principal.
    Si mrz_found=True  → _analyze_with_mrz()
    Si mrz_found=False → _analyze_without_mrz()
    """
    mrz_found = ocr_result.get("mrz_found", True)   # rétrocompatible
    fields    = ocr_result.get("fields", {})
    mrz_type  = ocr_result.get("mrz_type", "")

    if mrz_found and mrz_type not in ("", "None", "Unknown"):
        return _analyze_with_mrz(fields, mrz_type)
    else:
        return _analyze_without_mrz(fields)