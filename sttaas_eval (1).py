#!/usr/bin/env python3
"""
Évaluation WER / CER des transcriptions diarisées + rapport HTML interactif.

Pour chaque audio abc.mp3 du dossier AUDIOS, le script attend :
  - AUDIOS/abc.json               annotation de référence (export Label Studio :
                                  régions avec label Client/Conseiller + texte)
  - HYP/abc.transcription.json    transcription produite par sttaas_stereo.py
                                  ({"id", "audio", "transcription", "diarization"})

Usage :
    python sttaas_eval.py AUDIOS --hyp sortie/ [--out sortie/] [--embed-audio]

Sorties, dans --out (défaut : le dossier --hyp) :
    rapport.html      rapport interactif (un clic sur un segment lance l'audio à cet endroit)
    evaluation.csv    une ligne par appel évalué (pour comparer plusieurs runs)
    evaluation.json   toutes les métriques

Un appel sans annotation figure quand même dans le rapport : sa transcription est
affichée et écoutable, sans analyse d'erreur.

Dépendances : pip install numpy rapidfuzz num2words
"""
import argparse
import base64
import bisect
import csv
import json
import os
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

try:
    import numpy as np
    from rapidfuzz.distance import Levenshtein
except ImportError as e:
    sys.exit(f"Dépendance manquante ({e.name}) : pip install numpy rapidfuzz")
try:
    from num2words import num2words
except ImportError:                      # l'évaluation reste possible, mais moins juste
    num2words = None

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg"}
MIME = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg"}
ROLES = ("Client", "Conseiller")
OK, SUB, DEL, INS = 0, 1, 2, 3
COLLAR = 5.0        # s : deux mots ne peuvent être appariés que s'ils sont proches dans le temps

# Hésitations ignorées par le WER « tolérant » (jamais par le WER principal)
FILLERS = {"euh", "euhm", "heu", "hum", "hm", "hmm", "mh", "mmh", "mm", "bah", "ben", "hein"}
# Variantes d'écriture ramenées à une seule forme
EQUIV = {"okay": "ok", "mme": "madame", "mr": "monsieur", "mlle": "mademoiselle"}


# ------------------------------------------------------- normalisation
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _spell(tok: str) -> str:
    """'42' -> 'quarante-deux' ; les zéros de tête sont dits un par un ('06' -> 'zéro six')."""
    rest = tok.lstrip("0")
    lead = len(tok) - len(rest)
    if not rest:
        return " ".join(["zéro"] * len(tok))
    return " ".join(["zéro"] * lead + [num2words(int(rest), lang="fr")])


def _hour(m) -> str:
    h = int(m.group(1))
    txt = _spell(str(h))
    if h % 10 == 1 and h != 11:                       # une heure, vingt et une heures
        txt = re.sub(r"un$", "une", txt)
    return f" {txt} heure{'s' if h > 1 else ''} {m.group(2) or ''} "


def _numbers(t: str) -> str:
    # 1 500 / 1.500 -> 1500 (groupes de 3 chiffres seulement : épargne les n° de téléphone)
    t = re.sub(r"(?<!\d)(\d{1,3})((?:[ .]\d{3})+)(?!\d)",
               lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), t)
    t = re.sub(r"\b1(er|re|ère)s?\b", lambda m: " premier " if m.group(1) == "er" else " première ", t)
    t = re.sub(r"\b(\d+)(?:e|eme|ème|nd|nde)s?\b",
               lambda m: f" {num2words(int(m.group(1)), lang='fr', to='ordinal')} ", t)
    t = re.sub(r"\b(\d{1,2}) ?h ?(\d{2})?\b", _hour, t)
    t = re.sub(r"(\d),(\d)", r"\1 virgule \2", t)
    return re.sub(r"\d+", lambda m: f" {_spell(m.group(0))} ", t)


def normalize(text: str, tolerant: bool = False) -> list:
    """Texte -> liste de mots comparables. Appliquée à l'identique aux deux côtés."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = t.replace("’", "'").replace("œ", "oe").replace("æ", "ae")
    t = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", t)         # balises d'annotation : [rire], <bruit>
    t = t.replace("€", " euros ").replace("%", " pour cent ").replace("&", " et ")
    if num2words is not None:
        t = _numbers(t)
    t = re.sub(r"[^\w\s]|_", " ", t)                  # ponctuation, apostrophes, traits d'union
    toks = [EQUIV.get(w, w) for w in t.split()]
    if tolerant:
        toks = [strip_accents(w) for w in toks]
        toks = [w for w in toks if w not in FILLERS]
    return toks


# ------------------------------------------------------------- lecture
def role_of(label) -> str | None:
    lab = strip_accents(str(label or "").lower())
    if "client" in lab or "customer" in lab:
        return "Client"
    if "conseil" in lab or "agent" in lab or "advisor" in lab:
        return "Conseiller"
    return None


def _as_text(v) -> str:
    return " ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v or "")


def load_reference(path: Path, audio_name: str):
    """Export Label Studio -> segments [{start, end, speaker, text}], avertissements."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = [t for t in (data if isinstance(data, list) else [data])
             if isinstance(t, dict) and "annotations" in t]
    if not tasks:
        raise ValueError("pas un export Label Studio (clé « annotations » absente)")
    task = next((t for t in tasks
                 if str((t.get("data") or {}).get("audio", "")).endswith(audio_name)), tasks[0])
    anns = [a for a in task["annotations"] if not a.get("was_cancelled")]
    if not anns:
        raise ValueError("aucune annotation validée")
    warn = []
    if len(anns) > 1:
        anns.sort(key=lambda a: str(a.get("updated_at") or a.get("created_at") or ""))
        warn.append(f"{len(anns)} annotations dans le fichier : la plus récente est utilisée.")
    regions, length = {}, None
    for it in anns[-1].get("result", []):
        v = it.get("value") or {}
        if "start" not in v or "end" not in v:
            continue
        length = it.get("original_length") or length
        reg = regions.setdefault(it.get("id") or (v["start"], v["end"]),
                                 {"start": float(v["start"]), "end": float(v["end"]),
                                  "label": None, "text": ""})
        if v.get("labels"):
            reg["label"] = v["labels"][0]
        if v.get("text"):
            reg["text"] = (reg["text"] + " " + _as_text(v["text"])).strip()
    segs, empty, unknown = [], 0, 0
    for reg in sorted(regions.values(), key=lambda r: (r["start"], r["end"])):
        if not reg["text"].strip():
            empty += 1
            continue
        role = role_of(reg["label"])
        if role is None:
            unknown += 1
            continue
        segs.append({"start": reg["start"], "end": reg["end"], "speaker": role, "text": reg["text"]})
    if empty:
        warn.append(f"{empty} région(s) annotée(s) sans texte, ignorée(s).")
    if unknown:
        warn.append(f"{unknown} région(s) sans label Client/Conseiller reconnu : exclue(s) du "
                    f"calcul, ce qui fausse le score à la baisse comme à la hausse.")
    return segs, warn, length


def load_hypothesis(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "diarization" not in data:
        raise ValueError("clé « diarization » absente")
    segs = []
    for d in data["diarization"]:
        role = role_of(d.get("speaker"))
        if role and str(d.get("text", "")).strip():
            segs.append({"start": float(d["start"]), "end": float(d["end"]),
                         "speaker": role, "text": d["text"]})
    return sorted(segs, key=lambda s: (s["start"], s["end"]))


# ---------------------------------------------------------- alignement
def tokens(segs, role, tolerant=False):
    """Mots normalisés d'un rôle : (mot, n° de segment, début, fin). Les annotations ne
    datent que les segments : chaque mot reçoit une position interpolée dans le sien,
    au prorata de sa longueur."""
    out = []
    for i, s in enumerate(segs):
        if s["speaker"] != role:
            continue
        ws = normalize(s["text"], tolerant)
        total, acc, dur = sum(map(len, ws)) or 1, 0, s["end"] - s["start"]
        for w in ws:
            t0 = s["start"] + dur * acc / total
            acc += len(w)
            out.append((w, i, t0, s["start"] + dur * acc / total))
    return out


def align(R, H, collar=COLLAR):
    """Alignement de Levenshtein au niveau du mot, contraint par le temps : un mot de la
    référence ne peut être apparié (identique ou remplacé) qu'à un mot transcrit situé à
    moins de `collar` secondes. Sans cette contrainte, un passage omis à 0:30 et un texte
    inventé à 4:00 se compenseraient en « remplacements » qui n'ont aucun sens.
    Même principe que le tcpWER de MeetEval, les rôles étant ici connus."""
    n, m = len(R), len(H)
    ids = {}
    ra = np.array([ids.setdefault(w[0], len(ids)) for w in R], dtype=np.int64)
    ha = np.array([ids.setdefault(w[0], len(ids)) for w in H], dtype=np.int64)
    r0 = np.array([w[2] for w in R]) - collar
    r1 = np.array([w[3] for w in R]) + collar
    h0, h1 = np.array([w[2] for w in H]), np.array([w[3] for w in H])
    big, cols = 1 << 28, np.arange(m + 1, dtype=np.int64)
    dist = np.empty((n + 1, m + 1), dtype=np.int32)
    dist[0] = cols
    for i in range(1, n + 1):
        near = (h1 >= r0[i - 1]) & (h0 <= r1[i - 1])
        diag = dist[i - 1, :-1] + np.where(near, ha != ra[i - 1], big)
        cand = np.empty(m + 1, dtype=np.int64)
        cand[0] = i
        cand[1:] = np.minimum(dist[i - 1, 1:] + 1, diag)
        dist[i] = np.minimum.accumulate(cand - cols) + cols      # insertions en chaîne
    items, i, j = [], n, m
    while i or j:
        if i and j:
            near = h1[j - 1] >= r0[i - 1] and h0[j - 1] <= r1[i - 1]
            same = ra[i - 1] == ha[j - 1]
            if near and dist[i, j] == dist[i - 1, j - 1] + (0 if same else 1):
                items.append({"t": OK if same else SUB, "r": R[i - 1][0], "h": H[j - 1][0],
                              "rs": R[i - 1][1], "hs": H[j - 1][1]})
                i, j = i - 1, j - 1
                continue
        if i and dist[i, j] == dist[i - 1, j] + 1:
            items.append({"t": DEL, "r": R[i - 1][0], "h": None, "rs": R[i - 1][1], "hs": None})
            i -= 1
        else:
            items.append({"t": INS, "r": None, "h": H[j - 1][0], "rs": None, "hs": H[j - 1][1]})
            j -= 1
    items.reverse()
    return items


def overlap(seg, others) -> float:
    """Part de la durée de `seg` recouverte par les segments `others`."""
    dur = max(seg["end"] - seg["start"], 1e-6)
    cov = sum(max(0.0, min(seg["end"], o["end"]) - max(seg["start"], o["start"])) for o in others)
    return min(1.0, cov / dur)


def rows_for_role(items, ref, hyp, role):
    """Projette l'alignement sur les segments : une ligne par segment transcrit, plus une
    ligne « non transcrit » par segment de référence dont aucun mot n'a été retrouvé."""
    matched = defaultdict(list)                       # segment réf -> positions appariées
    for p, it in enumerate(items):
        if it["t"] in (OK, SUB):
            matched[it["rs"]].append(p)
    by_hyp, orphan = defaultdict(list), defaultdict(list)
    for p, it in enumerate(items):
        if it["t"] != DEL:
            by_hyp[it["hs"]].append(it)
            continue
        m = matched.get(it["rs"])
        if not m:                                     # segment entièrement omis
            orphan[it["rs"]].append(it)
            continue
        k = bisect.bisect_left(m, p)                  # mot apparié le plus proche
        near = min(m[max(0, k - 1):k + 1], key=lambda q: abs(q - p))
        by_hyp[items[near]["hs"]].append(it)
    other_ref = [s for s in ref if s["speaker"] != role]
    rows = []
    for hs, its in by_hyp.items():
        seg = hyp[hs]
        flag = None
        n_ins = sum(it["t"] == INS for it in its)
        # aucun mot retrouvé et surtout des ajouts : segment sans équivalent dans la référence
        if n_ins and 2 * n_ins >= len(its) and all(it["t"] != OK for it in its):
            flag = "xtalk" if overlap(seg, other_ref) >= 0.5 else "extra"
        rows.append({"k": "hyp", "sp": role, "s": seg["start"], "e": seg["end"],
                     "raw": seg["text"], "flag": flag,
                     "it": [[it["t"], it["r"], it["h"]] for it in its],
                     "err": sum(it["t"] != OK for it in its)})
    for rs, its in orphan.items():
        seg = ref[rs]
        rows.append({"k": "miss", "sp": role, "s": seg["start"], "e": seg["end"],
                     "raw": seg["text"], "flag": None,
                     "it": [[DEL, it["r"], None] for it in its], "err": len(its)})
    return rows


def rate(num, den):
    return num / den if den else None


def evaluate_file(ref, hyp, collar=COLLAR):
    """-> métriques par rôle et globales, lignes d'affichage, compteurs d'erreurs."""
    out = {"roles": {}, "rows": [], "conf": Counter(), "dels": Counter(), "ins": Counter(),
           "miss_dur": [], "miss_words": 0, "warn": []}
    tot = Counter()
    for role in ROLES:
        R, H = tokens(ref, role), tokens(hyp, role)
        items = align(R, H, collar)
        c = Counter(it["t"] for it in items)
        rw, hw = [w[0] for w in R], [w[0] for w in H]
        rs, hs = " ".join(rw), " ".join(hw)
        m = {"n": len(R), "hyp": len(H), "s": c[SUB], "d": c[DEL], "i": c[INS],
             "cd": Levenshtein.distance(rs, hs), "cn": len(rs),
             "fd": Levenshtein.distance(rw, hw)}      # sans contrainte de temps
        rt = tokens(ref, role, True)
        m["td"] = sum(it["t"] != OK for it in align(rt, tokens(hyp, role, True), collar))
        m["tn"] = len(rt)
        out["roles"][role] = m
        tot.update(m)
        for it in items:
            if it["t"] == SUB:
                out["conf"][(it["r"], it["h"])] += 1
            elif it["t"] == DEL:
                out["dels"][it["r"]] += 1
            elif it["t"] == INS:
                out["ins"][it["h"]] += 1
        rows = rows_for_role(items, ref, hyp, role)
        for r in rows:
            if r["k"] == "miss":
                out["miss_dur"].append(r["e"] - r["s"])
                out["miss_words"] += r["err"]
        out["rows"] += rows
    out["rows"].sort(key=lambda r: (r["s"], r["e"]))
    out["tot"] = dict(tot)

    # Canaux inversés ? On rescore en croisant les rôles.
    words = {(side, role): [w[0] for w in tokens(segs, role)]
             for side, segs in (("r", ref), ("h", hyp)) for role in ROLES}
    straight = tot["s"] + tot["d"] + tot["i"]
    crossed = (Levenshtein.distance(words["r", "Client"], words["h", "Conseiller"])
               + Levenshtein.distance(words["r", "Conseiller"], words["h", "Client"]))
    if tot["n"] and straight / tot["n"] > 0.5 and crossed < 0.6 * straight:
        out["warn"].append("Les rôles semblent inversés : le score est bien meilleur en "
                           "échangeant Client et Conseiller. Relancez la transcription avec --swap.")
    return out


def finish(m: dict) -> dict:
    """Ajoute les taux à un compteur {n, s, d, i, cd, cn, td, tn, fd}."""
    m = dict(m)
    m["wer"] = rate(m["s"] + m["d"] + m["i"], m["n"])
    m["cer"] = rate(m["cd"], m["cn"])
    m["tol"] = rate(m["td"], m["tn"])
    m["free"] = rate(m["fd"], m["n"])
    return m


# ---------------------------------------------------------- diagnostics
def pct(x, digits=1) -> str:
    return "n/d" if x is None else f"{x * 100:.{digits}f}".replace(".", ",") + " %"


def s_(n) -> str:
    return "s" if n > 1 else ""


def bootstrap_ci(files, n_boot=2000):
    """IC 95 % du WER global par rééchantillonnage des appels."""
    if len(files) < 5:
        return None
    rng, vals = random.Random(0), []
    pairs = [(f["m"]["s"] + f["m"]["d"] + f["m"]["i"], f["m"]["n"]) for f in files]
    for _ in range(n_boot):
        samp = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        den = sum(n for _, n in samp)
        if den:
            vals.append(sum(e for e, _ in samp) / den)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def diagnostics(tot, files, miss_dur, miss_words):
    d, add = [], lambda k, t, x: d.append({"k": k, "t": t, "x": x})
    err = tot["s"] + tot["d"] + tot["i"]
    if not tot["n"]:
        return d
    if num2words is None:
        add("warn", "Nombres non normalisés",
            "Le module num2words est absent : « 200 » et « deux cents » comptent comme une "
            "erreur. Installez-le (pip install num2words) avant d'interpréter les scores.")
    if err:
        ps, pd, pi = tot["s"] / err, tot["d"] / err, tot["i"] / err
        if pd >= 0.45:
            add("warn", f"Les omissions dominent ({pct(pd, 0)} des erreurs)",
                "Des mots prononcés n'arrivent pas dans la transcription. Cause la plus "
                "fréquente : la détection de parole écarte des passages avant Whisper. "
                "Essayez un vad_onset plus bas, et min_duration_on vers 0,1 si votre service "
                "l'accepte ; vérifiez aussi le niveau sonore du canal concerné.")
        elif pi >= 0.35:
            add("warn", f"Les ajouts sont nombreux ({pct(pi, 0)} des erreurs)",
                "La transcription contient des mots absents de la référence. Trois causes à "
                "départager à l'écoute : texte inventé sur un silence, voix de l'autre canal "
                "captée par fuite, ou parole réelle que l'annotateur n'a pas saisie.")
        elif ps >= 0.55:
            add("info", f"Les remplacements dominent ({pct(ps, 0)} des erreurs)",
                "La parole est bien détectée mais mal reconnue : qualité acoustique du 8 kHz, "
                "noms propres et vocabulaire métier. Regardez la liste des confusions "
                "fréquentes : celles qui se répètent se corrigent par dictionnaire en aval.")
    if miss_dur:
        short = sum(x < 1.5 for x in miss_dur)
        add("warn" if tot["d"] and miss_words / tot["d"] >= 0.3 else "info",
            f"{len(miss_dur)} segment{s_(len(miss_dur))} de la référence entièrement non "
            f"transcrit{s_(len(miss_dur))}",
            f"Cela représente {pct(rate(miss_words, tot['d']), 0)} des mots omis ; durée médiane "
            f"{statistics.median(miss_dur):.1f} s".replace(".", ",") +
            f", dont {short} de moins de 1,5 s. "
            "Les segments très courts (« oui », « d'accord ») perdus signalent un seuil de "
            "détection de parole trop strict : ce sont souvent des accords du client.")
    extra = sum(r["flag"] == "extra" for f in files for r in f["rows"])
    xtalk = sum(r["flag"] == "xtalk" for f in files for r in f["rows"])
    if extra or xtalk:
        add("info", f"{extra + xtalk} segment{s_(extra + xtalk)} transcrit{s_(extra + xtalk)} "
                    "sans équivalent dans la référence",
            f"Dont {xtalk} pendant que l'autre interlocuteur parle (fuite entre canaux "
            f"probable) et {extra} pendant un silence de la référence (texte inventé, ou "
            "parole non annotée). Utilisez le filtre « Ajouts isolés » et écoutez-les.")
    rc, ra = tot["roles"]["Client"], tot["roles"]["Conseiller"]
    if rc["wer"] is not None and ra["wer"] is not None and abs(rc["wer"] - ra["wer"]) >= 0.05:
        worse = "Client" if rc["wer"] > ra["wer"] else "Conseiller"
        add("info", f"Le canal {worse} est nettement moins bien transcrit",
            f"WER Client {pct(rc['wer'])}, Conseiller {pct(ra['wer'])}. Côté client, c'est "
            "attendu (téléphone mobile, bruit, débit) ; côté conseiller, cherchez plutôt un "
            "problème de casque, de niveau ou de fuite entre canaux.")
    if tot["wer"] is not None and tot["tol"] is not None and tot["wer"] - tot["tol"] >= 0.03:
        add("info", "Une part du score tient aux conventions d'écriture",
            f"En ignorant accents et hésitations, le WER passe de {pct(tot['wer'])} à "
            f"{pct(tot['tol'])}. Cet écart ne mesure pas la reconnaissance : Whisper omet les "
            "« euh » par construction, et les annotateurs accentuent inégalement.")
    if tot["wer"] is not None and tot["wer"] - tot["free"] >= 0.02:
        add("warn", "Des mots corrects mais décalés dans le temps",
            f"Sans contrainte de temps, le WER tomberait à {pct(tot['free'])}. L'écart vient de "
            f"mots identiques situés à plus de {COLLAR:.0f} s de leur place dans l'annotation : "
            "vérifiez que l'audio annoté est bien celui transcrit (même fichier, pas de "
            "décalage au début) et que les régions annotées sont correctement bornées.")
    if tot["n"] < 3000:
        add("info", "Échantillon réduit",
            f"{tot['n']} mots de référence sur {len(files)} appel{s_(len(files))} : un écart de un à deux "
            "points entre deux réglages n'est pas significatif à cette taille.")
    return d


# --------------------------------------------------------------- run
def audio_src(audio: Path, out: Path, embed: bool) -> str:
    if embed:
        mime = MIME.get(audio.suffix.lower(), "application/octet-stream")
        return f"data:{mime};base64," + base64.b64encode(audio.read_bytes()).decode("ascii")
    rel = os.path.relpath(audio.resolve(), out.resolve())
    return quote(Path(rel).as_posix())


def plain_rows(hyp):
    """Lignes d'affichage d'un appel sans référence : le texte tel que transcrit."""
    return [{"k": "hyp", "sp": s["speaker"], "s": s["start"], "e": s["end"], "raw": s["text"],
             "flag": None, "err": 0, "it": [[OK, None, w] for w in s["text"].split()]}
            for s in hyp]


def run(audio_dir: Path, hyp_dir: Path, out: Path, embed_audio=False, only=None,
        collar=COLLAR, prefix=""):
    audios = sorted(p for p in Path(audio_dir).iterdir() if p.suffix.lower() in AUDIO_EXT)
    if only:
        audios = [p for p in audios if p.name in only]
    out.mkdir(parents=True, exist_ok=True)
    files, skipped = [], []
    conf, dels, ins = Counter(), Counter(), Counter()
    miss_dur, miss_words = [], 0
    for audio in audios:
        ann = audio.with_suffix(".json")
        hyp_path = Path(hyp_dir) / f"{audio.stem}.transcription.json"
        bad = (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError)
        try:
            if not hyp_path.exists():
                raise ValueError(f"transcription {hyp_path.name} introuvable")
            hyp = load_hypothesis(hyp_path)
        except bad as e:
            skipped.append(f"{audio.name} : {e}")
            print(f"  ignoré — {audio.name} : {e}", file=sys.stderr)
            continue
        ref, why = None, f"Pas d'annotation {ann.name} à côté de l'audio"
        if ann.exists():
            try:
                ref, warn, length = load_reference(ann, audio.name)
                if not ref:
                    raise ValueError("aucun segment exploitable")
            except bad as e:
                ref, why = None, f"Annotation {ann.name} inexploitable ({e})"
        if ref is None:                               # transcription seule, sans évaluation
            files.append({"id": audio.stem, "name": audio.name, "m": None,
                          "dur": max([s["end"] for s in hyp], default=0),
                          "src": audio_src(audio, out, embed_audio), "miss": 0, "extra": 0,
                          "warn": [why + " : la transcription est affichée sans analyse "
                                   "d'erreur."], "rows": plain_rows(hyp)})
            print(f"  {audio.name} : sans évaluation — {why}", file=sys.stderr)
            continue
        ev = evaluate_file(ref, hyp, collar)
        m = finish(ev["tot"])
        m["roles"] = {r: finish(v) for r, v in ev["roles"].items()}
        conf.update(ev["conf"]); dels.update(ev["dels"]); ins.update(ev["ins"])
        miss_dur += ev["miss_dur"]; miss_words += ev["miss_words"]
        dur = length or max([s["end"] for s in ref + hyp], default=0)
        files.append({"id": audio.stem, "name": audio.name, "dur": dur, "m": m,
                      "src": audio_src(audio, out, embed_audio),
                      "miss": len(ev["miss_dur"]),
                      "extra": sum(r["flag"] is not None for r in ev["rows"]),
                      "warn": warn + ev["warn"], "rows": ev["rows"]})
        print(f"  {audio.name} : WER {pct(m['wer'])}, CER {pct(m['cer'])}", file=sys.stderr)
    if not files:
        print("Aucun rapport : aucune transcription abc.transcription.json trouvée. "
              + " ; ".join(skipped), file=sys.stderr)
        return None

    scored, tot = [f for f in files if f["m"]], None
    if scored:
        tot = Counter()
        roles = {r: Counter() for r in ROLES}
        for f in scored:
            tot.update({k: v for k, v in f["m"].items() if isinstance(v, int)})
            for r in ROLES:
                roles[r].update({k: v for k, v in f["m"]["roles"][r].items()
                                 if isinstance(v, int)})
        tot = finish(tot)
        tot["roles"] = {r: finish(v) for r, v in roles.items()}
        tot["ci"] = bootstrap_ci(scored)

    params = None
    detail = Path(hyp_dir) / f"{files[0]['id']}.detail.json"
    if detail.exists():
        try:
            params = json.loads(detail.read_text(encoding="utf-8")).get("params")
        except (ValueError, OSError):
            pass
    data = {"gen": datetime.now().strftime("%d/%m/%Y %H:%M"), "params": params,
            "numbers": num2words is not None, "fillers": sorted(FILLERS), "collar": collar,
            "tot": tot, "skipped": skipped,
            "diag": diagnostics(tot, scored, miss_dur, miss_words) if tot else [],
            "top": {"sub": [[r, h, c] for (r, h), c in conf.most_common(15)],
                    "del": [[w, c] for w, c in dels.most_common(15)],
                    "ins": [[w, c] for w, c in ins.most_common(15)]},
            "files": files}

    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    report = out / f"{prefix}rapport.html"
    report.write_text(HTML.replace("__DATA__", blob), encoding="utf-8")
    if not scored:
        print(f"{len(files)} appel(s) sans annotation : rapport sans évaluation -> {report}")
        return data
    light = {k: v for k, v in data.items() if k != "files"}
    light["files"] = [{k: v for k, v in f.items() if k not in ("rows", "src")} for f in files]
    (out / f"{prefix}evaluation.json").write_text(json.dumps(light, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    with open(out / f"{prefix}evaluation.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["appel", "duree_s", "mots_ref", "mots_hyp", "remplaces", "omis", "ajoutes",
                    "wer", "cer", "wer_client", "wer_conseiller", "wer_tolerant",
                    "wer_sans_contrainte_temps", "segments_non_transcrits", "ajouts_isoles"])
        num = lambda x: "" if x is None else f"{x:.4f}".replace(".", ",")
        for f in scored:
            m = f["m"]
            w.writerow([f["name"], f"{f['dur']:.1f}".replace(".", ","), m["n"], m["hyp"],
                        m["s"], m["d"], m["i"], num(m["wer"]), num(m["cer"]),
                        num(m["roles"]["Client"]["wer"]), num(m["roles"]["Conseiller"]["wer"]),
                        num(m["tol"]), num(m["free"]), f["miss"], f["extra"]])
    print(f"{len(scored)} appel(s) évalué(s) sur {len(files)} — WER {pct(tot['wer'])}, "
          f"CER {pct(tot['cer'])} -> {report}")
    return data


# ---------------------------------------------------------------- HTML
HTML = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Évaluation de la transcription</title>
<style>
:root{
  --page:#f4f6f9; --surface:#ffffff; --ink:#131a24; --ink2:#465264; --muted:#6f7b8c;
  --rule:#dde2ea; --soft:#eef1f6; --accent:#17365d; --on-accent:#ffffff;
  --sub:#2a78d6; --del:#eb6834; --ins:#1baf7a; --ok:#c9d1dd;
  --sub-bg:#2a78d61a; --del-bg:#eb68341a; --ins-bg:#1baf7a1f; --play:#17365d12;
  --sans:"Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
  --serif:Charter,"Bitstream Charter","Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
}
@media (prefers-color-scheme:dark){:root{
  --page:#0e1218; --surface:#161c25; --ink:#edf0f5; --ink2:#b8c1cf; --muted:#8a95a6;
  --rule:#29313d; --soft:#1d2530; --accent:#9cc3f5; --on-accent:#0e1218;
  --sub:#3987e5; --del:#d95926; --ins:#199e70; --ok:#3a4453;
  --sub-bg:#3987e52b; --del-bg:#d959262b; --ins-bg:#199e702b; --play:#9cc3f51a;
}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 var(--sans);padding-bottom:104px}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
header.top{padding:40px 0 8px}
h1{font:600 30px/1.2 var(--serif);margin:0 0 6px;letter-spacing:-.01em}
h2{font:600 21px/1.3 var(--serif);margin:0 0 4px}
h3{font:600 15px/1.4 var(--sans);margin:0 0 8px}
p{margin:0 0 10px} .lead{color:var(--ink2);max-width:72ch}
.meta{color:var(--muted);font-size:13px;margin:0}
section{margin:36px 0}
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:24px}
.num{font-variant-numeric:tabular-nums}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

/* synthèse */
.verdict{font:400 26px/1.3 var(--serif);margin:0 0 18px;max-width:30ch}
.verdict b{font-weight:600}
.hero{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:36px;align-items:start}
.waffle{display:grid;grid-template-columns:repeat(20,1fr);gap:3px;max-width:440px}
.waffle i{aspect-ratio:1;border-radius:3px;background:var(--ok)}
.waffle i.s{background:var(--sub)} .waffle i.d{background:var(--del)}
.waffle.plus{margin-top:10px} .waffle.plus i{background:var(--ins)}
.key{list-style:none;padding:0;margin:14px 0 0;display:grid;gap:5px;font-size:13.5px;color:var(--ink2)}
.key li{display:flex;gap:8px;align-items:baseline}
.key i{flex:none;width:11px;height:11px;border-radius:3px;background:var(--ok);transform:translateY(1px)}
.key i.s{background:var(--sub)} .key i.d{background:var(--del)} .key i.i{background:var(--ins)}
.key b{color:var(--ink);font-weight:600}
dl.metrics{margin:0;display:grid;gap:0}
dl.metrics div{display:grid;grid-template-columns:118px 1fr;gap:16px;padding:11px 0;border-top:1px solid var(--rule)}
dl.metrics div:first-child{border-top:0;padding-top:0}
dl.metrics dt{font:600 22px/1.2 var(--sans)}
dl.metrics dd{margin:0;color:var(--ink2);font-size:13.5px}
dl.metrics dd b{display:block;color:var(--ink);font-size:14.5px;font-weight:600}

/* constats */
.diag{list-style:none;margin:14px 0 0;padding:0}
.diag li{padding:14px 0 14px 18px;border-top:1px solid var(--rule);border-left:3px solid var(--rule);margin-left:0;padding-left:16px}
.diag li.warn{border-left-color:var(--del)}
.diag b{display:block;margin-bottom:2px}
.diag span{color:var(--ink2);display:block;max-width:86ch}
.cols{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:28px;margin-top:18px}
.cols ol{list-style:none;margin:0;padding:0;font-size:14px}
.cols li{display:flex;justify-content:space-between;gap:10px;padding:4px 0;border-top:1px solid var(--rule)}
.cols li span:last-child{color:var(--muted)}
.cols .w{font-family:var(--serif);font-size:15px}
.cols .arrow{color:var(--muted);padding:0 4px}
.cols p{color:var(--muted);font-size:13px;min-height:3.1em}

/* tableau des appels */
.scroll{overflow-x:auto;margin-top:14px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 10px;text-align:right;border-top:1px solid var(--rule);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{border-top:0;color:var(--muted);font-weight:600;font-size:12.5px;vertical-align:bottom}
thead button{all:unset;cursor:pointer;color:inherit;font:inherit}
thead button:hover{color:var(--ink)} thead button[aria-sort]::after{content:" ▾"}
thead button[aria-sort="ascending"]::after{content:" ▴"}
tbody tr{cursor:pointer} tbody tr:hover{background:var(--soft)}
tbody tr.cur{background:var(--play)}
td.bar{min-width:170px}
td.bar div{display:flex;align-items:center;gap:8px;justify-content:flex-end}
td.bar i{display:block;height:8px;border-radius:0 4px 4px 0;background:var(--accent);min-width:2px}
td.bar span{width:58px}

/* lecture d'un appel */
.dhead{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;margin-bottom:6px}
select,button.btn{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--rule);border-radius:7px;padding:7px 11px}
button.btn{cursor:pointer} button.btn:hover{background:var(--soft)}
select{max-width:100%;min-width:200px}
.filters{display:flex;flex-wrap:wrap;gap:6px 8px;margin:14px 0;align-items:center}
.filters button{font:inherit;font-size:13.5px;color:var(--ink2);background:transparent;border:1px solid var(--rule);border-radius:999px;padding:5px 12px;cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.filters label{font-size:13.5px;color:var(--ink2);display:inline-flex;gap:6px;align-items:center;margin-left:8px}
[hidden]{display:none!important}
.legend{display:flex;flex-wrap:wrap;gap:6px 22px;padding:12px 16px;background:var(--soft);border-radius:8px;font-size:13.5px;color:var(--ink2);margin-bottom:8px;align-items:baseline}
.legend span.ex{font:17px/2.2 var(--serif);color:var(--ink);margin-right:6px}
.alert{padding:10px 14px;border-left:3px solid var(--del);background:var(--del-bg);border-radius:0 8px 8px 0;margin:10px 0;font-size:14px}
.rows{margin-top:6px}
.row{display:grid;grid-template-columns:70px 104px minmax(0,1fr) 62px;gap:14px;padding:10px 12px;border-top:1px solid var(--rule);border-left:3px solid transparent;cursor:pointer;align-items:baseline}
.row:hover{background:var(--soft)}
.row.on{background:var(--play);border-left-color:var(--accent)}
.row .t{color:var(--muted);font-size:13px;display:flex;gap:6px;align-items:center}
.row .t svg{width:11px;height:11px;fill:currentColor;flex:none}
.row.on .t{color:var(--accent)}
.chip{display:inline-block;font-size:12.5px;font-weight:600;padding:1px 9px;border-radius:999px;border:1.5px solid var(--ink2);color:var(--ink2);line-height:1.5}
.chip.cli{background:var(--ink2);color:var(--surface)}
.tag{display:block;font-size:12px;color:var(--muted);margin-top:4px;line-height:1.3}
.txt{font:17px/2.25 var(--serif);overflow-wrap:anywhere}
.row.miss .txt{color:var(--ink2)}
.raw{display:none;font:13.5px/1.5 var(--sans);color:var(--muted);margin-top:2px}
.showraw .raw{display:block}
.e{font-size:13px;color:var(--muted);text-align:right}
.e b{color:var(--ink);font-weight:600}
ruby.sub{text-decoration:underline solid var(--sub) 2px;text-underline-offset:4px;background:var(--sub-bg);border-radius:3px;padding:0 2px}
ruby.sub rt{font:600 11.5px/1 var(--sans);color:var(--ink2);padding-bottom:1px}
.del{text-decoration:line-through solid var(--del) 2px;color:var(--ink2);background:var(--del-bg);border-radius:3px;padding:0 2px}
.ins{text-decoration:underline wavy var(--ins) 1.5px;text-underline-offset:4px;background:var(--ins-bg);border-radius:3px;padding:0 2px}
.empty{padding:26px 0;color:var(--muted)}

details.method{margin-top:8px}
details.method summary{cursor:pointer;font:600 21px/1.3 var(--serif)}
details.method div{max-width:78ch;color:var(--ink2);margin-top:12px}
details.method li{margin-bottom:6px} code{font-size:13px;background:var(--soft);padding:1px 5px;border-radius:4px}

/* lecteur */
#player{position:fixed;left:0;right:0;bottom:0;background:var(--surface);border-top:1px solid var(--rule);box-shadow:0 -6px 24px #0b14241a;z-index:5}
#player .wrap{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;padding-top:10px;padding-bottom:10px}
#player audio{height:38px;flex:1 1 300px;min-width:220px;max-width:480px}
#now{flex:1 1 240px;font-size:13.5px;color:var(--ink2);min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#now b{color:var(--ink)}
#player label{font-size:13.5px;color:var(--ink2);display:inline-flex;gap:6px;align-items:center}
#aerr{flex-basis:100%;color:var(--ink);font-size:13px;display:none}

@media (max-width:820px){
  .hero,.cols{grid-template-columns:1fr}
  .row{grid-template-columns:62px minmax(0,1fr) 48px;gap:4px 10px}
  .row .who{grid-column:2} .row .txt{grid-column:1/-1;grid-row:2}
  .verdict{font-size:22px} .panel{padding:18px} .wrap{padding:0 14px}
  body{padding-bottom:150px}
}
</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <h1>Évaluation de la transcription</h1>
  <p class="meta" id="meta"></p>
</header>

<p class="alert" id="noref" hidden>Aucune annotation de référence n'a été trouvée : ce rapport présente la transcription seule, sans analyse d'erreur. Pour l'obtenir, placez l'annotation <code>nom.json</code> à côté de l'audio <code>nom.mp3</code> et relancez.</p>
<section class="panel" id="syn" aria-labelledby="h-syn">
  <h2 id="h-syn" hidden>Synthèse</h2>
  <div class="hero">
    <div>
      <p class="verdict" id="verdict"></p>
      <div class="waffle" id="waffle" role="img"></div>
      <div class="waffle plus" id="waffle2" role="img"></div>
      <ul class="key" id="key"></ul>
    </div>
    <dl class="metrics" id="metrics"></dl>
  </div>
</section>

<section aria-labelledby="h-diag" id="sdiag">
  <h2 id="h-diag">Ce que disent les erreurs</h2>
  <p class="lead">Constats calculés sur l'ensemble des appels. Ce sont des pistes à confirmer à l'écoute, pas des certitudes.</p>
  <ul class="diag" id="diag"></ul>
  <div class="cols">
    <div><h3>Confusions fréquentes</h3><p>Mot attendu, puis mot transcrit. Une confusion qui revient se corrige par dictionnaire.</p><ol id="top-sub"></ol></div>
    <div><h3>Mots le plus souvent omis</h3><p>Présents dans la référence, absents de la transcription.</p><ol id="top-del"></ol></div>
    <div><h3>Mots le plus souvent ajoutés</h3><p>Transcrits alors que la référence ne les contient pas.</p><ol id="top-ins"></ol></div>
  </div>
</section>

<section aria-labelledby="h-files">
  <h2 id="h-files">Appels du run</h2>
  <p class="lead">Cliquez sur un appel pour le lire. Les en-têtes trient le tableau.</p>
  <div class="scroll"><table id="files"><thead></thead><tbody></tbody></table></div>
  <p class="meta" id="skipped" style="margin-top:10px"></p>
</section>

<section aria-labelledby="h-detail" id="detail">
  <h2 id="h-detail">Lecture d'un appel</h2>
  <p class="lead" id="dlead">Chaque ligne est un segment de la transcription produite, annoté des écarts avec la référence. Cliquez sur une ligne pour écouter l'audio à cet instant. Le texte est affiché tel qu'il est comparé (minuscules, sans ponctuation, nombres en lettres) ; cochez « Texte brut » pour voir l'original.</p>
  <div class="dhead" style="margin-top:14px">
    <button class="btn" id="prev" aria-label="Appel précédent">‹</button>
    <select id="pick" aria-label="Appel affiché"></select>
    <button class="btn" id="next" aria-label="Appel suivant">›</button>
    <span class="meta num" id="fmeta"></span>
  </div>
  <div id="fwarn"></div>
  <div class="legend" id="legend">
    <span><span class="ex"><ruby class="sub">plafonnement<rt>plafond</rt></ruby></span>remplacé : le mot attendu est écrit au-dessus</span>
    <span><span class="ex"><span class="del">oui</span></span>omis : prononcé, absent de la transcription</span>
    <span><span class="ex"><span class="ins">merci</span></span>ajouté : transcrit, absent de la référence</span>
  </div>
  <div class="filters" id="filters">
    <button data-f="all" aria-pressed="true">Tout</button>
    <button data-f="err" aria-pressed="false">Avec erreurs</button>
    <button data-f="miss" aria-pressed="false">Non transcrits</button>
    <button data-f="extra" aria-pressed="false">Ajouts isolés</button>
    <label><input type="checkbox" id="r-cli" checked> Client</label>
    <label><input type="checkbox" id="r-con" checked> Conseiller</label>
    <label><input type="checkbox" id="raw"> Texte brut</label>
  </div>
  <div class="rows" id="rows"></div>
</section>

<section>
<details class="method" id="dmethod">
  <summary>Comment ces chiffres sont calculés</summary>
  <div id="method"></div>
</details>
</section>
</div>

<div id="player">
  <div class="wrap">
    <audio id="audio" controls preload="metadata"></audio>
    <div id="now">Cliquez sur un segment pour l'écouter.</div>
    <label><input type="checkbox" id="chain"> Enchaîner les segments</label>
    <div id="aerr"></div>
  </div>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = (x, d = 1) => x == null ? 'n/d' : (x * 100).toFixed(d).replace('.', ',') + ' %';
const clock = t => { t = Math.max(0, t); const m = Math.floor(t / 60), s = Math.floor(t % 60); return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0'); };
const plural = (n, one, many) => n + ' ' + (n > 1 ? many : one);
const T = D.tot, audio = $('audio');
let LEAD = '', cur = -1, filter = 'all', stopAt = null, onRow = -1, sortKey = null, sortAsc = false;

/* ---- synthèse ---- */
(function () {
  const p = D.params ? Object.entries(D.params).map(([k, v]) => k + '=' + v).join(', ') : null;
  const nref = D.files.filter(f => f.m).length;
  $('meta').textContent = plural(D.files.length, 'appel', 'appels') +
    (T ? ', dont ' + nref + ' avec annotation, ' + T.n.toLocaleString('fr-FR') + ' mots de référence' : '') +
    '. Rapport du ' + D.gen + (p ? '. Paramètres : ' + p : '') + '.';
  if (!T) { ['syn', 'sdiag', 'dmethod', 'legend'].forEach(id => $(id).hidden = true); $('noref').hidden = false; return; }
  const s = Math.round(T.s / T.n * 100), d = Math.round(T.d / T.n * 100), i = Math.round(T.i / T.n * 100);
  const ok = Math.max(0, 100 - s - d);
  $('verdict').innerHTML = 'Sur 100 mots prononcés, <b>' + ok + '</b> sont transcrits correctement.';
  let h = '';
  for (let k = 0; k < 100; k++) h += '<i class="' + (k < ok ? '' : k < ok + s ? 's' : 'd') + '"></i>';
  $('waffle').innerHTML = h;
  $('waffle').setAttribute('aria-label', 'Cent mots de référence : ' + ok + ' corrects, ' + s + ' remplacés, ' + d + ' omis');
  $('waffle2').innerHTML = '<i></i>'.repeat(Math.min(i, 100));
  $('waffle2').setAttribute('aria-label', i + ' mots ajoutés pour cent mots de référence');
  $('key').innerHTML =
    '<li><i></i><span><b>' + ok + '</b> corrects</span></li>' +
    '<li><i class="s"></i><span><b>' + s + '</b> remplacés par un autre mot</span></li>' +
    '<li><i class="d"></i><span><b>' + d + '</b> omis</span></li>' +
    '<li><i class="i"></i><span><b>' + i + '</b> ajoutés en plus, sans équivalent dans la référence</span></li>';
  const ci = T.ci ? 'Intervalle à 95 % : ' + pct(T.ci[0]) + ' à ' + pct(T.ci[1]) + ' (rééchantillonnage des appels).'
                  : 'Moins de 5 appels : pas d\'intervalle de confiance calculable.';
  const M = [
    [pct(T.wer), 'WER, taux d\'erreur sur les mots', 'Remplacés + omis + ajoutés, divisés par le nombre de mots de la référence. Calculé rôle par rôle : un mot attribué au mauvais interlocuteur compte comme une erreur. ' + ci],
    [pct(T.cer), 'CER, taux d\'erreur sur les caractères', 'Même calcul lettre par lettre. Un CER bien plus bas que le WER signale des erreurs légères : accords, terminaisons, mots proches.'],
    [pct(T.roles.Client.wer), 'WER du canal Client', T.roles.Client.n.toLocaleString('fr-FR') + ' mots de référence.'],
    [pct(T.roles.Conseiller.wer), 'WER du canal Conseiller', T.roles.Conseiller.n.toLocaleString('fr-FR') + ' mots de référence.'],
    [pct(T.tol), 'WER tolérant', 'Sans les accents ni les hésitations (euh, ben, hein…). L\'écart avec le WER mesure le poids des conventions d\'écriture.']
  ];
  $('metrics').innerHTML = M.map(m => '<div><dt class="num">' + m[0] + '</dt><dd><b>' + m[1] + '</b>' + m[2] + '</dd></div>').join('');
})();

/* ---- constats ---- */
if (T) $('diag').innerHTML = D.diag.length
  ? D.diag.map(d => '<li class="' + d.k + '"><b>' + esc(d.t) + '</b><span>' + esc(d.x) + '</span></li>').join('')
  : '<li><span>Aucun constat particulier.</span></li>';
const none = '<li><span>Aucun</span><span></span></li>';
$('top-sub').innerHTML = D.top.sub.map(r => '<li><span class="w">' + esc(r[0]) + '<span class="arrow">devient</span>' + esc(r[1]) + '</span><span class="num">' + r[2] + '</span></li>').join('') || none;
$('top-del').innerHTML = D.top.del.map(r => '<li><span class="w">' + esc(r[0]) + '</span><span class="num">' + r[1] + '</span></li>').join('') || none;
$('top-ins').innerHTML = D.top.ins.map(r => '<li><span class="w">' + esc(r[0]) + '</span><span class="num">' + r[1] + '</span></li>').join('') || none;

/* ---- tableau ---- */
const COLS = [
  ['Appel', f => f.name, true], ['Durée', f => f.dur], ['Mots réf.', f => f.m ? f.m.n : -1],
  ['WER', f => f.m?.wer ?? -1], ['CER', f => f.m?.cer ?? -1], ['Remplacés', f => f.m ? f.m.s : -1],
  ['Omis', f => f.m ? f.m.d : -1], ['Ajoutés', f => f.m ? f.m.i : -1], ['Segments non transcrits', f => f.m ? f.miss : -1]
];
function table() {
  $('files').querySelector('thead').innerHTML = '<tr>' + COLS.map((c, k) =>
    '<th><button data-k="' + k + '"' + (sortKey === k ? ' aria-sort="' + (sortAsc ? 'ascending' : 'descending') + '"' : '') + '>' + c[0] + '</button></th>').join('') + '</tr>';
  const order = D.files.map((f, i) => i);
  if (sortKey != null) {
    const g = COLS[sortKey][1];
    order.sort((a, b) => { const x = g(D.files[a]), y = g(D.files[b]); return (x < y ? -1 : x > y ? 1 : 0) * (sortAsc ? 1 : -1); });
  }
  const max = Math.max(...D.files.map(f => f.m?.wer || 0), 0.0001);
  $('files').querySelector('tbody').innerHTML = order.map(i => {
    const f = D.files[i], m = f.m;
    if (!m) return '<tr data-i="' + i + '"' + (i === cur ? ' class="cur"' : '') + '><td>' + esc(f.name) + '</td><td class="num">' + clock(f.dur) +
      '</td><td colspan="7" style="text-align:left;color:var(--muted)">sans annotation : transcription seule</td></tr>';
    return '<tr data-i="' + i + '"' + (i === cur ? ' class="cur"' : '') + '><td>' + esc(f.name) + '</td><td class="num">' + clock(f.dur) +
      '</td><td class="num">' + m.n + '</td><td class="bar num"><div><i style="width:' + Math.round((m.wer || 0) / max * 96) + 'px"></i><span>' + pct(m.wer) +
      '</span></div></td><td class="num">' + pct(m.cer) + '</td><td class="num">' + m.s + '</td><td class="num">' + m.d +
      '</td><td class="num">' + m.i + '</td><td class="num">' + f.miss + '</td></tr>';
  }).join('');
}
$('files').addEventListener('click', e => {
  const b = e.target.closest('thead button'), r = e.target.closest('tbody tr');
  if (b) { const k = +b.dataset.k; sortAsc = sortKey === k ? !sortAsc : !!COLS[k][2]; sortKey = k; table(); }
  else if (r) { show(+r.dataset.i); $('detail').scrollIntoView(); }
});
if (D.skipped.length) $('skipped').textContent = 'Non évalués : ' + D.skipped.join(' ; ') + '.';

/* ---- lecture d'un appel ---- */
const PLAY = '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M1 0l8 5-8 5z"/></svg>';
function word(it) {
  const t = it[0], r = it[1], h = it[2];
  if (t === 0) return esc(h);
  if (t === 1) return '<ruby class="sub" title="attendu : ' + esc(r) + ', transcrit : ' + esc(h) + '">' + esc(h) + '<rt>' + esc(r) + '</rt></ruby>';
  if (t === 2) return '<span class="del" title="omis">' + esc(r) + '</span>';
  return '<span class="ins" title="ajouté">' + esc(h) + '</span>';
}
function visible(r) {
  if (r.sp === 'Client' ? !$('r-cli').checked : !$('r-con').checked) return false;
  if (filter === 'err') return r.err > 0;
  if (filter === 'miss') return r.k === 'miss';
  if (filter === 'extra') return !!r.flag;
  return true;
}
function rows() {
  const f = D.files[cur];
  const h = f.rows.map((r, i) => {
    if (!visible(r)) return '';
    const tag = r.k === 'miss' ? 'Non transcrit' : r.flag === 'xtalk' ? 'Ajout isolé, pendant que l\'autre parle' : r.flag === 'extra' ? 'Ajout isolé' : '';
    return '<div class="row' + (r.k === 'miss' ? ' miss' : '') + (i === onRow ? ' on' : '') + '" data-i="' + i + '" tabindex="0" role="button" aria-label="Écouter à ' + clock(r.s) + '">' +
      '<div class="t num">' + PLAY + clock(r.s) + '</div>' +
      '<div class="who"><span class="chip ' + (r.sp === 'Client' ? 'cli' : 'con') + '">' + r.sp + '</span>' + (tag ? '<span class="tag">' + tag + '</span>' : '') + '</div>' +
      '<div class="txt">' + r.it.map(word).join(' ') + '<div class="raw">' + (r.k === 'miss' ? 'Référence : ' : 'Transcription brute : ') + esc(r.raw) + '</div></div>' +
      '<div class="e num">' + (r.err ? '<b>' + r.err + '</b> err.' : '') + '</div></div>';
  }).join('');
  $('rows').innerHTML = h || '<div class="empty">Aucun segment ne correspond à ce filtre.</div>';
}
function show(i) {
  if (i < 0 || i >= D.files.length) return;
  const f = D.files[i], changed = i !== cur;
  cur = i;
  if (changed) {
    audio.pause(); stopAt = null; onRow = -1;
    $('aerr').style.display = 'none';
    audio.src = f.src;
    $('now').textContent = 'Cliquez sur un segment pour l\'écouter.';
  }
  $('pick').value = i;
  $('fmeta').textContent = f.m ? 'WER ' + pct(f.m.wer) + ', CER ' + pct(f.m.cer) + ', ' + clock(f.dur) + ', ' +
    plural(f.miss, 'segment non transcrit', 'segments non transcrits') : clock(f.dur) + ', sans annotation';
  if (!LEAD) LEAD = $('dlead').textContent;
  $('dlead').textContent = f.m ? LEAD : 'Chaque ligne est un segment de la transcription produite. Cliquez sur une ligne pour écouter l\'audio à cet instant.';
  $('legend').hidden = !f.m;
  $('filters').querySelectorAll('button[data-f]').forEach(b => b.hidden = !f.m && b.dataset.f !== 'all');
  if (!f.m && filter !== 'all') { filter = 'all'; $('filters').querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x.dataset.f === 'all')); }
  $('fwarn').innerHTML = f.warn.map(w => '<div class="alert">' + esc(w) + '</div>').join('');
  $('prev').disabled = i === 0; $('next').disabled = i === D.files.length - 1;
  rows(); table();
}
$('pick').innerHTML = D.files.map((f, i) => '<option value="' + i + '">' + esc(f.name) + '</option>').join('');
$('pick').onchange = e => show(+e.target.value);
$('prev').onclick = () => show(cur - 1);
$('next').onclick = () => show(cur + 1);
$('filters').addEventListener('click', e => {
  const b = e.target.closest('button[data-f]'); if (!b) return;
  filter = b.dataset.f;
  $('filters').querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b));
  rows();
});
$('r-cli').onchange = $('r-con').onchange = rows;
$('raw').onchange = e => $('rows').classList.toggle('showraw', e.target.checked);

/* ---- audio ---- */
function mark(i) {
  if (i === onRow) return;
  onRow = i;
  $('rows').querySelectorAll('.row.on').forEach(x => x.classList.remove('on'));
  const el = $('rows').querySelector('.row[data-i="' + i + '"]');
  if (el) el.classList.add('on');
  const r = D.files[cur].rows[i];
  if (r) $('now').innerHTML = '<b>' + r.sp + '</b>, ' + clock(r.s) + ' à ' + clock(r.e) + (r.k === 'miss' ? ' (non transcrit)' : '');
}
function play(i) {
  const r = D.files[cur].rows[i]; if (!r) return;
  const go = () => {
    audio.currentTime = Math.max(0, r.s - 0.15);
    stopAt = $('chain').checked ? null : r.e + 0.25;
    mark(i);
    const p = audio.play(); if (p) p.catch(() => {});
  };
  if (audio.readyState >= 1) go(); else { audio.addEventListener('loadedmetadata', go, { once: true }); audio.load(); }
}
$('rows').addEventListener('click', e => {
  const el = e.target.closest('.row'); if (!el || String(getSelection()) !== '') return;
  play(+el.dataset.i);
});
$('rows').addEventListener('keydown', e => {
  const el = e.target.closest('.row');
  if (el && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); play(+el.dataset.i); }
});
audio.addEventListener('timeupdate', () => {
  const t = audio.currentTime;
  if (stopAt != null && t >= stopAt) { audio.pause(); stopAt = null; return; }
  if (stopAt == null && !audio.paused && cur >= 0) {
    const R = D.files[cur].rows; let best = -1;
    for (let i = 0; i < R.length; i++) if (R[i].s <= t + 0.05 && t < R[i].e + 0.05 && visible(R[i])) best = i;
    if (best >= 0) mark(best);
  }
});
audio.addEventListener('error', () => {
  if (cur < 0) return;
  const f = D.files[cur], el = $('aerr');
  el.style.display = 'block';
  el.textContent = f.src.startsWith('data:')
    ? 'Ce navigateur ne sait pas lire ce format audio (les WAV en loi A ou µ ne sont pas pris en charge : convertissez en MP3 ou en WAV PCM).'
    : 'Audio introuvable ou illisible : « ' + decodeURIComponent(f.src) + ' ». Le rapport pointe vers l\'audio par un chemin relatif : gardez les deux au même endroit, ou régénérez avec --embed-audio.';
});

/* ---- méthode ---- */
if (T) $('method').innerHTML =
  '<p><b>WER</b> = (remplacés + omis + ajoutés) ÷ nombre de mots de la référence. Il peut dépasser 100 % si la transcription ajoute beaucoup de texte. <b>CER</b> : même formule au niveau du caractère, espaces compris, sans contrainte de temps.</p>' +
  '<ol><li><b>Référence.</b> Chaque région annotée dans Label Studio fournit un rôle (Client ou Conseiller), un texte et des bornes temporelles.</li>' +
  '<li><b>Regroupement par rôle.</b> Tous les textes du Client sont mis bout à bout dans l\'ordre du temps, côté référence comme côté transcription ; de même pour le Conseiller. Les deux rôles sont alignés séparément puis les erreurs sont additionnées. Ce choix évite de pénaliser un découpage en segments différent de celui de l\'annotateur, et sanctionne en revanche toute parole attribuée au mauvais interlocuteur.</li>' +
  '<li><b>Normalisation</b>, identique des deux côtés : minuscules, ponctuation supprimée, apostrophes et traits d\'union traités comme des espaces (« j\'ai » compte pour deux mots), ' +
  (D.numbers ? 'nombres écrits en toutes lettres (« 200 » devient « deux cents »), ' : '<b>nombres non convertis (module num2words absent)</b>, ') +
  'balises entre crochets retirées. Les accents sont conservés.</li>' +
  '<li><b>Alignement contraint par le temps.</b> Distance d\'édition minimale entre les deux suites de mots (Levenshtein), avec une règle : deux mots ne peuvent être appariés que s\'ils sont à moins de ' + D.collar + '\u00a0s l\'un de l\'autre. Sans elle, un passage omis au début et un texte inventé à la fin se compenseraient en faux « remplacements ». C\'est le principe du tcpWER (MeetEval), les rôles étant ici connus. Le WER classique, sans cette règle, vaut ' + pct(T.free) + ' sur ce corpus ; il est toujours inférieur ou égal. Les annotations ne datant que les segments, la position de chaque mot y est interpolée.</li>' +
  '<li><b>WER tolérant.</b> Même calcul après suppression des accents et des hésitations : ' + D.fillers.map(esc).join(', ') + '.</li></ol>' +
  '<p><b>Limites à garder en tête.</b> Le score mesure l\'écart à l\'annotation, pas à la vérité : une référence incomplète produit de faux « ajouts », et deux annotateurs ne s\'accordent jamais parfaitement. Un segment marqué « non transcrit » est un segment de la référence dont aucun mot n\'a été retrouvé. Les nombres dits chiffre par chiffre (numéros de compte, de téléphone) restent une source d\'erreurs apparentes. La moyenne est pondérée par le nombre de mots : un appel long pèse plus qu\'un appel court.</p>';

show(0);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audios", type=Path, help="dossier des audios et de leurs annotations abc.json")
    ap.add_argument("--hyp", type=Path, required=True,
                    help="dossier des abc.transcription.json")
    ap.add_argument("--out", type=Path, default=None, help="défaut : le dossier --hyp")
    ap.add_argument("--collar", type=float, default=COLLAR,
                    help=f"tolérance temporelle de l'alignement, en secondes (défaut {COLLAR:.0f})")
    ap.add_argument("--embed-audio", action="store_true",
                    help="intègre l'audio dans le HTML (fichier autonome mais volumineux, "
                         "et qui contient alors les enregistrements)")
    a = ap.parse_args()
    if not a.audios.is_dir():
        sys.exit(f"Dossier introuvable : {a.audios}")
    run(a.audios, a.hyp, a.out or a.hyp, embed_audio=a.embed_audio, collar=a.collar)


if __name__ == "__main__":
    main()
