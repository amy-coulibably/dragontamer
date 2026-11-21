#!/usr/bin/env python3
"""
Transcription diarisée d'appels stéréo 8 kHz via STTaaS (Whisper large-v3, batch).

Canal gauche = client, canal droit = conseiller. Chaque canal est transcrit
séparément (pas de diarisation acoustique), puis fusionné par horodatage des mots.

Configuration dans un fichier .env (voir .env.example) :
    STTAAS_URL=https://<hôte-du-service>          # sans /batch/...
    STTAAS_API_KEY=...
    STTAAS_CA_BUNDLE=/chemin/vers/ca-bundle.pem   # CA d'entreprise (PEM)

Usage :
    python sttaas_stereo.py appel.mp3 --out sortie/     # un appel
    python sttaas_stereo.py audios/   --out sortie/     # tous les appels d'un dossier

En fin de run, un rapport HTML est toujours généré (rapport.html pour un dossier,
<nom>.rapport.html pour un appel seul) : transcription cliquable pour écouter chaque
segment, et analyse WER/CER pour les appels dont l'annotation <nom>.json se trouve à
côté de l'audio. Sans annotation, l'appel figure dans le rapport sans évaluation.

Sorties par appel, dans --out :
    <nom>.transcription.json   {"id", "audio", "transcription", "diarization": [
                                 {"start", "end", "speaker": "Client"|"Conseiller", "text"}]}
    <nom>.txt                  dialogue lisible
    <nom>.detail.json          mots horodatés, paramètres, passages filtrés
    raw/                       réponses brutes du service (non récupérables deux fois)

Dépendances (aucun binaire externe, pas de ffmpeg) :
    pip install requests soundfile soxr numpy rapidfuzz num2words
"""
import argparse
import array
import base64
import json
import math
import os
import re
import sys
import time
import unicodedata
import wave
from pathlib import Path

ROLES = {"L": "client", "R": "conseiller"}
LABELS = {"client": "Client", "conseiller": "Conseiller"}
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg"}

# Hallucinations Whisper typiques sur silence (français). À enrichir avec vos constats.
HALLU = re.compile(
    r"sous[- ]titres?|amara\.org|merci d'avoir regard|abonnez[- ]vous|"
    r"soustitreur|radio[- ]canada|❤|♪",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- config
def load_env(path: Path) -> dict:
    """Lit un .env (CLE=valeur, guillemets et # commentaires acceptés).
    Comme python-dotenv, une variable déjà définie dans l'environnement prime."""
    if not path.is_file():
        sys.exit(f"Fichier .env introuvable : {path.resolve()}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        val = val.strip()
        if val[:1] in "\"'" and val[-1:] == val[:1] and len(val) >= 2:
            val = val[1:-1]
        else:
            val = val.split(" #", 1)[0].strip()
        os.environ.setdefault(key, val)

    cfg = {k: os.environ.get(k, "").strip()
           for k in ("STTAAS_URL", "STTAAS_API_KEY", "STTAAS_CA_BUNDLE")}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        sys.exit(f"Variable(s) manquante(s) dans {path} : {', '.join(missing)}")
    ca = Path(cfg["STTAAS_CA_BUNDLE"]).expanduser()
    if not ca.is_absolute():
        ca = path.parent / ca                 # chemin relatif = relatif au .env
    if not ca.is_file():
        sys.exit(f"Bundle CA introuvable : {ca}")
    cfg["STTAAS_CA_BUNDLE"] = str(ca.resolve())
    return cfg


# ----------------------------------------------------------------- audio
TARGET_SR = 16000


def _audio_libs():
    try:
        import numpy as np
        import soundfile as sf
        import soxr
    except ImportError as e:
        sys.exit(f"Dépendance manquante ({e.name}) : pip install soundfile soxr numpy")
    return np, sf, soxr


def duration_s(path: Path) -> float:
    _, sf, _ = _audio_libs()
    return sf.info(str(path)).duration


def split_channels(src: Path, workdir: Path) -> dict:
    """Stéréo -> 2 fichiers mono 16 kHz PCM 16 bits. Aucun débruitage.
    Lecture par libsndfile (WAV PCM / A-law / µ-law, FLAC, OGG, MP3), sans ffmpeg ;
    rééchantillonnage par soxr en qualité maximale (VHQ)."""
    np, sf, soxr = _audio_libs()
    try:
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    except sf.LibsndfileError as e:
        raise RuntimeError(f"Lecture audio impossible ({src}) : {e}") from None
    if data.shape[1] != 2:
        raise RuntimeError(
            f"{src.name} : {data.shape[1]} canal/canaux, 2 attendus "
            f"(gauche = client, droit = conseiller)")
    workdir.mkdir(parents=True, exist_ok=True)
    outs = {}
    for i, ch in enumerate("LR"):
        x = data[:, i]
        if sr != TARGET_SR:
            x = soxr.resample(x, sr, TARGET_SR, quality="VHQ")
        # libsndfile n'écrête pas par défaut : sans ceci, un dépassement du filtre
        # de rééchantillonnage se replierait en un claquement audible
        x = np.clip(x, -1.0, 32767 / 32768)
        outs[ch] = workdir / f"{src.stem}__{ROLES[ch]}.wav"
        sf.write(str(outs[ch]), x, TARGET_SR, subtype="PCM_16")
    return outs


def rms(path: Path, start: float, end: float) -> float:
    """Énergie RMS d'un intervalle (sert à arbitrer la diaphonie)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        w.setpos(max(0, min(w.getnframes(), int(start * sr))))
        frames = w.readframes(max(0, int((end - start) * sr)))
    s = array.array("h", frames)
    return math.sqrt(sum(x * x for x in s) / len(s)) if s else 0.0


# ------------------------------------------------------------------- API
class Sttaas:
    def __init__(self, base_url, api_key, ca_bundle, prefix="/batch/transcriptions"):
        import requests
        self.base = base_url.rstrip("/") + prefix
        self.s = requests.Session()
        self.s.verify = ca_bundle             # vérification TLS avec le CA fourni
        self.s.headers.update({"Authorization": f"Bearer {api_key}",
                               "Content-Type": "application/json"})
        self.ssl_error = requests.exceptions.SSLError
        self.ca = ca_bundle

    def _post(self, route, payload, tries=6):
        for i in range(tries):
            try:
                # verify passé à chaque requête : sinon une variable d'environnement
                # REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE prendrait le pas sur session.verify
                r = self.s.post(f"{self.base}/{route}", json=payload,
                                timeout=300, verify=self.ca)
            except self.ssl_error as e:
                raise RuntimeError(
                    f"Échec TLS avec le bundle {self.ca} : vérifiez qu'il contient "
                    f"la chaîne complète (racine + intermédiaires) au format PEM.\n{e}"
                ) from None
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 5 * 2 ** i))       # quota / incident : backoff
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{route} -> HTTP {r.status_code}: {r.text[:500]}")
            return r.json()
        raise RuntimeError(f"{route}: abandon après {tries} tentatives")

    def submit(self, files, params):
        payload = dict(params)
        payload["files"] = [
            {"filename": p.name,
             "content": base64.b64encode(p.read_bytes()).decode("ascii")}
            for p in files]
        return self._post("submit", payload)["job_id"]

    def wait(self, job_id, interval=5, timeout=4 * 3600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self._post("status", {"job_id": job_id})["status"]
            if st in ("done", "failed"):
                return st
            time.sleep(interval)
        raise TimeoutError(f"job {job_id} non terminé après {timeout}s")

    def result(self, job_id, raw_path: Path):
        """Le résultat n'est récupérable qu'UNE fois : écrit sur disque avant tout."""
        data = self._post("result", {"job_id": job_id}, tries=1)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data


# ---------------------------------------------------------------- fusion
def norm(txt: str) -> str:
    t = unicodedata.normalize("NFKD", txt.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", t).strip()


def words_of(result: dict) -> list:
    """Mots horodatés d'un canal. Les mots sans horodatage (fréquent pour les
    chiffres après alignement) héritent de la position du mot précédent."""
    out = []
    for seg in result.get("transcript", {}).get("segments", []):
        last = seg.get("start", 0.0)
        ws = seg.get("words") or [{"word": seg.get("text", ""),
                                   "start": seg.get("start"), "end": seg.get("end")}]
        for w in ws:
            txt = (w.get("word") or "").strip()
            if not txt:
                continue
            s = w.get("start", last)
            e = w.get("end", s)
            s = last if s is None else s
            e = s if e is None else max(e, s)
            out.append({"word": txt, "start": float(s), "end": float(e)})
            last = e
    return out


def utterances(words: list, speaker: str, max_gap: float) -> list:
    """Regroupe les mots d'un canal en énoncés, coupés sur les pauses."""
    utts, cur = [], []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > max_gap:
            utts.append(cur)
            cur = []
        cur.append(w)
    if cur:
        utts.append(cur)
    return [{"speaker": speaker, "start": u[0]["start"], "end": u[-1]["end"],
             "text": " ".join(w["word"] for w in u), "words": u} for u in utts]


def drop_hallucinations(utts: list, log: list) -> list:
    keep = []
    for u in utts:
        toks = norm(u["text"]).split()
        looped = len(toks) >= 8 and len(set(toks)) / len(toks) < 0.25
        if HALLU.search(u["text"]) or looped:
            log.append({"raison": "hallucination", **{k: u[k] for k in ("speaker", "start", "end", "text")}})
        else:
            keep.append(u)
    return keep


def drop_crosstalk(wa: list, wb: list, ra: str, rb: str, wavs: dict, log: list,
                   tol=0.4, min_run=3):
    """Diaphonie, au niveau du mot : une suite d'au moins `min_run` mots identiques
    et simultanés sur les deux canaux est une fuite ; on la retire du canal le plus
    faible. Un "oui" ou "d'accord" simultané isolé reste légitime."""
    na, nb = [norm(w["word"]) for w in wa], [norm(w["word"]) for w in wb]
    pairs, j0 = [], 0
    for i, w in enumerate(wa):
        while j0 < len(wb) and wb[j0]["start"] < w["start"] - tol:
            j0 += 1
        j = j0
        while j < len(wb) and wb[j]["start"] <= w["start"] + tol:
            if na[i] and na[i] == nb[j] and (not pairs or j > pairs[-1][1]):
                pairs.append((i, j))
                break
            j += 1
    runs, cur = [], []
    for p in pairs:
        if cur and p[0] == cur[-1][0] + 1 and p[1] == cur[-1][1] + 1:
            cur.append(p)
        else:
            if len(cur) >= min_run:
                runs.append(cur)
            cur = [p]
    if len(cur) >= min_run:
        runs.append(cur)
    da, db = set(), set()
    for run in runs:
        ia, ib = [p[0] for p in run], [p[1] for p in run]
        ea = rms(wavs[ra], wa[ia[0]]["start"], wa[ia[-1]]["end"])
        eb = rms(wavs[rb], wb[ib[0]]["start"], wb[ib[-1]]["end"])
        role, ws, idx, dead = (rb, wb, ib, db) if ea >= eb else (ra, wa, ia, da)
        dead.update(idx)
        log.append({"raison": "diaphonie", "speaker": role,
                    "start": ws[idx[0]]["start"], "end": ws[idx[-1]]["end"],
                    "text": " ".join(ws[k]["word"] for k in idx)})
    return ([w for i, w in enumerate(wa) if i not in da],
            [w for j, w in enumerate(wb) if j not in db])


def merge_turns(utts: list, same_speaker_gap: float) -> list:
    turns = []
    for u in sorted(utts, key=lambda x: (x["start"], x["end"])):
        if turns and turns[-1]["speaker"] == u["speaker"] \
                and u["start"] - turns[-1]["end"] <= same_speaker_gap:
            t = turns[-1]
            t["end"] = max(t["end"], u["end"])
            t["text"] += " " + u["text"]
            t["words"] += u["words"]
        else:
            turns.append({**u, "words": list(u["words"])})
    return turns


def build_dialogue(res_by_role: dict, wavs: dict, utt_gap=0.7, turn_gap=1.5):
    log = []
    (ra, res_a), (rb, res_b) = res_by_role.items()
    wa, wb = drop_crosstalk(words_of(res_a), words_of(res_b), ra, rb, wavs, log)
    utts = utterances(wa, ra, utt_gap) + utterances(wb, rb, utt_gap)
    utts = drop_hallucinations(utts, log)
    return merge_turns(utts, turn_gap), log


def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def clean_text(t: str) -> str:
    """Recolle les apostrophes détachées par l'alignement ("j 'ai" -> "j'ai")."""
    t = re.sub(r"\s+(['’])\s*(?=\w)", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def to_export(audio: Path, turns: list) -> dict:
    """Format de livraison : id / audio / transcription / diarization."""
    diar = [{"start": round(t["start"], 3), "end": round(t["end"], 3),
             "speaker": LABELS[t["speaker"]], "text": clean_text(t["text"])}
            for t in turns]
    return {"id": audio.stem, "audio": audio.name,
            "transcription": " ".join(d["text"] for d in diar),
            "diarization": diar}


def transcribe_file(audio: Path, out: Path, api, params: dict, swap=False) -> Path:
    """Transcrit un appel stéréo et écrit <stem>.transcription.json (+ .txt, .detail.json)."""
    wavs_ch = split_channels(audio, out / "wav")
    roles = {"L": "conseiller", "R": "client"} if swap else ROLES
    wavs = {roles[ch]: p for ch, p in wavs_ch.items()}
    files = list(wavs.values())
    # Doc : au-delà d'1 h, un seul fichier par requête.
    groups = [[f] for f in files] if duration_s(files[0]) > 3600 else [files]

    by_file = {}
    for n, grp in enumerate(groups):
        job = api.submit(grp, params)
        print(f"  job {job} soumis ({len(grp)} fichier(s))", file=sys.stderr)
        status = api.wait(job)
        data = api.result(job, out / "raw" / f"{audio.stem}__{n}_{job}.json")
        for r in data.get("results", []):
            if "error" in r or status == "failed":
                raise RuntimeError(f"{r.get('filename')}: {r.get('error', status)}")
            by_file[r["filename"]] = r
    missing = [p.name for p in files if p.name not in by_file]
    if missing:
        raise RuntimeError(f"résultat absent pour {', '.join(missing)}")

    turns, log = build_dialogue({role: by_file[p.name] for role, p in wavs.items()}, wavs)
    stem = out / audio.stem
    dump = lambda o: json.dumps(o, ensure_ascii=False, indent=2)
    final = Path(f"{stem}.transcription.json")
    final.write_text(dump(to_export(audio, turns)), encoding="utf-8")
    Path(f"{stem}.detail.json").write_text(dump(
        {"audio": audio.name, "params": params, "turns": turns, "filtres": log}),
        encoding="utf-8")
    Path(f"{stem}.txt").write_text("\n".join(
        f"[{fmt(t['start'])}] {LABELS[t['speaker']].upper()} : {clean_text(t['text'])}"
        for t in turns) + "\n", encoding="utf-8")
    print(f"  {len(turns)} tours, {len(log)} passage(s) filtré(s) -> {final.name}",
          file=sys.stderr)
    return final


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, help="un fichier audio, ou un dossier d'audios")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--env", type=Path, default=Path(".env"),
                    help="fichier .env (défaut : ./.env)")
    ap.add_argument("--prefix", default="/batch/transcriptions",
                    help="ou /v1/batch/transcriptions selon votre déploiement")
    ap.add_argument("--vad-onset", type=float, default=0.6)
    ap.add_argument("--min-duration-on", type=float, default=None,
                    help="envoyé seulement si précisé (ex. 0.1) : absent d'une des deux docs")
    ap.add_argument("--no-hotwords", action="store_true")
    ap.add_argument("--swap", action="store_true", help="si gauche = conseiller")
    ap.add_argument("--force", action="store_true",
                    help="retranscrit même si la sortie existe déjà")
    ap.add_argument("--no-report", action="store_true",
                    help="ne génère pas le rapport HTML en fin de run")
    ap.add_argument("--embed-audio", action="store_true",
                    help="intègre l'audio dans le rapport HTML (fichier autonome, mais "
                         "qui contient alors les enregistrements)")
    a = ap.parse_args()
    if not a.no_report:
        import sttaas_eval                    # avant le run : une dépendance manquante
                                              # doit échouer tout de suite, pas à la fin

    cfg = load_env(a.env)                     # avant tout traitement audio
    if a.audio.is_dir():
        audios = sorted(p for p in a.audio.iterdir() if p.suffix.lower() in AUDIO_EXT)
        if not audios:
            sys.exit(f"Aucun audio ({', '.join(sorted(AUDIO_EXT))}) dans {a.audio}")
    else:
        audios = [a.audio]
    (a.out / "raw").mkdir(parents=True, exist_ok=True)

    params = {"language": "fr", "detect_language": False,
              "enable_diarization": False,
              "enable_default_hotwords": not a.no_hotwords,
              "vad_onset": a.vad_onset}
    if a.min_duration_on is not None:
        params["min_duration_on"] = a.min_duration_on
    api = Sttaas(cfg["STTAAS_URL"], cfg["STTAAS_API_KEY"],
                 cfg["STTAAS_CA_BUNDLE"], a.prefix)

    failed = []
    for n, audio in enumerate(audios, 1):
        print(f"[{n}/{len(audios)}] {audio.name}", file=sys.stderr)
        if (a.out / f"{audio.stem}.transcription.json").exists() and not a.force:
            print("  déjà transcrit (--force pour refaire)", file=sys.stderr)
            continue
        try:
            transcribe_file(audio, a.out, api, params, a.swap)
        except (RuntimeError, TimeoutError) as e:
            if len(audios) == 1:
                raise
            failed.append(audio.name)
            print(f"  ÉCHEC : {e}", file=sys.stderr)
    if failed:
        print(f"{len(failed)} échec(s) : {', '.join(failed)}", file=sys.stderr)

    if not a.no_report:
        single = not a.audio.is_dir()
        sttaas_eval.run(a.audio.parent if single else a.audio, a.out, a.out,
                        embed_audio=a.embed_audio,
                        only=[a.audio.name] if single else None,
                        prefix=f"{a.audio.stem}." if single else "")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TimeoutError) as e:
        sys.exit(f"ERREUR : {e}")
