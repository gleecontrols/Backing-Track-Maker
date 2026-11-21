"""FastAPI app: upload -> analyse -> (correct chords) -> arrange -> render -> export."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import analysis
import arrange
import render

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
JOBS = DATA / "jobs"
LIB = DATA / "library"
for d in (JOBS, LIB):
    d.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD = 120 * 1024 * 1024
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".mp4", ".webm"}
JOB_TTL = 24 * 3600
ID_RE = re.compile(r"^[a-f0-9]{32}$")
CID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
FORMATS = {"mp3": ("final.mp3", "audio/mpeg"), "wav": ("final.wav", "audio/wav"), "mid": ("final.mid", "audio/midi")}

app = FastAPI(title="Piano Backing Track Generator")
_lock = threading.Lock()


# ---------------------------------------------------------------- helpers
def job_dir(job_id: str) -> Path:
    if not ID_RE.match(job_id):
        raise HTTPException(404, "Unknown job.")
    p = JOBS / job_id
    if not p.exists():
        raise HTTPException(404, "That job has expired or does not exist. Please analyse the song again.")
    return p


def load_meta(job_id: str) -> dict:
    return json.loads((job_dir(job_id) / "meta.json").read_text())


def save_meta(job_id: str, meta: dict) -> None:
    p = JOBS / job_id / "meta.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta))
    tmp.replace(p)


def cleanup_old_jobs() -> None:
    now = time.time()
    for p in JOBS.iterdir():
        try:
            if p.is_dir() and now - p.stat().st_mtime > JOB_TTL:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 _.-]", "", s or "").strip()
    return s[:60] or "track"


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DATA / "library.db")
    con.row_factory = sqlite3.Row
    con.execute(
        """CREATE TABLE IF NOT EXISTS tracks(
            id TEXT PRIMARY KEY, client TEXT, title TEXT, artist TEXT, key TEXT,
            bpm REAL, duration REAL, style TEXT, created TEXT)"""
    )
    return con


def check_cid(cid: str | None) -> str:
    if not cid or not CID_RE.match(cid):
        raise HTTPException(400, "Missing or invalid library id.")
    return cid


# ---------------------------------------------------------------- models
class ChordEdit(BaseModel):
    start: float
    end: float
    chord: str


class ChordsBody(BaseModel):
    chords: list[ChordEdit]


class FxBody(BaseModel):
    preset: str | None = None
    intensity: float | None = None
    decay: float | None = None
    mix: float | None = None
    volume: float | None = None
    width: float | None = None
    tone: float | None = None


class GenerateBody(BaseModel):
    style: str = "ambient"
    transpose: int = Field(0, ge=-12, le=12)
    bpm: float | None = Field(None, ge=40, le=220)
    sustain: float = Field(0.8, ge=0, le=1)
    fx: FxBody = FxBody()


# ---------------------------------------------------------------- analysis
@app.post("/api/analyse")
def analyse_song(
    title: str = Form(""),
    artist: str = Form(""),
    rights: bool = Form(False),
    file: UploadFile | None = File(None),
):
    title, artist = title.strip()[:120], artist.strip()[:120]
    if file is None or not file.filename:
        # Deliberately NOT inventing chords from a title. No licensed chord source is configured.
        return {
            "status": "needs_audio",
            "message": (
                "I can't work out the real chords from a title alone, and I won't guess them. "
                "Please upload a recording of the song you have permission to use."
            ),
        }
    if not rights:
        raise HTTPException(400, "Please confirm you have permission to use this recording.")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Try MP3, WAV, M4A, FLAC or OGG.")

    cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    jd = JOBS / job_id
    jd.mkdir(parents=True)
    src = jd / f"upload{ext}"
    size = 0
    with src.open("wb") as f:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                shutil.rmtree(jd, ignore_errors=True)
                raise HTTPException(413, "That file is too large (limit 120 MB).")
            f.write(chunk)
    try:
        result = analysis.analyse(src, jd)
    except ValueError as e:
        shutil.rmtree(jd, ignore_errors=True)
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(jd, ignore_errors=True)
        raise HTTPException(500, f"Analysis failed: {e}")
    finally:
        for junk in (src, jd / "source.wav"):
            junk.unlink(missing_ok=True)  # the user's recording is not kept after analysis

    meta = {"id": job_id, "title": title or "Untitled", "artist": artist or "Unknown artist",
            "created": datetime.now(timezone.utc).isoformat(), "analysis": result, "last": None}
    save_meta(job_id, meta)
    return {"status": "ok", "job_id": job_id, "title": meta["title"], "artist": meta["artist"], **result}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    m = load_meta(job_id)
    return {"job_id": job_id, "title": m["title"], "artist": m["artist"], **m["analysis"]}


@app.put("/api/jobs/{job_id}/chords")
def correct_chords(job_id: str, body: ChordsBody):
    meta = load_meta(job_id)
    a = meta["analysis"]
    old = {(c["start"], c["end"]): c for c in a["chords"]}
    new = []
    last_end = None
    for c in body.chords:
        if c.end <= c.start:
            raise HTTPException(422, "A chord ends before it starts.")
        if last_end is not None and c.start < last_end - 0.01:
            raise HTTPException(422, "Chords overlap.")
        try:
            root, q = analysis.parse_chord(c.chord)
        except ValueError as e:
            raise HTTPException(422, str(e))
        prev = old.get((round(c.start, 3), round(c.end, 3)))
        flats = a["key"]["flats"]
        label = analysis.chord_label(root, q, flats)
        entry = {
            "start": round(c.start, 3), "end": round(c.end, 3), "chord": label, "root": root, "quality": q,
            "confidence": prev["confidence"] if prev else 0.0,
            "energy": prev["energy"] if prev else 0.5,
            "beats": prev["beats"] if prev else 1,
            "source": prev["source"] if prev and prev["chord"] == label else "manual",
        }
        new.append(entry)
        last_end = c.end
    if not new:
        raise HTTPException(422, "There must be at least one chord.")
    a["chords"] = new
    meta["last"] = None  # arrangement is now stale
    save_meta(job_id, meta)
    return {"chords": new}


# ---------------------------------------------------------------- generation
def _dry_key(meta: dict, body: GenerateBody) -> str:
    a = meta["analysis"]
    blob = json.dumps({
        "c": [(c["start"], c["end"], c["chord"]) for c in a["chords"]],
        "s": [(s["start"], s["end"], s["kind"]) for s in a["sections"]],
        "bpm": a["bpm"]["value"], "style": body.style, "tr": body.transpose, "tb": body.bpm,
        "sus": round(body.sustain, 3), "sf": render.find_soundfont().name,
    }, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _finish(job_id: str, meta: dict, dry: np.ndarray, fx: render.FxParams, info: dict, body: GenerateBody) -> dict:
    jd = JOBS / job_id
    audio = render.apply_fx(dry, fx)
    render.write_wav(jd / "final.wav", audio)
    render.write_mp3(jd / "final.wav", jd / "final.mp3")
    report = render.audio_report(audio)
    a = meta["analysis"]
    names = arrange.transposed_chord_names(a["chords"], a["key"]["tonic"], a["key"]["mode"], body.transpose)
    flats = arrange.uses_flats(a["key"]["tonic"], a["key"]["mode"], body.transpose)
    tonic = (a["key"]["tonic"] + body.transpose) % 12
    out_key = f"{analysis.note_name(tonic, flats)} {a['key']['mode']}"
    version = int(time.time() * 1000)
    result = {
        "status": "ok", "job_id": job_id, "version": version,
        "key": out_key, "bpm": info["tempo"], "duration": report["duration"], "style": body.style,
        "style_name": arrange.STYLES[body.style], "sections": info["sections"],
        "chord_names": names, "report": report,
        "approximation_note": "This is a piano approximation of the original arrangement.",
        "fx": fx.__dict__,
    }
    meta["last"] = {"info": info, "result": result, "request": json.loads(body.model_dump_json())}
    save_meta(job_id, meta)
    return result


@app.post("/api/jobs/{job_id}/generate")
def generate(job_id: str, body: GenerateBody):
    meta = load_meta(job_id)
    if body.style not in arrange.STYLES:
        raise HTTPException(422, "Unknown arrangement style.")
    a = meta["analysis"]
    jd = JOBS / job_id
    try:
        pm, info = arrange.build_midi(
            a["chords"], a["sections"], a["bpm"]["value"], a["beats_per_bar"], a["first_beat"],
            arrange.ArrangeParams(style=body.style, transpose=body.transpose, target_bpm=body.bpm, sustain=body.sustain),
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    pm.write(str(jd / "final.mid"))

    key = _dry_key(meta, body)
    dry_path = jd / f"dry_{key}.npy"
    with _lock:  # the synth is not thread-safe
        if dry_path.exists():
            dry = np.load(dry_path)
        else:
            dry = render.render_dry(pm)
            for old in jd.glob("dry_*.npy"):
                old.unlink(missing_ok=True)
            np.save(dry_path, dry)
    fx = render.FxParams.from_dict(body.fx.model_dump())
    return _finish(job_id, meta, dry, fx, info, body)


@app.post("/api/jobs/{job_id}/mix")
def remix(job_id: str, fx_body: FxBody):
    """Change reverb / tone / volume / width without re-arranging or re-rendering the piano."""
    meta = load_meta(job_id)
    last = meta.get("last")
    jd = JOBS / job_id
    dry_files = list(jd.glob("dry_*.npy"))
    if not last or not dry_files:
        raise HTTPException(409, "Generate the piano track first.")
    body = GenerateBody(**last["request"])
    dry = np.load(dry_files[0])
    fx = render.FxParams.from_dict(fx_body.model_dump())
    body.fx = fx_body
    return _finish(job_id, meta, dry, fx, last["info"], body)


@app.get("/api/jobs/{job_id}/file/{fmt}")
def download(job_id: str, fmt: str, dl: int = 0, v: int = 0):
    meta = load_meta(job_id)
    if fmt not in FORMATS:
        raise HTTPException(404, "Unknown format.")
    fname, mime = FORMATS[fmt]
    p = JOBS / job_id / fname
    if not p.exists():
        raise HTTPException(404, "Generate the piano track first.")
    headers = {"Cache-Control": "no-store"}
    name = f"{safe_name(meta['title'])} - {safe_name(meta['artist'])} (piano).{fmt}"
    return FileResponse(p, media_type=mime, filename=name if dl else None, headers=headers)


# ---------------------------------------------------------------- library
@app.post("/api/library")
def save_to_library(payload: dict, x_client_id: str | None = Header(None)):
    cid = check_cid(x_client_id)
    job_id = str(payload.get("job_id", ""))
    meta = load_meta(job_id)
    last = meta.get("last")
    jd = JOBS / job_id
    if not last or not (jd / "final.mp3").exists():
        raise HTTPException(409, "Generate the piano track before saving it.")
    res = last["result"]
    tid = uuid.uuid4().hex
    td = LIB / tid
    td.mkdir(parents=True)
    for fname in ("final.mp3", "final.wav", "final.mid"):
        shutil.copy2(jd / fname, td / fname)
    with db() as con:
        con.execute(
            "INSERT INTO tracks VALUES(?,?,?,?,?,?,?,?,?)",
            (tid, cid, meta["title"], meta["artist"], res["key"], res["bpm"], res["duration"],
             res["style_name"], datetime.now(timezone.utc).isoformat()),
        )
    return {"status": "ok", "id": tid}


@app.get("/api/library")
def list_library(x_client_id: str | None = Header(None)):
    cid = check_cid(x_client_id)
    with db() as con:
        rows = con.execute("SELECT * FROM tracks WHERE client=? ORDER BY created DESC", (cid,)).fetchall()
    return {"tracks": [dict(r) for r in rows]}


@app.get("/api/library/{tid}/file/{fmt}")
def library_file(tid: str, fmt: str, cid: str, dl: int = 0):
    check_cid(cid)
    if not ID_RE.match(tid) or fmt not in FORMATS:
        raise HTTPException(404, "Not found.")
    with db() as con:
        row = con.execute("SELECT * FROM tracks WHERE id=? AND client=?", (tid, cid)).fetchone()
    if not row:
        raise HTTPException(404, "Not found.")
    fname, mime = FORMATS[fmt]
    name = f"{safe_name(row['title'])} - {safe_name(row['artist'])} (piano).{fmt}"
    return FileResponse(LIB / tid / fname, media_type=mime, filename=name if dl else None)


@app.delete("/api/library/{tid}")
def delete_track(tid: str, x_client_id: str | None = Header(None)):
    cid = check_cid(x_client_id)
    if not ID_RE.match(tid):
        raise HTTPException(404, "Not found.")
    with db() as con:
        cur = con.execute("DELETE FROM tracks WHERE id=? AND client=?", (tid, cid))
    if cur.rowcount:
        shutil.rmtree(LIB / tid, ignore_errors=True)
    return {"status": "ok"}


@app.get("/api/health")
def health():
    return {"ok": True, "soundfont": render.find_soundfont().name}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
