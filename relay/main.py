"""Oasis cloud relay: the phone's captures reach the Mac when they are not on
the same Wi-Fi, and the Mac's results reach the phone anywhere.

The relay speaks the same job API as a Mac station on the local network
(GET /info, POST /pair/account, /jobs ...), so the apps use one client for
both. It keeps no data and no secrets: every request carries the caller's
Supabase session token, which the relay forwards to Supabase, where row
policies limit each account to its own jobs and files. Big files never pass
through here: the relay hands out signed upload and download URLs and the
apps talk to Supabase Storage directly, in parts under the plan's 50 MB
object limit.

Run locally:
    SUPABASE_URL=... SUPABASE_KEY=<publishable key> uvicorn main:app --port 8000
"""

import json
import os
import secrets
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BUCKET = "oasis"
VERSION = 1
STATION = {"id": "cloud", "name": "Oasis cloud", "version": VERSION, "owner": None, "relay": True}
PART_BYTES = 40 * 1024 * 1024
STAGES = ["reconstruct", "densify", "shapes", "splat"]
SIGNED_SECONDS = 3600

app = FastAPI(title="Oasis relay", version=str(VERSION))


class Supabase:
    """Supabase's REST, Auth and Storage APIs, called as the signed-in user."""

    def __init__(self, token: str, client: httpx.AsyncClient | None = None):
        self.token = token
        self.client = client or app.state.client

    @property
    def headers(self) -> dict[str, str]:
        return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {self.token}"}

    async def user(self) -> dict[str, Any]:
        r = await self.client.get(f"{SUPABASE_URL}/auth/v1/user", headers=self.headers)
        if r.status_code != 200:
            raise HTTPException(401, "sign in again")
        return r.json()

    async def rest(self, method: str, table: str, params: dict[str, str] | None = None,
                   body: Any = None, prefer: str = "return=representation") -> Any:
        r = await self.client.request(method, f"{SUPABASE_URL}/rest/v1/{table}", params=params,
                                      headers={**self.headers, "Prefer": prefer, "Content-Type": "application/json"},
                                      content=json.dumps(body) if body is not None else None)
        if r.status_code == 401:
            raise HTTPException(401, "sign in again")
        if r.status_code >= 400:
            raise HTTPException(502, f"database: {r.text[:200]}")
        return r.json() if r.content else None

    async def signed_upload(self, path: str) -> str:
        r = await self.client.post(f"{SUPABASE_URL}/storage/v1/object/upload/sign/{BUCKET}/{path}",
                                   headers=self.headers, json={})
        if r.status_code >= 400:
            raise HTTPException(502, f"storage: {r.text[:200]}")
        return f"{SUPABASE_URL}/storage/v1{r.json()['url']}"

    async def signed_download(self, path: str) -> str:
        r = await self.client.post(f"{SUPABASE_URL}/storage/v1/object/sign/{BUCKET}/{path}",
                                   headers=self.headers, json={"expiresIn": SIGNED_SECONDS})
        if r.status_code >= 400:
            raise HTTPException(404 if r.status_code == 404 else 502, f"storage: {r.text[:200]}")
        return f"{SUPABASE_URL}/storage/v1{r.json()['signedURL']}"

    async def delete_objects(self, paths: list[str]) -> None:
        if not paths:
            return
        r = await self.client.request("DELETE", f"{SUPABASE_URL}/storage/v1/object/{BUCKET}",
                                      headers=self.headers, json={"prefixes": paths})
        if r.status_code >= 400:
            raise HTTPException(502, f"storage: {r.text[:200]}")

    async def list_objects(self, prefix: str) -> list[str]:
        r = await self.client.post(f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}", headers=self.headers,
                                   json={"prefix": prefix, "limit": 1000})
        if r.status_code >= 400:
            return []
        return [f"{prefix}/{o['name']}" for o in r.json() if o.get("name")]


@app.on_event("startup")
async def startup() -> None:
    if not hasattr(app.state, "client"):
        app.state.client = httpx.AsyncClient(timeout=30)


def bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "sign in first")
    return authorization[7:].strip()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_out(row: dict[str, Any]) -> dict[str, Any]:
    """A jobs row as the Job the apps decode (camelCase, like the Mac station)."""
    return {
        "id": row["id"], "name": row["name"], "createdAt": row["created_at"], "capturedAt": row["captured_at"],
        "status": row["status"], "stages": row.get("stages") or [], "stageIndex": row.get("stage_index") or 0,
        "message": row.get("message"), "filesExpected": row.get("files_expected") or [],
        "filesReceived": row.get("files_received") or [], "results": row.get("results") or [],
        "parts": row.get("parts") or {}, "station": row.get("station"), "cloud": True,
    }


def safe_name(name: str) -> bool:
    return bool(name) and "/" not in name and ".." not in name and not name.startswith(".")


async def load_job(db: Supabase, job_id: str) -> dict[str, Any]:
    rows = await db.rest("GET", "jobs", {"id": f"eq.{job_id}", "select": "*"})
    if not rows:
        raise HTTPException(404, "no such job")
    return rows[0]


# MARK: routes the phone and the Mac share

@app.get("/info")
async def info() -> dict[str, Any]:
    return STATION


@app.post("/pair/account")
async def pair_account(request: Request) -> dict[str, Any]:
    body = await request.json()
    token = body.get("accessToken", "")
    user = await Supabase(token).user()
    # The session token itself is the credential; the apps refresh it as usual.
    return {"token": token, "station": {**STATION, "owner": user.get("id")}}


@app.get("/jobs")
async def jobs(authorization: str | None = Header(default=None),
               status: str | None = Query(default=None)) -> list[dict[str, Any]]:
    db = Supabase(bearer(authorization))
    params = {"select": "*", "order": "created_at.desc", "limit": "100"}
    if status:
        params["status"] = f"eq.{status}"
    return [job_out(r) for r in await db.rest("GET", "jobs", params)]


@app.get("/jobs/{job_id}")
async def job(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    return job_out(await load_job(Supabase(bearer(authorization)), job_id))


# MARK: the phone's side

@app.post("/jobs")
async def create_job(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = Supabase(bearer(authorization))
    body = await request.json()
    files = [f for f in body.get("files", []) if safe_name(f)]
    if not files:
        raise HTTPException(400, "no files")
    user = await db.user()
    row = {
        "id": secrets.token_hex(4), "owner": user["id"], "name": str(body.get("name", "Capture"))[:80],
        "device": str(body.get("device", ""))[:80] or None, "captured_at": body.get("capturedAt") or now(),
        "status": "receiving", "files_expected": files, "stages": STAGES,
    }
    rows = await db.rest("POST", "jobs", body=row)
    return job_out(rows[0])


@app.post("/jobs/{job_id}/upload")
async def upload_url(job_id: str, request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """A signed URL to PUT one part of one input file to."""
    db = Supabase(bearer(authorization))
    body = await request.json()
    name, part = body.get("file", ""), int(body.get("part", 0))
    if not safe_name(name) or part < 0:
        raise HTTPException(400, "bad file")
    row = await load_job(db, job_id)
    if row["status"] != "receiving":
        raise HTTPException(409, "already sent")
    path = f"{row['owner']}/{job_id}/inputs/{name}.part{part:04d}"
    return {"url": await db.signed_upload(path), "partBytes": PART_BYTES}


@app.post("/jobs/{job_id}/received")
async def received(job_id: str, request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """All parts of one file are up. When every expected file is in, the job queues."""
    db = Supabase(bearer(authorization))
    body = await request.json()
    name, parts = body.get("file", ""), int(body.get("parts", 1))
    row = await load_job(db, job_id)
    if name not in row["files_expected"] or parts < 1:
        raise HTTPException(400, "not an expected file")
    got = list(dict.fromkeys((row.get("files_received") or []) + [name]))
    partmap = {**(row.get("parts") or {}), name: parts}
    change = {"files_received": got, "parts": partmap}
    if set(got) >= set(row["files_expected"]):
        change["status"] = "queued"
    rows = await db.rest("PATCH", "jobs", {"id": f"eq.{job_id}"}, change)
    return job_out(rows[0])


@app.post("/jobs/{job_id}/start")
async def start(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = Supabase(bearer(authorization))
    row = await load_job(db, job_id)
    if row["status"] == "receiving":
        if set(row.get("files_received") or []) < set(row["files_expected"]):
            raise HTTPException(409, "files still missing")
        rows = await db.rest("PATCH", "jobs", {"id": f"eq.{job_id}"}, {"status": "queued"})
        row = rows[0]
    return job_out(row)


@app.get("/jobs/{job_id}/results/{name}")
async def result(job_id: str, name: str, authorization: str | None = Header(default=None)) -> RedirectResponse:
    """Sends the caller to a signed download of one result."""
    db = Supabase(bearer(authorization))
    if not safe_name(name):
        raise HTTPException(400, "bad name")
    row = await load_job(db, job_id)
    return RedirectResponse(await db.signed_download(f"{row['owner']}/{job_id}/results/{name}"), status_code=302)


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = Supabase(bearer(authorization))
    row = await load_job(db, job_id)
    prefix = f"{row['owner']}/{job_id}"
    await db.delete_objects(await db.list_objects(f"{prefix}/inputs") + await db.list_objects(f"{prefix}/results"))
    await db.rest("DELETE", "jobs", {"id": f"eq.{job_id}"}, prefer="return=minimal")
    return {"deleted": job_id}


# MARK: the Mac's side (a worker signed in to the same account)

@app.post("/jobs/{job_id}/claim")
async def claim(job_id: str, request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Takes a queued job: only one Mac gets it (the update is conditional on the status)."""
    db = Supabase(bearer(authorization))
    body = await request.json()
    rows = await db.rest("PATCH", "jobs", {"id": f"eq.{job_id}", "status": "eq.queued"},
                         {"status": "running", "station": str(body.get("station", ""))[:80], "stage_index": 0,
                          "message": None})
    if not rows:
        raise HTTPException(409, "not queued")
    return job_out(rows[0])


@app.post("/jobs/{job_id}/download")
async def download_url(job_id: str, request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Signed URLs for every part of one input file, in order."""
    db = Supabase(bearer(authorization))
    body = await request.json()
    name = body.get("file", "")
    row = await load_job(db, job_id)
    parts = int((row.get("parts") or {}).get(name, 0))
    if not safe_name(name) or parts < 1:
        raise HTTPException(404, "no such file")
    urls = [await db.signed_download(f"{row['owner']}/{job_id}/inputs/{name}.part{i:04d}") for i in range(parts)]
    return {"urls": urls}


@app.patch("/jobs/{job_id}")
async def progress(job_id: str, request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Progress, outcome and results from the Mac."""
    db = Supabase(bearer(authorization))
    body = await request.json()
    change: dict[str, Any] = {}
    if "stageIndex" in body:
        change["stage_index"] = int(body["stageIndex"])
    if "message" in body:
        change["message"] = (str(body["message"])[:500] if body["message"] is not None else None)
    if body.get("status") in ("running", "done", "failed"):
        change["status"] = body["status"]
    if isinstance(body.get("results"), list):
        change["results"] = [
            {"name": r["name"], "kind": r.get("kind", "report"), "bytes": int(r.get("bytes", 0))}
            for r in body["results"] if isinstance(r, dict) and safe_name(str(r.get("name", "")))
        ]
    if not change:
        raise HTTPException(400, "nothing to change")
    rows = await db.rest("PATCH", "jobs", {"id": f"eq.{job_id}"}, change)
    if not rows:
        raise HTTPException(404, "no such job")
    return job_out(rows[0])


@app.post("/jobs/{job_id}/results/upload")
async def result_upload_url(job_id: str, request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = Supabase(bearer(authorization))
    body = await request.json()
    name = body.get("name", "")
    if not safe_name(name):
        raise HTTPException(400, "bad name")
    row = await load_job(db, job_id)
    return {"url": await db.signed_upload(f"{row['owner']}/{job_id}/results/{name}")}


@app.post("/jobs/{job_id}/inputs/delete")
async def delete_inputs(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """The Mac has the capture: free the space (the plan gives 1 GB)."""
    db = Supabase(bearer(authorization))
    row = await load_job(db, job_id)
    paths = await db.list_objects(f"{row['owner']}/{job_id}/inputs")
    await db.delete_objects(paths)
    await db.rest("PATCH", "jobs", {"id": f"eq.{job_id}"}, {"inputs_deleted": True}, prefer="return=minimal")
    return {"deleted": len(paths)}
