"""The relay against a fake Supabase: the phone's flow and the Mac's flow."""

import json
import os
import re
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

os.environ["SUPABASE_URL"] = "https://fake.supabase.co"
os.environ["SUPABASE_KEY"] = "pk"
import main  # noqa: E402


class FakeSupabase:
    """Just enough of Auth, PostgREST and Storage for the relay."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.objects: set[str] = set()
        self.users = {"tok-phone": "user-1", "tok-mac": "user-1", "tok-other": "user-2"}

    def handler(self, request: httpx.Request) -> httpx.Response:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        assert request.headers.get("apikey") == "pk"
        user = self.users.get(token)
        path = request.url.path
        if path == "/auth/v1/user":
            return httpx.Response(200, json={"id": user}) if user else httpx.Response(401)
        if not user:
            return httpx.Response(401, json={"message": "bad jwt"})
        if path == "/rest/v1/jobs":
            return self.rest(request, user)
        if path.startswith("/storage/v1/object/upload/sign/oasis/"):
            key = path.split("/upload/sign/oasis/")[1]
            assert key.startswith(user + "/"), "row policy: own folder only"
            self.objects.add(key)
            return httpx.Response(200, json={"url": f"/object/upload/sign/oasis/{key}?token=signed"})
        if path.startswith("/storage/v1/object/sign/oasis/"):
            key = path.split("/sign/oasis/")[1]
            if key not in self.objects or not key.startswith(user + "/"):
                return httpx.Response(404, json={"message": "not found"})
            return httpx.Response(200, json={"signedURL": f"/object/sign/oasis/{key}?token=dl"})
        if path == "/storage/v1/object/list/oasis":
            prefix = json.loads(request.content)["prefix"]
            names = [k[len(prefix) + 1:] for k in self.objects if k.startswith(prefix + "/")]
            return httpx.Response(200, json=[{"name": n} for n in names])
        if path == "/storage/v1/object/oasis" and request.method == "DELETE":
            for p in json.loads(request.content)["prefixes"]:
                self.objects.discard(p)
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    def rest(self, request: httpx.Request, user: str) -> httpx.Response:
        q = parse_qs(str(request.url.query, "utf8"))
        def matches(row):
            if row["owner"] != user:
                return False
            for key, values in q.items():
                if key in ("select", "order", "limit"):
                    continue
                op, value = values[0].split(".", 1)
                assert op == "eq"
                if str(row.get(key)) != value:
                    return False
            return True
        if request.method == "GET":
            rows = [r for r in self.rows.values() if matches(r)]
            rows.sort(key=lambda r: r["created_at"], reverse=True)
            return httpx.Response(200, json=rows)
        if request.method == "POST":
            row = json.loads(request.content)
            assert row["owner"] == user, "row policy: owner must be the caller"
            row = {"created_at": f"2026-09-18T00:00:{len(self.rows):02d}Z", "stage_index": 0, "message": None,
                   "files_received": [], "parts": {}, "results": [], "station": None, **row}
            self.rows[row["id"]] = row
            return httpx.Response(201, json=[row])
        if request.method == "PATCH":
            change = json.loads(request.content)
            out = []
            for row in self.rows.values():
                if matches(row):
                    row.update(change)
                    out.append(row)
            return httpx.Response(200, json=out)
        if request.method == "DELETE":
            gone = [k for k, r in self.rows.items() if matches(r)]
            for k in gone:
                del self.rows[k]
            return httpx.Response(204)
        return httpx.Response(405)


@pytest.fixture
def relay():
    fake = FakeSupabase()
    main.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    transport = httpx.ASGITransport(app=main.app)
    client = httpx.AsyncClient(transport=transport, base_url="http://relay")
    return client, fake


def auth(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.anyio
async def test_phone_sends_a_capture_and_the_mac_processes_it(relay):
    client, fake = relay
    phone, mac = auth("tok-phone"), auth("tok-mac")

    r = await client.get("/info")
    assert r.json()["relay"] is True and r.json()["id"] == "cloud"
    r = await client.post("/pair/account", json={"accessToken": "tok-phone", "device": "iPhone"})
    assert r.status_code == 200 and r.json()["token"] == "tok-phone" and r.json()["station"]["owner"] == "user-1"
    r = await client.post("/pair/account", json={"accessToken": "nope", "device": "iPhone"})
    assert r.status_code == 401

    # The phone creates a job and uploads its files in parts.
    r = await client.post("/jobs", json={"name": "Bedroom", "files": ["capture.json", "video.mov"], "capturedAt": "2026-09-18T01:00:00Z", "device": "iPhone 13"}, headers=phone)
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "receiving" and job["stages"] == main.STAGES and job["cloud"] is True
    jid = job["id"]
    r = await client.post(f"/jobs/{jid}/upload", json={"file": "capture.json", "part": 0}, headers=phone)
    assert r.json()["url"].startswith("https://fake.supabase.co/storage/v1/object/upload/sign/oasis/user-1/" + jid + "/inputs/capture.json.part0000")
    assert r.json()["partBytes"] == main.PART_BYTES
    r = await client.post(f"/jobs/{jid}/received", json={"file": "capture.json", "parts": 1}, headers=phone)
    assert r.json()["status"] == "receiving" and r.json()["filesReceived"] == ["capture.json"]
    r = await client.post(f"/jobs/{jid}/start", headers=phone)
    assert r.status_code == 409, "the video is still missing"
    for part in range(3):
        await client.post(f"/jobs/{jid}/upload", json={"file": "video.mov", "part": part}, headers=phone)
    r = await client.post(f"/jobs/{jid}/received", json={"file": "video.mov", "parts": 3}, headers=phone)
    assert r.json()["status"] == "queued" and r.json()["parts"] == {"capture.json": 1, "video.mov": 3}
    r = await client.post(f"/jobs/{jid}/upload", json={"file": "video.mov", "part": 3}, headers=phone)
    assert r.status_code == 409, "no more uploads once sent"

    # Another account sees nothing of it.
    r = await client.get("/jobs", headers=auth("tok-other"))
    assert r.json() == []
    r = await client.get(f"/jobs/{jid}", headers=auth("tok-other"))
    assert r.status_code == 404

    # The Mac (same account) finds it, claims it once, downloads the parts, reports progress.
    r = await client.get("/jobs?status=queued", headers=mac)
    assert [j["id"] for j in r.json()] == [jid]
    r = await client.post(f"/jobs/{jid}/claim", json={"station": "MacBook"}, headers=mac)
    assert r.json()["status"] == "running" and r.json()["station"] == "MacBook"
    r = await client.post(f"/jobs/{jid}/claim", json={"station": "Other Mac"}, headers=mac)
    assert r.status_code == 409, "claimed only once"
    r = await client.post(f"/jobs/{jid}/download", json={"file": "video.mov"}, headers=mac)
    assert len(r.json()["urls"]) == 3 and "video.mov.part0002" in r.json()["urls"][2]
    r = await client.post(f"/jobs/{jid}/download", json={"file": "nothing.bin"}, headers=mac)
    assert r.status_code == 404
    r = await client.post(f"/jobs/{jid}/inputs/delete", headers=mac)
    assert r.json()["deleted"] == 4
    r = await client.patch(f"/jobs/{jid}", json={"stageIndex": 2, "message": "shapes"}, headers=mac)
    assert r.json()["stageIndex"] == 2 and r.json()["message"] == "shapes"
    r = await client.post(f"/jobs/{jid}/results/upload", json={"name": "splat-filled.splat"}, headers=mac)
    assert "/results/splat-filled.splat" in r.json()["url"]
    r = await client.patch(f"/jobs/{jid}", json={"status": "done", "results": [{"name": "splat-filled.splat", "kind": "splat", "bytes": 6000000}, {"name": "../evil", "kind": "splat", "bytes": 1}]}, headers=mac)
    assert r.json()["status"] == "done" and [x["name"] for x in r.json()["results"]] == ["splat-filled.splat"]

    # The phone follows the job and fetches the result through a signed link.
    r = await client.get(f"/jobs/{jid}", headers=phone)
    assert r.json()["status"] == "done"
    r = await client.get(f"/jobs/{jid}/results/splat-filled.splat", headers=phone, follow_redirects=False)
    assert r.status_code == 302 and "results/splat-filled.splat?token=dl" in r.headers["location"]
    r = await client.get(f"/jobs/{jid}/results/missing.splat", headers=phone, follow_redirects=False)
    assert r.status_code == 404

    # Deleting the job removes its files too.
    r = await client.delete(f"/jobs/{jid}", headers=phone)
    assert r.json() == {"deleted": jid}
    assert fake.rows == {} and not [k for k in fake.objects if jid in k]


@pytest.mark.anyio
async def test_without_a_token_nothing_works(relay):
    client, _ = relay
    for path in ["/jobs", "/jobs/x"]:
        assert (await client.get(path)).status_code == 401
    assert (await client.post("/jobs", json={"name": "x", "files": ["a"]})).status_code == 401
    assert (await client.post("/jobs", json={"name": "x", "files": ["../a"]}, headers=auth("tok-phone"))).status_code == 400


@pytest.fixture
def anyio_backend():
    return "asyncio"
