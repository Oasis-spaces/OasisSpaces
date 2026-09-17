# Oasis relay

A small web service (FastAPI, free Render instance) that lets the phone app
and the Mac app exchange captures and results when they are not on the same
Wi-Fi. It speaks the same job API as a Mac on the local network, so the apps
use one client for both; see the docstring in `main.py` for the design.

Deploy: Render web service, root directory `relay`, build
`pip install -r requirements.txt`, start `uvicorn main:app --host 0.0.0.0 --port $PORT`,
environment `SUPABASE_URL` and `SUPABASE_KEY` (the project's publishable key).
The database side is the `jobs` table and the `oasis` bucket in the Supabase
project (migration `cloud_jobs_and_files`).

Test: `~/.venvs/oasis-relay/bin/python -m pytest relay/tests`.
