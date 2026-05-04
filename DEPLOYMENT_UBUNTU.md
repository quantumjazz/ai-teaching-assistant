# Ubuntu Laptop Deployment

This app can be uploaded and hosted the same way as the local `uni_dashboard`
deployment:

```text
GitHub over SSH -> Ubuntu laptop clone -> systemd -> Cloudflare Tunnel HTTPS URL
```

Cloudflare provides the public HTTPS URL. The laptop only runs a local service
bound to `127.0.0.1`, so no router port forwarding is needed.

In systemd snippets, replace `YOUR_USER` with the Ubuntu username from `whoami`.

## Course URLs

Run one app service per course. Use folder names for course data and DNS-safe
hostnames for public URLs:

```text
https://heuristics.visiometrica.com
  -> /home/YOUR_USER/course-data/heuristics
  -> 127.0.0.1:8011

https://society-economics-business.visiometrica.com
  -> /home/YOUR_USER/course-data/society_economics_business
  -> 127.0.0.1:8012
```

Keep the tracked synthetic files in `documents/demo_*.txt` for a safe GitHub and
Render-style public demo. Put real course documents in the external
`COURSE_ROOT` folders below, not in Git.

A single running app URL currently serves one active course root. It does not
yet provide an in-app course picker that switches indexes per request.

## Upload To GitHub

Use GitHub SSH URLs, matching the `uni_dashboard` workflow. If your Mac already
pushes `uni_dashboard` with `git@github.com:...`, the SSH key is already set up.

Create an empty GitHub repo named `ai-teaching-assistant`, then on the Mac:

```bash
cd /Users/victor/Documents/Projects/ai-teaching-assistant
git status --short
git remote -v
```

If there is no `origin` remote yet:

```bash
git remote add origin git@github.com:quantumjazz/ai-teaching-assistant.git
```

If `origin` exists but points somewhere else:

```bash
git remote set-url origin git@github.com:quantumjazz/ai-teaching-assistant.git
```

Commit and push:

```bash
git branch -M main
git add .
git commit -m "Prepare AI teaching assistant deployment"
git push -u origin main
```

The repository ignores `.env`, generated `data/`, and private course documents.
Only the tracked demo documents are uploaded by default.

If SSH is not configured on the Ubuntu laptop, create a key there and add the
public key to GitHub:

```bash
ssh-keygen -t ed25519 -C "ubuntu-laptop"
cat ~/.ssh/id_ed25519.pub
```

Test GitHub access:

```bash
ssh -T git@github.com
```

## Clone App Code Once

On the Ubuntu laptop:

```bash
mkdir -p ~/apps
cd ~/apps
git clone git@github.com:quantumjazz/ai-teaching-assistant.git
cd ai-teaching-assistant

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Create Course Data Folders

Create external course roots. Each course gets its own settings, documents,
generated index data, and secrets:

```bash
mkdir -p ~/course-data/heuristics/documents
mkdir -p ~/course-data/heuristics/data
mkdir -p ~/course-data/society_economics_business/documents
mkdir -p ~/course-data/society_economics_business/data

cp ~/apps/ai-teaching-assistant/settings.txt ~/course-data/heuristics/settings.txt
cp ~/apps/ai-teaching-assistant/settings.txt ~/course-data/society_economics_business/settings.txt
```

Edit each settings file so the course title, description, and instructions match
the course:

```bash
nano ~/course-data/heuristics/settings.txt
nano ~/course-data/society_economics_business/settings.txt
```

Create `.env` for each course. Reuse the same `OPENAI_API_KEY`; use a stable
secret for each service:

```bash
cat > ~/course-data/heuristics/.env <<'EOF'
OPENAI_API_KEY=your_api_key_here
FLASK_SECRET_KEY=replace_with_a_long_random_secret_for_heuristics
AUTO_INDEX_ON_STARTUP=true
SESSION_COOKIE_SECURE=true
EOF

cat > ~/course-data/society_economics_business/.env <<'EOF'
OPENAI_API_KEY=your_api_key_here
FLASK_SECRET_KEY=replace_with_a_long_random_secret_for_society_economics_business
AUTO_INDEX_ON_STARTUP=true
SESSION_COOKIE_SECURE=true
EOF

chmod 600 ~/course-data/heuristics/.env
chmod 600 ~/course-data/society_economics_business/.env
```

Put course PDF, DOCX, or TXT files here:

```bash
~/course-data/heuristics/documents/
~/course-data/society_economics_business/documents/
```

The ingestion pipeline scans subfolders too, so you can organize literature
inside each course, for example `articles/`, `slides/`, and `syllabus/`.

If you are copying files from the Mac to the Ubuntu laptop, run this from a Mac
terminal, not from inside the Ubuntu SSH session:

```bash
scp -r /Users/victor/Documents/Courses/heuristics/literature/* victor@UBUNTU_IP:/home/victor/course-data/heuristics/documents/
scp -r /Users/victor/Documents/Courses/society_economics_business/literature/* victor@UBUNTU_IP:/home/victor/course-data/society_economics_business/documents/
```

If you are already logged into the Ubuntu laptop, exit that shell first or open
a new terminal tab on the Mac before running `scp`.

Before indexing a large course folder, list image-only or scanned PDFs. These
must be OCRed or moved out of `documents/`; otherwise indexing stops before the
web server starts:

```bash
cd ~/apps/ai-teaching-assistant
source venv/bin/activate

python - <<'PY'
from pathlib import Path
from PyPDF2 import PdfReader

docs = Path.home() / "course-data/society_economics_business/documents"
bad = []
for path in sorted(docs.rglob("*.pdf")):
    try:
        reader = PdfReader(str(path))
        has_text = False
        for page in reader.pages:
            if (page.extract_text() or "").strip():
                has_text = True
                break
        if not has_text:
            bad.append(path)
    except Exception:
        bad.append(path)

for path in bad:
    print(path)
print(f"\nUnreadable or scanned PDFs: {len(bad)}")
PY
```

For a quick demo, move scanned PDFs out of the course documents tree and OCR
them later:

```bash
mkdir -p ~/course-data/society_economics_business/needs_ocr
mv ~/course-data/society_economics_business/documents/04_auctions/Milgrom_1989_Auctions_and_Bidding_Primer.pdf \
  ~/course-data/society_economics_business/needs_ocr/
```

## Build Initial Indexes

After adding documents, build each course index:

```bash
cd ~/apps/ai-teaching-assistant
source venv/bin/activate

COURSE_ROOT=$HOME/course-data/heuristics make index
COURSE_ROOT=$HOME/course-data/society_economics_business make index
```

If embedding fails with `429` and `insufficient_quota`, the OpenAI API key has
no available API credits or has hit its monthly spend limit. Fix billing or use
a key from a project with available API quota, then rerun the failed indexing
step.

If embedding fails with `rate_limit_exceeded` and says the request is too large
for the project's tokens-per-minute limit, reduce embedding batch size and add a
pause between batches in that course's `settings.txt`:

```text
max_tokens_per_batch=10000
embedding_rate_limit_sleep_seconds=65
```

With a 40,000 TPM limit and a large course corpus, initial embedding can take a
while. Let the command run until `embedded_data.pkl`, `faiss_index.bin`, and
`index_report.json` are written.

A new key can still fail if it belongs to the same organization/project without
quota. It can also fail if the process is still reading an older key. Verify the
course `.env` key without printing the full secret:

```bash
COURSE_ROOT=$HOME/course-data/society_economics_business python - <<'PY'
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path.home() / "course-data/society_economics_business/.env", override=True)
key = os.getenv("OPENAI_API_KEY", "")
print(f"OPENAI_API_KEY fingerprint: {key[:8]}...{key[-4:]} length={len(key)}")
PY
```

If the fingerprint is not the new key, update
`~/course-data/society_economics_business/.env`, then retry.

For lower embedding cost, set this in each course `settings.txt` before
indexing:

```text
openai_embedding_model=text-embedding-3-small
```

After changing the embedding model, rebuild that course's full index:

```bash
COURSE_ROOT=$HOME/course-data/society_economics_business make clean-data
COURSE_ROOT=$HOME/course-data/society_economics_business make index
```

## Smoke Test Locally

Test each service manually before creating systemd units:

```bash
cd ~/apps/ai-teaching-assistant
source venv/bin/activate

COURSE_ROOT=$HOME/course-data/heuristics HOST=127.0.0.1 PORT=8011 sh scripts/start_web.sh
```

Open `http://127.0.0.1:8011/setup` in a browser on the Ubuntu laptop. Stop it
with `Ctrl+C`, then test the second course:

```bash
COURSE_ROOT=$HOME/course-data/society_economics_business HOST=127.0.0.1 PORT=8012 sh scripts/start_web.sh
```

Open `http://127.0.0.1:8012/setup` in a browser on the Ubuntu laptop.

Use `127.0.0.1` only in a browser running on the Ubuntu laptop itself. To test
from the Mac over the LAN, bind the temporary smoke-test server to `0.0.0.0`:

```bash
COURSE_ROOT=$HOME/course-data/society_economics_business HOST=0.0.0.0 PORT=8012 sh scripts/start_web.sh
```

Then open `http://UBUNTU_IP:8012/setup` from the Mac.

If `scripts/start_web.sh` stops with an indexing error, gunicorn has not started
and Safari will not be able to connect. Fix the reported document first, then
run the start command again.

For scanned or image-only PDFs, either OCR the file or temporarily move it out
of the course `documents/` folder before indexing:

```bash
mkdir -p ~/course-data/society_economics_business/needs_ocr
mv ~/course-data/society_economics_business/documents/02_agency/Holmstrom_Milgrom_1991_Multitask_Principal_Agent.pdf \
  ~/course-data/society_economics_business/needs_ocr/
COURSE_ROOT=$HOME/course-data/society_economics_business make index
```

## systemd Services

Create `/etc/systemd/system/ai-teaching-assistant-heuristics.service`:

```ini
[Unit]
Description=AI Teaching Assistant - Heuristics
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/apps/ai-teaching-assistant
Environment="PATH=/home/YOUR_USER/apps/ai-teaching-assistant/venv/bin:/usr/bin"
Environment="COURSE_ROOT=/home/YOUR_USER/course-data/heuristics"
EnvironmentFile=/home/YOUR_USER/course-data/heuristics/.env
Environment="HOST=127.0.0.1"
Environment="PORT=8011"
ExecStart=/bin/sh scripts/start_web.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/ai-teaching-assistant-society-economics-business.service`:

```ini
[Unit]
Description=AI Teaching Assistant - Society, Economics, Business
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/apps/ai-teaching-assistant
Environment="PATH=/home/YOUR_USER/apps/ai-teaching-assistant/venv/bin:/usr/bin"
Environment="COURSE_ROOT=/home/YOUR_USER/course-data/society_economics_business"
EnvironmentFile=/home/YOUR_USER/course-data/society_economics_business/.env
Environment="HOST=127.0.0.1"
Environment="PORT=8012"
ExecStart=/bin/sh scripts/start_web.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable both:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-teaching-assistant-heuristics
sudo systemctl enable ai-teaching-assistant-society-economics-business
sudo systemctl start ai-teaching-assistant-heuristics
sudo systemctl start ai-teaching-assistant-society-economics-business
sudo systemctl status ai-teaching-assistant-heuristics
sudo systemctl status ai-teaching-assistant-society-economics-business
```

Useful logs:

```bash
sudo journalctl -u ai-teaching-assistant-heuristics -f
sudo journalctl -u ai-teaching-assistant-society-economics-business -f
```

## Public HTTPS Routing

First check how the existing laptop tunnel is wired:

```bash
sudo systemctl status caddy
sudo systemctl cat cloudflared
```

If `caddy` is active, use the Caddy route below. That matches the original
`uni_dashboard` pattern: Cloudflare Tunnel forwards traffic to Caddy, and Caddy
routes hostnames to local app ports.

### Option A: Caddy Route

Edit Caddy:

```bash
sudo nano /etc/caddy/Caddyfile
```

Add:

```caddy
heuristics.visiometrica.com {
    reverse_proxy 127.0.0.1:8011
}

society-economics-business.visiometrica.com {
    reverse_proxy 127.0.0.1:8012
}
```

Validate and reload:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Make sure both DNS names are routed to the existing tunnel:

```bash
cloudflared tunnel route dns laptop-server heuristics.visiometrica.com
cloudflared tunnel route dns laptop-server society-economics-business.visiometrica.com
sudo systemctl restart cloudflared
```

### Option B: Direct Cloudflared Route

Use this only if you are not using Caddy for the existing apps. Add ingress
rules to the `cloudflared` config file that the service actually uses. Check the
path in `sudo systemctl cat cloudflared`; it is often
`/home/victor/.cloudflared/config.yml` when installed with an explicit config,
but can differ.

The app hostname rules must appear above the final `http_status:404` catch-all:

```yaml
ingress:
  - hostname: heuristics.visiometrica.com
    service: http://127.0.0.1:8011

  - hostname: society-economics-business.visiometrica.com
    service: http://127.0.0.1:8012

  - service: http_status:404
```

Create the DNS routes and restart the tunnel:

```bash
cloudflared tunnel route dns laptop-server heuristics.visiometrica.com
cloudflared tunnel route dns laptop-server society-economics-business.visiometrica.com
sudo systemctl restart cloudflared
```

If the public URL returns 404, the hostname is usually not matching the active
Cloudflare Tunnel or Caddy config. Check logs:

```bash
sudo journalctl -u cloudflared -n 100 --no-pager
sudo journalctl -u caddy -n 100 --no-pager
```

The apps should then be live at:

```text
https://heuristics.visiometrica.com
https://society-economics-business.visiometrica.com
```

Using one domain with paths such as
`https://teaching-assistant.visiometrica.com/heuristics` is possible, but it
requires additional app code: request-level course selection, per-course session
handling, and route-aware health/document pages.

## Updating

```bash
cd ~/apps/ai-teaching-assistant
git pull --ff-only origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart ai-teaching-assistant-heuristics
sudo systemctl restart ai-teaching-assistant-society-economics-business
```

When documents change, rebuild the affected course index and restart only that
course:

```bash
COURSE_ROOT=$HOME/course-data/heuristics make index
sudo systemctl restart ai-teaching-assistant-heuristics

COURSE_ROOT=$HOME/course-data/society_economics_business make index
sudo systemctl restart ai-teaching-assistant-society-economics-business
```
