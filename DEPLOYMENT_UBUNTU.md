# Ubuntu Laptop Deployment

This app can be uploaded and hosted the same way as the local `uni_dashboard`
deployment:

```text
GitHub over SSH -> Ubuntu laptop clone -> systemd -> Cloudflare Tunnel HTTPS URL
```

Cloudflare provides the public HTTPS URL. The laptop only runs a local service
bound to `127.0.0.1`, so no router port forwarding is needed.

## Recommended Demo Shape

Use one public demo course first:

```text
https://teaching-assistant.visiometrica.com
```

Keep the tracked synthetic files in `documents/demo_*.txt` for a safe GitHub and
Render-style public demo. Use real course documents only on private deployments,
or keep them outside the Git repo with `COURSE_ROOT`.

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

## Single-Course Laptop Setup

On the Ubuntu laptop:

```bash
mkdir -p ~/apps
cd ~/apps
git clone git@github.com:YOUR_USER/ai-teaching-assistant.git
cd ai-teaching-assistant

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create a local environment file:

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=your_api_key_here
FLASK_SECRET_KEY=replace_with_a_long_random_secret
AUTO_INDEX_ON_STARTUP=true
SESSION_COOKIE_SECURE=true
EOF

chmod 600 .env
```

Add PDF, DOCX, or TXT files under `documents/`, then test the index and app:

```bash
source venv/bin/activate
make index
HOST=127.0.0.1 PORT=8010 sh scripts/start_web.sh
```

Open `http://127.0.0.1:8010/setup` on the laptop. Stop gunicorn with
`Ctrl+C` after the smoke test.

## systemd Service

Create `/etc/systemd/system/ai-teaching-assistant.service`:

```ini
[Unit]
Description=AI Teaching Assistant
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/apps/ai-teaching-assistant
Environment="PATH=/home/YOUR_USER/apps/ai-teaching-assistant/venv/bin:/usr/bin"
EnvironmentFile=/home/YOUR_USER/apps/ai-teaching-assistant/.env
Environment="HOST=127.0.0.1"
Environment="PORT=8010"
ExecStart=/bin/sh scripts/start_web.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-teaching-assistant
sudo systemctl start ai-teaching-assistant
sudo systemctl status ai-teaching-assistant
```

Useful logs:

```bash
sudo journalctl -u ai-teaching-assistant -f
```

## Cloudflare Tunnel Route

Add an ingress rule to the existing `~/.cloudflared/config.yml` used by the
other laptop apps:

```yaml
ingress:
  - hostname: teaching-assistant.visiometrica.com
    service: http://127.0.0.1:8010

  - service: http_status:404
```

Create the DNS route and restart the tunnel:

```bash
cloudflared tunnel route dns laptop-server teaching-assistant.visiometrica.com
sudo systemctl restart cloudflared
```

The app should then be live at:

```text
https://teaching-assistant.visiometrica.com
```

## URL And Courses

A single running app URL currently serves one active course root. It does not
yet provide an in-app course picker that switches indexes per request.

For one public demo URL, point that service at one `COURSE_ROOT`:

```text
https://teaching-assistant.visiometrica.com -> COURSE_ROOT=/home/YOUR_USER/course-data/heuristics
```

For multiple courses today, use separate services and separate hostnames:

```text
https://heuristics.visiometrica.com -> COURSE_ROOT=/home/YOUR_USER/course-data/heuristics
https://research-methods.visiometrica.com -> COURSE_ROOT=/home/YOUR_USER/course-data/research-methods
```

Using one domain with paths such as
`https://teaching-assistant.visiometrica.com/heuristics` is possible, but it
requires additional app code: request-level course selection, per-course session
handling, and route-aware health/document pages.

## Multiple Course Services

Do not put unrelated courses into one global `documents/` folder unless the
assistant is intended to answer across all of them. The current retriever builds
one knowledge base per running app instance.

For multiple courses on the same laptop, run one service per course. The code
supports external course roots:

```text
~/course-data/
  heuristics/
    settings.txt
    .env
    documents/
    data/
  research-methods/
    settings.txt
    .env
    documents/
    data/
```

Each course service can share the same Git checkout but set a different
`COURSE_ROOT` and port:

```ini
[Service]
WorkingDirectory=/home/YOUR_USER/apps/ai-teaching-assistant
Environment="PATH=/home/YOUR_USER/apps/ai-teaching-assistant/venv/bin:/usr/bin"
Environment="COURSE_ROOT=/home/YOUR_USER/course-data/heuristics"
EnvironmentFile=/home/YOUR_USER/course-data/heuristics/.env
Environment="HOST=127.0.0.1"
Environment="PORT=8011"
ExecStart=/bin/sh scripts/start_web.sh
```

Then add one Cloudflare ingress hostname per course:

```yaml
ingress:
  - hostname: heuristics.visiometrica.com
    service: http://127.0.0.1:8011
  - hostname: research-methods.visiometrica.com
    service: http://127.0.0.1:8012
  - service: http_status:404
```

This keeps documents, generated FAISS data, settings, and secrets separated
while reusing one application codebase.

## Updating

```bash
cd ~/apps/ai-teaching-assistant
git pull --ff-only origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart ai-teaching-assistant
```

When documents change, rebuild that course index:

```bash
COURSE_ROOT=/home/YOUR_USER/course-data/heuristics make index
sudo systemctl restart ai-teaching-assistant-heuristics
```
