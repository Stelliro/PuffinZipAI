# Cloudflare Tunnel Setup — puffinzipai.stelliro.com

> Expose your PuffinZipAI server at **puffinzipai.stelliro.com** using a
> free Cloudflare Tunnel. Works on local machines, RunPod, Vast.ai, or
> any cloud server — no port forwarding, firewall changes, or public IP
> needed.

---

## Overview

```
Browser → https://puffinzipai.stelliro.com/login
         ↓ (Cloudflare DNS)
Cloudflare Edge → Encrypted Tunnel → localhost:<PORT>/login
                                      ↓
                                     Flask serves /login
```

`cloudflared` maintains a persistent outbound connection from your server
(local PC, RunPod pod, etc.) to Cloudflare's edge. Cloudflare routes
incoming HTTPS requests through that tunnel to your Flask server.

**Port is auto-detected.** The `start.sh` launcher generates a
`config.yml` with whichever port the server actually uses (5001 locally,
8080 on RunPod, etc.), so you never need to hard-code a port.

---

## Quick Start: RunPod / Cloud Pods

If you already have a tunnel created on your local PC, deploying to
RunPod is just **one env var**.

### Option A — Token (simplest)

1. On your **local PC** (where the tunnel was created):
   ```bash
   cloudflared tunnel token puffinzipai
   ```
   Copy the long token string.

2. Configure the tunnel's public hostname in the
   [Cloudflare Zero Trust dashboard](https://one.dash.cloudflare.com/):
   → **Networks** → **Tunnels** → select **puffinzipai** → **Public Hostname** →
   - **Subdomain**: `puffinzipai`
   - **Domain**: `stelliro.com`
   - **Service**: `HTTP` → `localhost:8080` (or whichever port your pod uses)

3. On **RunPod**, set the environment variable:
   ```
   CLOUDFLARE_TUNNEL_TOKEN=eyJ...  (the token from step 1)
   ```
   Or add it to `.env`:
   ```bash
   echo "CLOUDFLARE_TUNNEL_TOKEN=eyJ..." >> /workspace/PuffinZipAI/.env
   ```

4. Run `bash start.sh` — cloudflared is auto-installed and the tunnel starts.

### Option B — Base64 Credentials (auto port detection)

This approach generates `config.yml` dynamically with the **actual port**,
so you don't need to configure the port in the Cloudflare dashboard.

1. On your **local PC**, encode your tunnel credentials:
   ```bash
   # Linux / macOS
   base64 -w0 ~/.cloudflared/<tunnel-id>.json

   # Windows (PowerShell)
   [Convert]::ToBase64String([IO.File]::ReadAllBytes("$env:USERPROFILE\.cloudflared\<tunnel-id>.json"))
   ```
   Copy the base64 string.

2. On **RunPod**, add to `.env`:
   ```bash
   echo "CLOUDFLARE_TUNNEL_CREDS=<base64-string>" >> /workspace/PuffinZipAI/.env
   ```

3. Run `bash start.sh` — cloudflared is auto-installed, config is generated
   with the correct port, and the tunnel starts.

> **Tip:** `<tunnel-id>` is the UUID of your tunnel. Find it with
> `cloudflared tunnel list` on your local PC, or just look for the `.json`
> file in `~/.cloudflared/`.

---

## Full Setup (from scratch)

If you don't have a tunnel yet, follow these steps on any machine.

### Prerequisites

- A **Cloudflare account** (free): https://dash.cloudflare.com/sign-up
- Your domain's DNS managed by Cloudflare (nameservers updated at your registrar)
- **PuffinZipAI** installed with credentials configured

### Step 1: Move DNS to Cloudflare (one-time)

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Click **"Add a site"** → enter your domain → select **Free plan**
3. Cloudflare scans existing DNS — verify records look correct
4. Cloudflare gives you two nameservers
5. Update nameservers at your domain registrar
6. Wait for propagation (5–30 minutes, up to 24 hours)
7. Cloudflare dashboard shows **"Active"** when ready

> ⚠️ Copy any existing DNS records (email MX, other subdomains) from
> your registrar to Cloudflare during the scan in step 3.

### Step 2: Install cloudflared

> **On RunPod:** `start.sh` installs cloudflared **automatically**. Skip this step.

**Linux (Debian/Ubuntu):**
```bash
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cf.deb
sudo dpkg -i /tmp/cf.deb
```

**Windows (winget):**
```powershell
winget install Cloudflare.cloudflared
```

**macOS (Homebrew):**
```bash
brew install cloudflare/cloudflare/cloudflared
```

**Verify:**
```bash
cloudflared --version
```

### Step 3: Authenticate

```bash
cloudflared tunnel login
```

Opens browser → log in to Cloudflare → select your domain → authorizes
cloudflared. Certificate saved to `~/.cloudflared/cert.pem`.

### Step 4: Create a named tunnel

```bash
cloudflared tunnel create puffinzipai
```

Credentials saved to `~/.cloudflared/<tunnel-id>.json`.
Note the **Tunnel ID** (a UUID).

### Step 5: Route DNS

```bash
cloudflared tunnel route dns puffinzipai puffinzipai.stelliro.com
```

Creates a CNAME record: `puffinzipai.stelliro.com → <tunnel-id>.cfargotunnel.com`

### Step 6: Create tunnel config

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: puffinzipai
credentials-file: /home/<user>/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: puffinzipai.stelliro.com
    service: http://localhost:5001
  - service: http_status:404
```

> **Note:** `start.sh` **auto-generates** this config with the correct port.
> You only need this file if running cloudflared manually.

### Step 7: Test

```bash
cloudflared tunnel run puffinzipai
```

In another terminal, start PuffinZipAI:

```bash
bash start.sh
```

Visit **https://puffinzipai.stelliro.com/login** — you should see the login page.

---

## How start.sh Handles Tunnels

`start.sh` automates the entire tunnel lifecycle:

| Step | What happens |
|------|-------------|
| **Auto-install** | Downloads & installs cloudflared on Linux if missing |
| **Token check** | If `CLOUDFLARE_TUNNEL_TOKEN` is set → `cloudflared tunnel run --token` |
| **Creds check** | If `CLOUDFLARE_TUNNEL_CREDS` is set → decode, generate config.yml with actual port |
| **Local check** | Falls back to existing `~/.cloudflared/*.json` credentials |
| **Dynamic config** | Always generates config.yml with the server's actual port |
| **Cleanup** | Stops tunnel process on Ctrl+C / exit |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CLOUDFLARE_TUNNEL_TOKEN` | Tunnel token (from `cloudflared tunnel token <name>`) |
| `CLOUDFLARE_TUNNEL_CREDS` | Base64-encoded credentials JSON |
| `CLOUDFLARE_TUNNEL_NAME` | Tunnel name (default: `puffinzipai`) |
| `PUFFIN_CUSTOM_URL` | Custom URL shown in banner (also set via `webui_credentials.json`) |

All variables can go in `.env` (git-ignored, loaded by `start.sh`).

---

## Automatic Startup Options

### Option A: start.sh handles it (recommended)

`start.sh` auto-detects cloudflared and starts the tunnel before launching
the server. Just run `bash start.sh`.

### Option B: systemd service (Linux)

```bash
cloudflared service install
```

### Option C: Windows service

```powershell
cloudflared service install
```

---

## Architecture

| Layer | What happens |
|-------|-------------|
| **DNS** | `puffinzipai.stelliro.com` → CNAME → `<tunnel-id>.cfargotunnel.com` |
| **Cloudflare Edge** | Terminates TLS, routes through tunnel |
| **cloudflared** | Outbound-only connection from server to Cloudflare |
| **ProxyFix** | Reads `X-Forwarded-*` headers from Cloudflare |
| **Flask** | Serves login at `/login`, dashboard at `/dashboard`, APIs at `/api/*` |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "tunnel not found" | Run `cloudflared tunnel list` to verify |
| 502 Bad Gateway | Flask server not running — start it first, or check the port |
| Login page won't load | Check `custom_url` in `webui_credentials.json` |
| ERR_NAME_NOT_RESOLVED | DNS not propagated yet — wait or check nameservers |
| "connection refused" | cloudflared can't reach Flask — verify port matches |
| Tunnel starts but site down | Check `cloudflared tunnel info puffinzipai` for connector status |
| Wrong port on RunPod | Use `CLOUDFLARE_TUNNEL_CREDS` (auto-generates config with correct port) |

---

## Quick Test (No Account Needed)

For a quick test without setting up DNS, use a temporary tunnel:

```bash
cloudflared tunnel --url http://localhost:5001
```

Gives a random `https://xxxxx.trycloudflare.com` URL. Not persistent.

---

## Cost

**Free.** Cloudflare Tunnels, DNS, and the free plan are all $0. No
bandwidth limits for tunnels.
