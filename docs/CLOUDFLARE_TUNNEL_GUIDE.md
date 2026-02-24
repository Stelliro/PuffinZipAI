# Cloudflare Tunnel Setup — stelliro.com/PuffinZipAI

> Expose your local PuffinZipAI server at **stelliro.com/PuffinZipAI** using a
> free Cloudflare Tunnel. No port forwarding, firewall changes, or public IP
> needed.

---

## Overview

```
Browser → stelliro.com/PuffinZipAI/login
         ↓ (Cloudflare DNS)
Cloudflare Edge → Encrypted Tunnel → localhost:5001/PuffinZipAI/login
                                      ↓ (PrefixMiddleware strips /PuffinZipAI)
                                     Flask serves /login
```

Your PC runs `cloudflared` which maintains a persistent outbound connection to
Cloudflare's edge. Cloudflare routes incoming requests for your domain through
that tunnel to your local Flask server. The `PrefixMiddleware` strips the
`/PuffinZipAI` prefix so Flask routes work normally.

---

## Prerequisites

- A **Cloudflare account** (free plan is fine): https://dash.cloudflare.com/sign-up
- **stelliro.com** added to Cloudflare (nameservers updated at Spaceship)
- **PuffinZipAI** installed with credentials configured

---

## Step 1: Move DNS to Cloudflare (one-time)

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Click **"Add a site"** → enter `stelliro.com` → select **Free plan**
3. Cloudflare scans existing DNS records — verify they look correct
4. Cloudflare gives you two nameservers (e.g. `ada.ns.cloudflare.com`)
5. Go to **Spaceship** → Domain Settings → Nameservers → change to Cloudflare's
6. Wait for propagation (usually 5–30 minutes, can take up to 24 hours)
7. Cloudflare dashboard shows **"Active"** when ready

> ⚠️ Any DNS records (email MX, other subdomains) should be copied from
> Spaceship to Cloudflare during the scan in step 3. Verify before changing
> nameservers.

---

## Step 2: Install cloudflared

### Windows (recommended: winget)

```powershell
winget install Cloudflare.cloudflared
```

### Windows (alternative: direct download)

Download from https://github.com/cloudflare/cloudflared/releases/latest
→ `cloudflared-windows-amd64.exe` → rename to `cloudflared.exe` → put in PATH.

### Verify

```powershell
cloudflared --version
```

---

## Step 3: Authenticate cloudflared

```powershell
cloudflared tunnel login
```

This opens your browser → log in to Cloudflare → select `stelliro.com` →
authorizes cloudflared. A certificate is saved to
`%USERPROFILE%\.cloudflared\cert.pem`.

---

## Step 4: Create a named tunnel

```powershell
cloudflared tunnel create puffinzipai
```

This creates a tunnel and saves credentials to
`%USERPROFILE%\.cloudflared\<tunnel-id>.json`.

Note the **Tunnel ID** (a UUID like `a1b2c3d4-...`).

---

## Step 5: Route DNS

```powershell
cloudflared tunnel route dns puffinzipai stelliro.com
```

This creates a CNAME record in Cloudflare DNS:
`stelliro.com → <tunnel-id>.cfargotunnel.com`

---

## Step 6: Create tunnel config

Create `%USERPROFILE%\.cloudflared\config.yml`:

```yaml
tunnel: puffinzipai
credentials-file: C:\Users\<YOU>\.cloudflared\<tunnel-id>.json

ingress:
  - hostname: stelliro.com
    service: http://localhost:5001
  - service: http_status:404
```

Replace `<YOU>` with your Windows username and `<tunnel-id>` with the UUID from
step 4.

---

## Step 7: Test the tunnel

```powershell
cloudflared tunnel run puffinzipai
```

Then in another terminal, start PuffinZipAI:

```powershell
.\start.bat
```

Visit **https://stelliro.com/PuffinZipAI/login** — you should see the login page.

---

## Automatic Startup (Optional)

### Option A: start.bat handles it

`start.bat` auto-detects `cloudflared` and starts the tunnel in the background
before launching the server. Just run `start.bat` — both the tunnel and server
start together.

### Option B: Windows Service

Install cloudflared as a Windows service (runs at boot):

```powershell
cloudflared service install
```

Then `start.bat` only needs to start the Flask server.

### Option C: Task Scheduler

Create a scheduled task that runs `cloudflared tunnel run puffinzipai` at login.

---

## How It Works

| Layer | What happens |
|-------|-------------|
| **DNS** | `stelliro.com` → CNAME → `<tunnel-id>.cfargotunnel.com` |
| **Cloudflare Edge** | Terminates TLS, routes through tunnel |
| **cloudflared** | Outbound-only connection from your PC to Cloudflare |
| **PrefixMiddleware** | Strips `/PuffinZipAI` from request path |
| **ProxyFix** | Reads `X-Forwarded-*` headers from Cloudflare |
| **Flask** | Serves dashboard at `/`, login at `/login`, APIs at `/api/*` |
| **Landing page** | `index.html` at project root served at `stelliro.com/` |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "tunnel not found" | Run `cloudflared tunnel list` to verify |
| 502 Bad Gateway | PuffinZipAI server not running — start it first |
| Login page won't load | Check `custom_url` in `webui_credentials.json` matches your URL |
| ERR_NAME_NOT_RESOLVED | DNS not propagated yet — wait or check nameservers |
| "connection refused" | Ensure `public_access: false` is OK (cloudflared connects to 127.0.0.1) |
| Wrong URL case | Ensure `custom_url` path case matches the URL you visit |

---

## Quick Test (No Account Needed)

For a quick test without setting up DNS, use a temporary tunnel:

```powershell
cloudflared tunnel --url http://localhost:5001
```

This gives a random `https://xxxxx.trycloudflare.com` URL. The dashboard will be
at `https://xxxxx.trycloudflare.com/PuffinZipAI/login`. This URL changes every
time and is not your custom domain — it's just for testing.

---

## Cost

**Free.** Cloudflare Tunnels, DNS, and the free plan are all $0. No bandwidth
limits for tunnels.
