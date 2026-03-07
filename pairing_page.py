"""Helpers for building the web-based pairing page."""

from __future__ import annotations

import base64
import io

import qrcode


def make_qr_png_base64(qr_url: str) -> str:
    """Create a base64-encoded PNG QR image from a pairing URL."""
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(qr_url)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def render_pairing_page_html(
    *,
    qr_base64: str,
    local_url: str,
    pair_token: str,
    expires_in_seconds: int,
    pairing_code: str = "",
) -> str:
    """Render pairing page HTML."""
    expires_minutes = max(1, expires_in_seconds // 60)
    # Format pairing code with dash for readability (e.g., "123-456")
    formatted_code = f"{pairing_code[:3]}-{pairing_code[3:]}" if len(pairing_code) == 6 else pairing_code
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Code Bridge - QR Pairing</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                background: #f5f5f5;
            }}
            .container {{
                text-align: center;
                background: white;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.1);
                max-width: 400px;
            }}
            h1 {{
                color: #333;
                margin-bottom: 8px;
            }}
            .subtitle {{
                color: #666;
                margin-bottom: 24px;
            }}
            .qr-code {{
                margin: 24px 0;
            }}
            .qr-code img {{
                width: 250px;
                height: 250px;
            }}
            .divider {{
                display: flex;
                align-items: center;
                margin: 24px 0;
                color: #999;
            }}
            .divider::before, .divider::after {{
                content: "";
                flex: 1;
                border-bottom: 1px solid #ddd;
            }}
            .divider span {{
                padding: 0 16px;
                font-size: 14px;
            }}
            .pairing-code {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-size: 32px;
                font-weight: bold;
                letter-spacing: 4px;
                padding: 16px 24px;
                border-radius: 12px;
                margin: 16px 0;
                font-family: 'SF Mono', Monaco, monospace;
            }}
            .code-label {{
                color: #666;
                font-size: 14px;
                margin-bottom: 8px;
            }}
            .info {{
                color: #888;
                font-size: 14px;
            }}
            .server-url {{
                background: #f0f0f0;
                padding: 8px 16px;
                border-radius: 8px;
                font-family: monospace;
                margin: 16px 0;
                font-size: 14px;
            }}
            .expires {{
                color: #e67e22;
                font-weight: 500;
            }}
            .success {{
                display: none;
            }}
            .success.show {{
                display: block;
            }}
            .success h2 {{
                color: #27ae60;
                margin-bottom: 16px;
            }}
            .success .checkmark {{
                font-size: 64px;
                margin-bottom: 16px;
            }}
            .qr-section.hide {{
                display: none;
            }}
            .buttons {{
                margin-top: 24px;
                display: flex;
                gap: 12px;
                justify-content: center;
            }}
            .btn {{
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                text-decoration: none;
                transition: all 0.2s;
            }}
            .btn-refresh {{
                background: #3498db;
                color: white;
                border: none;
            }}
            .btn-refresh:hover {{
                background: #2980b9;
            }}
            .btn-dashboard {{
                background: white;
                color: #333;
                border: 1px solid #ddd;
            }}
            .btn-dashboard:hover {{
                background: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Code Bridge</h1>
            <div class="qr-section" id="qrSection">
                <p class="subtitle">Pair with the mobile app</p>

                <!-- 6-digit Code Section -->
                <div class="code-label">Enter this code in the app</div>
                <div class="pairing-code">{formatted_code}</div>
                <div class="server-url">{local_url}</div>

                <div class="divider"><span>or scan QR code</span></div>

                <!-- QR Code Section -->
                <div class="qr-code">
                    <img src="data:image/png;base64,{qr_base64}" alt="QR Code">
                </div>

                <p class="expires" id="expiresText">Expires in {expires_minutes} minutes</p>

                <div class="buttons">
                    <button class="btn btn-refresh" onclick="location.reload()">Refresh</button>
                    <a href="/dashboard" class="btn btn-dashboard">Go to Dashboard</a>
                </div>
            </div>
            <div class="success" id="successSection">
                <div class="checkmark">✅</div>
                <h2>Pairing Complete!</h2>
                <p class="subtitle">You can close this page</p>
            </div>
        </div>
        <script>
            const token = "{pair_token}";
            let remainingSeconds = {expires_in_seconds};
            const expiresText = document.getElementById('expiresText');

            // Countdown timer
            function updateCountdown() {{
                if (remainingSeconds <= 0) {{
                    expiresText.textContent = 'Expired - Please refresh';
                    expiresText.style.color = '#e74c3c';
                    return;
                }}
                const minutes = Math.floor(remainingSeconds / 60);
                const seconds = remainingSeconds % 60;
                if (minutes > 0) {{
                    expiresText.textContent = `Expires in ${{minutes}}m ${{seconds}}s`;
                }} else {{
                    expiresText.textContent = `Expires in ${{seconds}}s`;
                }}
                remainingSeconds--;
            }}
            updateCountdown();
            setInterval(updateCountdown, 1000);

            // Token status check
            const checkInterval = setInterval(async () => {{
                try {{
                    const res = await fetch(`/api/pair/token-status/${{token}}`);
                    const data = await res.json();
                    if (data.used) {{
                        clearInterval(checkInterval);
                        document.getElementById('qrSection').classList.add('hide');
                        document.getElementById('successSection').classList.add('show');
                        setTimeout(() => window.close(), 2000);
                    }}
                }} catch (e) {{
                    // Ignore errors
                }}
            }}, 1000);
        </script>
    </body>
    </html>
    """
