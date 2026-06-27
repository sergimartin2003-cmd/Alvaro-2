"""Sistema de alertas multi-canal (email, SMS, Slack, Discord, Telegram).

Cada canal usa solo librerías estándar o ``requests``. Twilio se importa de
forma perezosa. Los canales no configurados se omiten con un resultado claro.
"""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class MultiChannelAlertSystem:
    def __init__(self, config: dict) -> None:
        self.config = config or {}
        self.alert_history: list[dict] = []

    def send_alert(self, alert_data: dict, channels: list[str] | None = None) -> dict:
        channels = channels or self.config.get("default_channels", ["email"])
        results = {}
        dispatch = {
            "email": self._send_email,
            "sms": self._send_sms,
            "slack": self._send_slack,
            "discord": self._send_discord,
            "telegram": self._send_telegram,
        }
        for ch in channels:
            fn = dispatch.get(ch)
            if fn is None:
                results[ch] = {"success": False, "error": "canal desconocido"}
                continue
            try:
                results[ch] = fn(alert_data)
            except Exception as exc:  # pragma: no cover - depende de red
                results[ch] = {"success": False, "error": str(exc)}

        self.alert_history.append({"alert": alert_data, "channels": channels, "results": results})
        return results

    def _send_email(self, a: dict) -> dict:
        cfg = self.config["smtp"]
        msg = MIMEMultipart()
        msg["From"] = cfg["from_email"]
        msg["To"] = cfg["to_email"]
        msg["Subject"] = f"[{a.get('priority', 'INFO')}] {a.get('title', '')}"
        body = (
            f"<h2>{a.get('title','')}</h2>"
            f"<p><b>Asset:</b> {a.get('asset')}</p>"
            f"<p><b>Signal:</b> {a.get('signal')}</p>"
            f"<p><b>Confidence:</b> {a.get('confidence', 0):.2%}</p>"
            f"<hr><p>{a.get('message','')}</p>"
        )
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP(cfg["server"], cfg["port"]) as server:
            server.starttls()
            server.login(cfg["username"], cfg["password"])
            server.send_message(msg)
        return {"success": True}

    def _send_sms(self, a: dict) -> dict:
        from twilio.rest import Client

        cfg = self.config["twilio"]
        client = Client(cfg["account_sid"], cfg["auth_token"])
        msg = client.messages.create(
            body=f"{a.get('priority')}: {a.get('title')} {a.get('signal')} {a.get('asset')}",
            from_=cfg["from_number"],
            to=cfg["to_number"],
        )
        return {"success": True, "message_sid": msg.sid}

    def _send_slack(self, a: dict) -> dict:
        import requests

        colors = {"HIGH": "#ff0000", "MEDIUM": "#ffa500", "LOW": "#00ff00"}
        payload = {
            "attachments": [
                {
                    "color": colors.get(a.get("priority"), "#808080"),
                    "title": a.get("title"),
                    "text": a.get("message"),
                    "fields": [
                        {"title": "Asset", "value": str(a.get("asset")), "short": True},
                        {"title": "Signal", "value": str(a.get("signal")), "short": True},
                    ],
                }
            ]
        }
        r = requests.post(self.config["slack"]["webhook_url"], json=payload, timeout=10)
        return {"success": r.status_code == 200}

    def _send_discord(self, a: dict) -> dict:
        import requests

        colors = {"HIGH": 0xFF0000, "MEDIUM": 0xFFA500, "LOW": 0x00FF00}
        payload = {
            "embeds": [
                {
                    "title": a.get("title"),
                    "description": a.get("message"),
                    "color": colors.get(a.get("priority"), 0x808080),
                }
            ]
        }
        r = requests.post(self.config["discord"]["webhook_url"], json=payload, timeout=10)
        return {"success": r.status_code in (200, 204)}

    def _send_telegram(self, a: dict) -> dict:
        import requests

        cfg = self.config["telegram"]
        text = (
            f"🚨 *{a.get('priority')} ALERT*\n\n*{a.get('title')}*\n"
            f"Asset: {a.get('asset')}\nSignal: {a.get('signal')}\n\n{a.get('message','')}"
        )
        url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
        r = requests.post(
            url, data={"chat_id": cfg["chat_id"], "text": text, "parse_mode": "Markdown"}, timeout=10
        )
        return {"success": r.status_code == 200}

    def prioritize_alerts(self, alerts: list[dict]) -> list[dict]:
        rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        ordered = sorted(
            alerts,
            key=lambda x: (rank.get(x.get("priority"), 0), x.get("confidence", 0)),
            reverse=True,
        )
        return ordered[: self.config.get("max_alerts_per_hour", 10)]
