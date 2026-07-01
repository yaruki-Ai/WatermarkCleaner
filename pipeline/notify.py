"""Notifications de fin de traitement — email (Gmail) ou push (ntfy.sh). Optionnel."""
from __future__ import annotations

import smtplib
import ssl
import urllib.request
from email.message import EmailMessage


def send_push(topic: str, title: str, message: str, success: bool = True) -> None:
    """Envoie une notification push via ntfy.sh (aucun identifiant requis)."""
    # Le titre part dans un header HTTP -> on le garde en ASCII pur.
    safe_title = title.encode("ascii", "ignore").decode("ascii") or "WatermarkCleaner"
    req = urllib.request.Request(
        f"https://ntfy.sh/{topic.strip()}",
        data=message.encode("utf-8"),
        headers={
            "Title": safe_title,
            "Priority": "default",
            "Tags": "white_check_mark" if success else "x",
        },
        method="POST",
    )
    urllib.request.urlopen(req, timeout=15)


def try_push(topic: str, title: str, message: str, success: bool = True) -> None:
    """Variante non-bloquante : un échec d'envoi ne casse jamais le traitement."""
    if not topic:
        return
    try:
        send_push(topic, title, message, success)
        print(f"🔔 Notification push envoyée (salon ntfy : {topic.strip()})")
    except Exception as e:
        print(f"⚠️ Notification push échouée : {e}")


def send_email(gmail_address: str, app_password: str, subject: str, body: str) -> None:
    """Envoie un email via Gmail (nécessite un 'mot de passe d'application')."""
    msg = EmailMessage()
    msg["From"] = gmail_address
    msg["To"] = gmail_address
    msg["Subject"] = subject
    msg.set_content(body)

    ctx = ssl.create_default_context()
    # timeout pour ne jamais bloquer le pipeline si le serveur SMTP ne répond pas.
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=20) as server:
        server.login(gmail_address, app_password.replace(" ", ""))
        server.send_message(msg)


def try_send(gmail_address: str, app_password: str, subject: str, body: str) -> None:
    """Variante non-bloquante : un échec d'email ne casse jamais le traitement."""
    if not gmail_address or not app_password:
        return
    try:
        send_email(gmail_address, app_password, subject, body)
        print(f"📧 Email de notification envoyé à {gmail_address}")
    except Exception as e:
        print(f"⚠️ Envoi de l'email échoué : {e}")
