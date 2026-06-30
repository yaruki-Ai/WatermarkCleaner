"""Notification par email (Gmail SMTP) en fin de traitement — optionnel."""
from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage


def send_email(gmail_address: str, app_password: str, subject: str, body: str) -> None:
    """Envoie un email via Gmail (nécessite un 'mot de passe d'application')."""
    msg = EmailMessage()
    msg["From"] = gmail_address
    msg["To"] = gmail_address
    msg["Subject"] = subject
    msg.set_content(body)

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
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
