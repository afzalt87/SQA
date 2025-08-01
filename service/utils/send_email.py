import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from service.utils.config import get_logger

logger = get_logger()


def send_email(subject, body, to_emails, log_path=None):
    from_email = os.getenv("EMAIL_FROM")
    from_password = os.getenv("EMAIL_PASSWORD")
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    try:
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_emails
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Attach log file if requested
        if log_path and os.path.exists(log_path):
            with open(log_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{os.path.basename(log_path)}"',
                )
                msg.attach(part)

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_emails, text)
        server.quit()
        logger.info(f"Email sent to {to_emails}")
    except Exception as e:
        logger.exception(f"Failed to send email: {e}")
        raise
