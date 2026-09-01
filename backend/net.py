"""
Network configuration for Assemblage.

Certificate verification is ON by default. It can be disabled for networks that
run TLS inspection proxies (schools, corporate wifi) by setting:

    ASSEMBLAGE_INSECURE_SSL=1

This used to be unconditional, which meant every user of the tool downloaded
model weights and images with verification disabled in order to work around one
developer's school wifi. Opt-in is the correct default.
"""

import os

import aiohttp

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

HEADERS = {"User-Agent": BROWSER_UA}


def insecure_ssl_enabled() -> bool:
    return os.environ.get("ASSEMBLAGE_INSECURE_SSL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def make_connector() -> aiohttp.TCPConnector:
    """Builds a TCP connector honouring the insecure-SSL opt-in."""
    if insecure_ssl_enabled():
        return aiohttp.TCPConnector(ssl=False, limit=32)
    return aiohttp.TCPConnector(limit=32)


def apply_ssl_workarounds() -> None:
    """
    Installs the system trust store, and only disables verification if the user
    explicitly asked for it. Safe to call when truststore is not installed.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
        print("[Net] System trust store injected.")
    except ImportError:
        print("[Net] truststore not installed; using default certifi bundle.")

    if insecure_ssl_enabled():
        import ssl

        os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
        os.environ["CURL_CA_BUNDLE"] = ""
        ssl._create_default_https_context = ssl._create_unverified_context
        print(
            "[Net] WARNING: ASSEMBLAGE_INSECURE_SSL is set. Certificate "
            "verification is disabled for this process."
        )
