from __future__ import annotations

import time
from collections.abc import Callable

import requests


def request_with_retry(
    url: str,
    *,
    method: str = "GET",
    max_attempts: int = 5,
    timeout: int = 600,
    stream: bool = False,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """Exécute une requête HTTP avec backoff exponentiel sur 429 et 5xx."""
    for attempt in range(max_attempts):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                timeout=timeout,
                stream=stream,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                if attempt + 1 < max_attempts:
                    print(f"HTTP 429: nouvelle tentative dans {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue
            if response.status_code >= 500:
                wait_seconds = 2**attempt
                if attempt + 1 < max_attempts:
                    print(f"HTTP {response.status_code}: nouvelle tentative dans {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt + 1 >= max_attempts:
                raise
            wait_seconds = 2**attempt
            print(f"Erreur réseau: nouvelle tentative dans {wait_seconds}s...")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Échec après {max_attempts} tentatives pour {url}")
