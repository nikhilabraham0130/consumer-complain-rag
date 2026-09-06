"""
DNS Fallback Patch for restricted network environments (e.g. campus Wi-Fi or OpenDNS).

Detects if api.deepseek.com is being sinkholed to a network block page (e.g. 146.112.x.x)
and transparently routes to the verified AWS CloudFront edge IP for DeepSeek.
"""

import logging
import socket

logger = logging.getLogger(__name__)

_DEEPSEEK_CLOUDFRONT_IPS = ["3.173.21.63", "18.160.10.124", "18.160.10.18"]
_SINKHOLE_PREFIXES = ("146.112.", "0.0.0.0")

_orig_getaddrinfo = socket.getaddrinfo


def _patched_getaddrinfo(host, port, *args, **kwargs):
    if host == "api.deepseek.com":
        try:
            res = _orig_getaddrinfo(host, port, *args, **kwargs)
            # If resolved to Cisco Umbrella / OpenDNS sinkhole, reroute to verified edge IP
            if any(r[4][0].startswith(_SINKHOLE_PREFIXES) for r in res):
                return _orig_getaddrinfo(_DEEPSEEK_CLOUDFRONT_IPS[0], port, *args, **kwargs)
            return res
        except Exception:
            return _orig_getaddrinfo(_DEEPSEEK_CLOUDFRONT_IPS[0], port, *args, **kwargs)
    return _orig_getaddrinfo(host, port, *args, **kwargs)


def apply_dns_patch() -> None:
    """Applies the safe DNS getaddrinfo patch."""
    socket.getaddrinfo = _patched_getaddrinfo
