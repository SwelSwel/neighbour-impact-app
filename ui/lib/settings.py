"""UI settings: where the API is, and the labels used on every page. Edit labels and units here."""
from __future__ import annotations

import os

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", API_URL).rstrip("/")   # address a browser can reach (for /docs links)
API_KEY = os.getenv("API_KEY", "")
TIMEOUT_S = float(os.getenv("API_TIMEOUT_S", "60"))

RAW8 = ["Traffic", "Connected_Users", "Throughput", "Latency", "Packet_Loss", "Alarm_Count", "Load", "Capacity"]
FAILED_KPIS = ["Traffic", "Connected_Users", "Load", "Capacity", "Throughput"]
LAST_WEEK = ["Latency", "Utilization", "Packet_Loss", "Traffic", "Load"]
TARGETS = {"Latency": "ms", "Utilization": "%", "Packet_Loss": "%"}
READINGS = [("now", "Last reading before the outage"), ("h3_before", "3 hours earlier"),
            ("h6_before", "6 hours earlier")]

WARNING_COLOURS = {"severe": "#F8D0CD", "degraded": "#FDECC8"}
