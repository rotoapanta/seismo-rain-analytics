#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for seismo-rain-analytics project.
"""

from datetime import date, time as dtime, datetime

CM_PER_IN = 2.54
A3_SIZE_CM = (42.0, 29.7)
A4_SIZE_CM = (29.7, 21.0)

def cm_to_in(x: float) -> float:
    return float(x) / CM_PER_IN

def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def parse_time(s: str) -> dtime:
    return datetime.strptime(s, "%H:%M").time()

def in_timerange(t: dtime, t0: dtime | None, t1: dtime | None) -> bool:
    if t0 is None or t1 is None:
        return True
    # rango inclusivo [t0, t1]
    return t0 <= t <= t1
