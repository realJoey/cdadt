"""Assembled aircraft: named discipline sets ready to size.

A discipline set is a modeling decision -- which physics is used for each domain -- and
this package is where those decisions are written down as named, importable functions
rather than repeated in every run script.

:mod:`cdadt.aircraft.jet_transport`
    The tube-and-wing jet transport set: empirical drag and weight buildups, a scalable
    turbofan deck, tail volume coefficient sizing.
"""

from cdadt.aircraft.jet_transport import jet_transport_disciplines

__all__ = ["jet_transport_disciplines"]
