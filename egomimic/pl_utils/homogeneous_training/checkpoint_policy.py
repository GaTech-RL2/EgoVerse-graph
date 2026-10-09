def policy(original, mode, family, counters):
    identities = {}
    def wrapped(fn, *args, **kw):
        assert kw.get("use_reentrant") is False
        if fn not in identities: identities[fn] = len(identities)
        bypass = (mode == "dit-off" and family == "dit") or (mode == "decoder-off" and family == "decoder") or (mode == "dit-half" and family == "dit" and identities[fn] % 2 == 0)
        counters[family + ("/direct" if bypass else "/checkpoint")] = counters.get(family + ("/direct" if bypass else "/checkpoint"), 0) + 1
        if not bypass: return original(fn, *args, **kw)
        assert set(kw) <= {"use_reentrant"}, "do not silently drop checkpoint options"
        return fn(*args)
    return wrapped
