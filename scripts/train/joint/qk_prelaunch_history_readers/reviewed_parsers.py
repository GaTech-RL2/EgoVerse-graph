"""Fixed service parsers for exact historical source hashes and call roles.

The original Python files are immutable evidence data. Their bytes are checked
here, but are never compiled, imported, or otherwise executed. Every returned
module is an explicit relative import from this maintained service package.
"""
import hashlib


PARSER_BINDINGS = {
    ('08d5bd354a0eb81dad4b9664037e711ac4f7ecc2b57355db2e4154571c49697c',
     '_exact_prelaunch_native_guard'): 'native_guard_v1',
    ('bc278ad53d6701d69b2df0b2df37a1286d75b1abd9be5d6c726b1c35539254f0',
     '_unchanged_prior_prelaunch'): 'prelaunch_retry',
    ('f6df73a12f7d4401376fd26929ee2678d46c50ed52c48be95c7ca50f0a36f71f',
     '_exact_original_native'): 'native_guard_v2',
    ('d5ddc74a794bdd8f1417c0ae9081d4666cb4d9e1149e5969f84e18914b4232e5',
     '_full_v3_exact_native'): 'native_guard_v3',
    ('7949af3f62b75aecf4a887b62a9ab9f01d284864e9ca6c52725f7e04d340ce5e',
     '_full_v3_exact_early_native'): 'early_native_parser_v3',
    ('ec932acc0497e841282028ef0db94d9e6439ce218bd7f3b9e62aa181b4ffed53',
     '_exact_full_v4_native_held'): 'native_guard_v4',
}


def reviewed_parser(raw, digest, name):
    """Authenticate old source as data, then select a fixed service parser."""
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Unreviewed parser: ' + name)
    parser = PARSER_BINDINGS.get((digest, name))
    if parser == 'native_guard_v1':
        from . import native_guard_v1
        return native_guard_v1
    if parser == 'prelaunch_retry':
        from . import prelaunch_retry
        return prelaunch_retry
    if parser == 'native_guard_v2':
        from . import native_guard_v2
        return native_guard_v2
    if parser == 'native_guard_v3':
        from . import native_guard_v3
        return native_guard_v3
    if parser == 'early_native_parser_v3':
        from . import early_native_parser_v3
        return early_native_parser_v3
    if parser == 'native_guard_v4':
        from . import native_guard_v4
        return native_guard_v4
    raise ValueError('Unsupported reviewed historical parser role: ' + name)
