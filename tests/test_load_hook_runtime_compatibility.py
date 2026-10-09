from egomimic.utils.runtime_compatibility import register_module_load_pre_hook


def test_public_module_aware_hook_api_remains_primary():
    class Modern:
        def register_load_state_dict_pre_hook(self, hook):
            self.hook = hook
            return "public-handle"

    module = Modern()
    hook = object()
    assert register_module_load_pre_hook(module, hook) == "public-handle"
    assert module.hook is hook


def test_legacy_api_explicitly_requests_module_argument():
    class Legacy:
        def _register_load_state_dict_pre_hook(self, hook, *, with_module):
            self.binding = hook, with_module
            return "legacy-handle"

    module = Legacy()
    hook = object()
    assert register_module_load_pre_hook(module, hook) == "legacy-handle"
    assert module.binding == (hook, True)


def test_unsupported_api_fails_closed():
    import pytest

    with pytest.raises(TypeError, match="module-aware"):
        register_module_load_pre_hook(object(), object())
