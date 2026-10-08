"""Isolated RNG recipe capture/replay; no reseeding or augmentation changes.

Capture must observe a native source-major forward, then a candidate draws the
same calls in that order and consumes them by source/boundary. Unknown RNG,
changed call shape/dtype/device/order, explicit generators and output buffers
fail closed. This is a test mechanism, not production adoption.
"""
from contextlib import contextmanager
import torch


def rng_state():
    return (torch.get_rng_state().clone(),
            [x.clone() for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else [])


def same_rng(left, right):
    return torch.equal(left[0], right[0]) and len(left[1]) == len(right[1]) and all(
        torch.equal(a, b) for a, b in zip(left[1], right[1]))


def set_rng(state):
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def descriptor(value):
    if torch.is_tensor(value):
        assert value.layout == torch.strided
        return ("tensor", tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
    if isinstance(value, (tuple, list)):
        return ("sequence", type(value), tuple(descriptor(v) for v in value))
    assert value is None or isinstance(value, (int, float, str, bool, torch.dtype, torch.device, torch.layout, torch.memory_format))
    return ("scalar", value)


def materialize(spec):
    if spec[0] == "tensor":
        _, shape, stride, dtype, device = spec
        return torch.empty_strided(shape, stride, dtype=dtype, device=device)
    if spec[0] == "sequence":
        return spec[1](materialize(v) for v in spec[2])
    return spec[1]


class NativeDrawReplay:
    names = ("rand", "randn", "randint", "rand_like", "randn_like")

    def __init__(self):
        self.recipe = []
        self.boundary = None
        self.originals = None
        self.values = {}
        self.positions = {}

    @contextmanager
    def at(self, source, phase):
        before = self.boundary
        self.boundary = (source, phase)
        try:
            yield
        finally:
            self.boundary = before

    @contextmanager
    def functions(self, mode):
        assert self.originals is None
        self.originals = {name: getattr(torch, name) for name in self.names}
        def wrap(name):
            def call(*args, **kwargs):
                assert self.boundary is not None, "random draw outside native source boundary"
                assert kwargs.get("generator") is None and kwargs.get("out") is None
                signature = (name, descriptor(args), tuple(sorted((k, descriptor(v)) for k, v in kwargs.items())))
                if mode == "capture":
                    value = self.originals[name](*args, **kwargs)
                    self.recipe.append((self.boundary, signature))
                    return value
                assert mode == "replay"
                index = self.positions.get(self.boundary, 0)
                queue = self.values.get(self.boundary, [])
                assert index < len(queue), "extra random draw"
                expected, value = queue[index]
                assert signature == expected, "random draw contract changed"
                self.positions[self.boundary] = index + 1
                return value
            return call
        for name in self.names:
            setattr(torch, name, wrap(name))
        try:
            yield
        finally:
            for name, original in self.originals.items():
                setattr(torch, name, original)
            self.originals = None

    def draw_native_order(self):
        assert self.originals is None and self.recipe
        self.values = {}
        self.positions = {}
        for boundary, signature in self.recipe:
            name, args, kwargs = signature
            value = getattr(torch, name)(*materialize(args),
                **{key: materialize(spec) for key, spec in kwargs})
            self.values.setdefault(boundary, []).append((signature, value))
        return rng_state()

    def complete(self, expected_rng):
        assert all(self.positions.get(key, 0) == len(values)
                   for key, values in self.values.items()), "missing random draw"
        assert same_rng(expected_rng, rng_state()), "unrecorded RNG operation"
        self.values.clear()
        self.positions.clear()


def native_split_crop(module, images, sources, rows, replay, original):
    """Run the unchanged native crop per original source, then concatenate."""
    assert len(sources) == len(rows) == 2 and sum(rows) == len(images)
    assert module.crop_scope == "frame", "episode grouping requires separate proof"
    parts = images.split(rows, dim=0)
    outputs = []
    for source, part in zip(sources, parts):
        with replay.at(source, "prefix"):
            outputs.append(original(module, part))
    return torch.cat(outputs, dim=0)
