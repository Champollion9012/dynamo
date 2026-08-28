# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The push-egress token-frame fast path, through the real bridge.

``push_egress.rs`` encodes the steady-state TRT-LLM decode frame -- a dict of
exactly ``token_ids`` then ``index`` -- straight to msgpack, skipping serde and
pythonize entirely.  The wire layout is pinned byte-for-byte against the generic
encoder by Rust unit tests in ``python_payload.rs``; what those cannot cover is
the half that needs libpython, which the extension-module build does not link
into ``cargo test``:

* the shape and type gate that decides whether a dict is eligible at all, and
* the reused encode buffer, including the rewind after a frame is rejected
  partway through writing.

Both are exercised here against the built extension, end to end over the request
plane, by asserting that whatever the handler pushes is exactly what the client
receives.  A gate that wrongly accepted a frame would corrupt its value, and a
rewind that left bytes behind would corrupt the frame after it -- which is why
:func:`test_mixed_eligible_and_ineligible_frames_in_one_stream` interleaves the
two kinds rather than testing them in separate streams.

Comparison is type-strict (see :func:`_identical`).  Plain ``==`` would pass on
the most interesting failure: ``True == 1`` in Python, so a ``bool`` wrongly
encoded as the integer ``1`` compares equal to what was sent.
"""

import asyncio
import contextlib

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


# ---------------------------------------------------------------------------
# Frame cases: what the handler pushes, and what the client must get back.
# ---------------------------------------------------------------------------
#
# `sent` goes to `response_sender.send()` verbatim; `expected` is what the frame
# must decode to at the client. They differ only where the wire model has no
# equivalent of the Python type -- a tuple has no msgpack form of its own and
# comes back as a list on either path.

# Eligible: the fast path must take these, and they must survive it unchanged.
_ELIGIBLE = [
    ({"token_ids": [5], "index": 0}, None),  # the steady-state frame
    ({"token_ids": [], "index": 0}, None),  # engine produced no new tokens
    ({"token_ids": [1, 2, 3], "index": 2}, None),  # multi-token chunk
    ({"token_ids": list(range(40)), "index": 0}, None),  # past the fixarray limit
    ({"token_ids": [0, 127, 128, 255, 256, 65535, 65536], "index": 1}, None),
    ({"token_ids": [4294967295], "index": 4294967295}, None),  # u32 bounds
]

# Ineligible: each trips a different clause of the gate and must fall back to
# the generic encoder with its value intact.
_INELIGIBLE = [
    # bool is an int subclass, and pythonize tests PyBool before PyInt, so the
    # generic path encodes these as msgpack bools. The fast path must not
    # quietly turn them into 1/0.
    ({"token_ids": [True], "index": 0}, None),
    ({"token_ids": [1], "index": True}, None),
    # Negative and past-u64 values encode as sint/bin generically.
    ({"token_ids": [-1], "index": 0}, None),
    ({"token_ids": [1], "index": -1}, None),
    ({"token_ids": [2**64], "index": 0}, None),
    # Not a list.
    ({"token_ids": (1, 2), "index": 0}, {"token_ids": [1, 2], "index": 0}),
    ({"token_ids": 5, "index": 0}, None),
    # Not exactly two keys, or not these two, or not in this order.
    ({"token_ids": [1], "index": 0, "finish_reason": "stop"}, None),
    ({"token_ids": [1]}, None),
    ({"index": 0, "token_ids": [1]}, None),
    ({"text": "hello", "index": 0}, None),
    # A non-int in the list, rejected partway through writing the frame.
    ({"token_ids": [1, "two", 3], "index": 0}, None),
    ({"token_ids": [1, None], "index": 0}, None),
    ({"token_ids": [1, 2.5], "index": 0}, None),
    # The annotated envelope must still be interpreted, not encoded as data.
    (
        {"_dynamo_annotated": True, "data": {"token_ids": [1], "index": 0}},
        {"token_ids": [1], "index": 0},
    ),
    # Not a dict at all.
    ("just a string", None),
    ([1, 2, 3], None),
]

_CASES = {
    "eligible": _ELIGIBLE,
    "ineligible": _INELIGIBLE,
    # Interleaved, so a rejected frame's rewind is followed by a frame that
    # would inherit its bytes if the rewind were wrong.
    "mixed": [
        frame
        for pair in zip(_ELIGIBLE, _INELIGIBLE[: len(_ELIGIBLE)])
        for frame in pair
    ],
    # One long run of the fast path, so the encode buffer is refilled several
    # times rather than serving every frame from its first chunk.
    "many": [({"token_ids": [i], "index": 0}, None) for i in range(512)],
}


def _sent(case: str):
    return [sent for sent, _ in _CASES[case]]


def _expected(case: str):
    return [sent if expected is None else expected for sent, expected in _CASES[case]]


def _identical(actual, expected) -> bool:
    """Equality that does not conflate ``bool`` with ``int``.

    ``True == 1`` and ``False == 0`` in Python, so plain ``==`` cannot tell a
    bool that survived correctly from one the fast path flattened to an integer
    -- the exact regression the gate's exact-type check exists to prevent.
    """
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _identical(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _identical(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


# ---------------------------------------------------------------------------
# Handler + endpoint
# ---------------------------------------------------------------------------


async def _push_handler(request, context, response_sender=None):
    """Push every frame of the requested case, then close.

    Declaring ``response_sender`` is what makes Rust select the push engine
    (``push_egress.rs::handler_supports_push``).  It must still be an async
    generator that yields nothing on the push path, and Rust advances it exactly
    once per request.

    The pull arm is reached by the health check and in-process callers, which
    pass no sender; it must keep working, so it yields the same frames instead.
    """
    frames = _sent(request["case"])

    if response_sender is None:
        for frame in frames:
            yield frame
        return

    for frame in frames:
        response_sender.send(frame)
    response_sender.close()


@pytest.fixture
async def push_client(runtime):
    endpoint = runtime.endpoint("push-egress-token-frame.backend.generate")
    server_task = asyncio.ensure_future(
        endpoint.serve_endpoint(
            _push_handler,
            health_check_payload={"case": "eligible"},
        )
    )
    client = await endpoint.client()
    try:
        await client.wait_for_instances()
        yield client
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


async def _round_trip(client, case: str):
    stream = await client.generate({"case": case})
    return [response.data() async for response in stream]


def _assert_round_trip(actual, case: str):
    expected = _expected(case)
    count_message = f"case {case!r}: expected {len(expected)} frames, got {len(actual)}"
    assert len(actual) == len(expected), count_message
    for position, (got, want) in enumerate(zip(actual, expected)):
        assert _identical(got, want), (
            f"case {case!r}: frame {position} came back as {got!r} "
            f"(type {type(got).__name__}), expected {want!r}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_eligible_frames_survive_the_fast_path(push_client):
    """Frames the fast path accepts must reach the client unchanged.

    Covers the empty token list, multi-token chunks, and the list lengths and
    integer magnitudes that cross msgpack's fixarray and uint marker
    boundaries.
    """
    _assert_round_trip(await _round_trip(push_client, "eligible"), "eligible")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_ineligible_frames_fall_back_intact(push_client):
    """Every near-miss must fall back to the generic encoder losing nothing.

    Each entry trips a different clause of the gate: a bool where an int is
    expected, a negative or oversized integer, a non-list, the wrong key set or
    order, a non-int inside the list, an annotated envelope, and non-dicts.
    """
    _assert_round_trip(await _round_trip(push_client, "ineligible"), "ineligible")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_mixed_eligible_and_ineligible_frames_in_one_stream(push_client):
    """The reused encode buffer must not leak bytes between frames.

    A rejected frame can be abandoned partway through writing, and the buffer it
    was being written into is the same one the next frame uses. If the rewind
    were wrong, the following frame would carry the abandoned prefix and fail to
    decode -- which only shows up when the two kinds are interleaved.
    """
    _assert_round_trip(await _round_trip(push_client, "mixed"), "mixed")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_long_fast_path_run_refills_the_encode_buffer(push_client):
    """A run long enough to exhaust and refill the buffer's chunk repeatedly.

    Frames are cut out of a shared chunk with ``split``, so this covers the
    transition to a fresh chunk -- and, because the frames are numbered, proves
    none is duplicated or dropped across it.
    """
    _assert_round_trip(await _round_trip(push_client, "many"), "many")
