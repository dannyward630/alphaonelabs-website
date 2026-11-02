"""Tests for the EarthMind data-ingestion endpoint."""

import asyncio
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


try:
    import workers  # noqa: F401
except ModuleNotFoundError:
    workers = types.ModuleType("workers")

    class Response:
        def __init__(self, body, headers=None, status=200):
            self.body = body
            self.headers = headers or {}
            self.status = status

    class WorkerEntrypoint:
        pass

    workers.Response = Response
    workers.WorkerEntrypoint = WorkerEntrypoint
    sys.modules["workers"] = workers

from index import _ingest_reading


class FakeRequest:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class FakeStatement:
    def __init__(self):
        self.bindings = None

    def bind(self, *bindings):
        self.bindings = bindings
        return self

    async def run(self):
        return SimpleNamespace(meta=SimpleNamespace(last_row_id=42))


class FakeDatabase:
    def __init__(self):
        self.statement = FakeStatement()
        self.query = None

    def prepare(self, query):
        self.query = query
        return self.statement


class FakeEnvironment:
    def __init__(self):
        self.DB = FakeDatabase()


def valid_reading(**overrides):
    return {
        "source": "sensor-1",
        "source_type": "iot_sensor",
        "parameter": "rainfall",
        "value": 12.5,
        "timestamp": "2026-11-01T12:00:00Z",
        **overrides,
    }


def response_json(response):
    return json.loads(response.body)


def ingest(body, env):
    return asyncio.run(_ingest_reading(FakeRequest(body), env))


@pytest.mark.parametrize(("latitude", "longitude"), [(-90, -180), (90, 180)])
def test_ingest_accepts_zero_and_boundary_coordinates(latitude, longitude):
    env = FakeEnvironment()

    response = ingest(
        valid_reading(value=0, latitude=latitude, longitude=longitude), env
    )

    assert response.status == 201
    assert response_json(response) == {"id": 42, "status": "created"}
    assert env.DB.statement.bindings[3:7] == (
        float(latitude), float(longitude), "rainfall", 0.0
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "nan", "inf"])
def test_ingest_rejects_non_finite_values(value):
    env = FakeEnvironment()

    response = ingest(valid_reading(value=value), env)

    assert response.status == 400
    assert response_json(response) == {"error": "'value' must be finite"}
    assert env.DB.query is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latitude", -90.01),
        ("latitude", 90.01),
        ("longitude", -180.01),
        ("longitude", 180.01),
        ("latitude", float("nan")),
        ("longitude", float("inf")),
        ("latitude", "north"),
        ("longitude", True),
    ],
)
def test_ingest_rejects_invalid_coordinates(field, value):
    env = FakeEnvironment()

    response = ingest(valid_reading(**{field: value}), env)

    assert response.status == 400
    assert field in response_json(response)["error"]
    assert env.DB.query is None


@pytest.mark.parametrize(
    "body",
    [
        valid_reading(value=None),
        valid_reading(value=""),
        {key: value for key, value in valid_reading().items() if key != "value"},
    ],
)
def test_ingest_still_requires_a_present_value(body):
    env = FakeEnvironment()

    response = ingest(body, env)

    assert response.status == 400
    assert response_json(response) == {"error": "Missing required fields: value"}
    assert env.DB.query is None


def test_ingest_rejects_non_object_json():
    env = FakeEnvironment()

    response = ingest(["not", "an", "object"], env)

    assert response.status == 400
    assert response_json(response) == {"error": "JSON body must be an object"}
    assert env.DB.query is None


def test_ingest_allows_omitted_optional_coordinates():
    env = FakeEnvironment()

    response = ingest(valid_reading(), env)

    assert response.status == 201
    assert env.DB.statement.bindings[3:5] == (None, None)
