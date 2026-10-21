"""Режим gateway: апдейты приходят от шлюза lab_manager по HTTP с общим секретом."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from aiohttp.test_utils import TestClient, TestServer

from telepathy.bot.gateway import SECRET_HEADER, UPDATE_PATH, create_gateway_app
from telepathy.db import Database
from tests.bot.fake_telegram import message_update
from tests.bot.harness import ANYA, App, make_app

SECRET = "s3cret-gateway-token"  # noqa: S105 — тестовый


@pytest.fixture
async def app(db: Database) -> App:
    return make_app(db)


async def serve(app: App, *, background: bool) -> TestClient:
    web_app = create_gateway_app(app.dp, app.bot, secret=SECRET, background=background)
    client = TestClient(TestServer(web_app))
    await client.start_server()
    return client


@pytest.fixture
async def client(app: App) -> AsyncIterator[TestClient]:
    client = await serve(app, background=False)
    yield client
    await client.close()


def body(text: str) -> str:
    # Так шлюз сериализует апдейт: имена полей Bot API, без пустых значений.
    return message_update(ANYA, text).model_dump_json(by_alias=True, exclude_none=True)


async def test_update_with_secret_is_handled(client: TestClient, app: App) -> None:
    response = await client.post(
        UPDATE_PATH,
        data=body("/start"),
        headers={SECRET_HEADER: SECRET, "Content-Type": "application/json"},
    )

    assert response.status == 200
    assert "Телепатия" in app.tg.last(ANYA.id)


@pytest.mark.parametrize("headers", [{}, {SECRET_HEADER: "wrong"}])
async def test_bad_secret_is_rejected(
    client: TestClient, app: App, headers: dict[str, str]
) -> None:
    response = await client.post(UPDATE_PATH, data=body("/start"), headers=headers)

    assert response.status == 401
    assert app.tg.requests == []


async def test_background_mode_replies_before_handling(app: App) -> None:
    client = await serve(app, background=True)
    try:
        response = await client.post(
            UPDATE_PATH, data=body("/start"), headers={SECRET_HEADER: SECRET}
        )
        assert response.status == 200
        for _ in range(100):
            if app.tg.requests:
                break
            await asyncio.sleep(0.01)
    finally:
        await client.close()

    assert "Телепатия" in app.tg.last(ANYA.id)
