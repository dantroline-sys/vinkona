#!/usr/bin/env python
"""
Text-mode test client for the cascade server (no audio).  Connects with
?mode=text, lets you type at the prompt, and prints the assistant's bubbles —
a quick way to exercise the fast LM + the text protocol without the Flutter app.

  python test_text_chat.py --url wss://127.0.0.1:8998 [--persona vinkona]

Only the cascade server + fast LM need to be running (TTS/Whisper not used).
"""

import argparse
import asyncio
import json
import ssl
import sys

import aiohttp


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="wss://127.0.0.1:8998")
    ap.add_argument("--persona", default=None)
    args = ap.parse_args()

    path = "/api/chat?mode=text" + (f"&persona={args.persona}" if args.persona else "")
    ssl_ctx = ssl.create_default_context()           # accept the self-signed cert
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(args.url + path, ssl=ssl_ctx) as ws:

            async def reader():
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY and msg.data:
                        kind = msg.data[0]
                        if kind == 0x00:
                            print("[connected — type a message]")
                        elif kind == 0x02:
                            try:
                                b = json.loads(msg.data[1:])
                                print(f"  {b['role']}: {b['text']}")
                            except Exception:
                                pass
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

            reader_task = asyncio.create_task(reader())
            loop = asyncio.get_running_loop()
            try:
                while True:
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                    if not line:
                        break
                    text = line.strip()
                    if not text:
                        continue
                    await ws.send_bytes(b"\x04" + json.dumps({"text": text}).encode("utf8"))
            finally:
                reader_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
