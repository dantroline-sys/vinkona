#!/usr/bin/env python3
"""Generate the launcher's icon set with the stdlib only (no PIL): a teal
rounded square with a white V.  Writes the sizes Tauri's bundler wants into
src-tauri/icons/.  Deterministic — rerun freely, commit the output once."""
import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent / "src-tauri" / "icons"
BG = (26, 111, 106, 255)        # teal
FG = (245, 250, 249, 255)       # near-white


def pixel(x, y, n):
    """RGBA for (x, y) in an n×n icon: rounded-corner background + a V glyph."""
    r = n * 0.18                 # corner radius
    for cx, cy in ((r, r), (n - r, r), (r, n - r), (n - r, n - r)):
        if (x < r or x > n - r) and (y < r or y > n - r):
            if (x - cx) ** 2 + (y - cy) ** 2 > r * r and \
               ((x < r or x > n - r) and (y < r or y > n - r)):
                corner = (min(x, n - x) < r and min(y, n - y) < r)
                if corner and (x - (r if x < r else n - r)) ** 2 + \
                        (y - (r if y < r else n - r)) ** 2 > r * r:
                    return (0, 0, 0, 0)
    # the V: two straight strokes meeting at the bottom centre
    u, v = x / n, y / n          # 0..1
    w = 0.11                     # stroke half-width
    top, bot = 0.24, 0.80
    if top <= v <= bot:
        t = (v - top) / (bot - top)          # 0 at top, 1 at the vertex
        left = 0.26 + t * (0.50 - 0.26)      # left stroke centre
        right = 0.74 - t * (0.74 - 0.50)     # right stroke centre
        if abs(u - left) < w * (1 - 0.35 * t) or abs(u - right) < w * (1 - 0.35 * t):
            return FG
    return BG


def png_bytes(n):
    rows = b""
    for y in range(n):
        rows += b"\x00" + b"".join(bytes(pixel(x, y, n)) for x in range(n))
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
    ihdr = struct.pack(">IIBBBBB", n, n, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pngs = {n: png_bytes(n) for n in (32, 128, 256, 512)}
    (OUT / "32x32.png").write_bytes(pngs[32])
    (OUT / "128x128.png").write_bytes(pngs[128])
    (OUT / "128x128@2x.png").write_bytes(pngs[256])
    (OUT / "icon.png").write_bytes(pngs[512])
    # ICO: a container of PNG entries (Vista+ format) — 32 + 256.
    entries, data, off = [], b"", 6 + 16 * 2
    for n in (32, 256):
        p = pngs[n]
        entries.append(struct.pack("<BBBBHHII", n % 256, n % 256, 0, 0, 1, 32,
                                   len(p), off + len(data)))
        data += p
    (OUT / "icon.ico").write_bytes(struct.pack("<HHH", 0, 1, 2)
                                   + b"".join(entries) + data)
    # ICNS: PNG-payload members (ic07=128, ic08=256, ic09=512).
    members = b""
    for tag, n in ((b"ic07", 128), (b"ic08", 256), (b"ic09", 512)):
        p = pngs[n]
        members += tag + struct.pack(">I", 8 + len(p)) + p
    (OUT / "icon.icns").write_bytes(b"icns" + struct.pack(">I", 8 + len(members))
                                    + members)
    print(f"wrote {len(list(OUT.iterdir()))} icons -> {OUT}")


if __name__ == "__main__":
    main()
