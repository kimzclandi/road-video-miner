"""Strict range-only reader for public ZIP/ZIP64 archives; validate ETag and CRC."""

import io
import struct
import zipfile

import requests


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        h = requests.head(url, timeout=30)
        h.raise_for_status()
        self.size = int(h.headers["Content-Length"])
        self.etag = h.headers["ETag"]
        self.pos = 0
        self.bytes_read = 0
        self.requests = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError("negative seek")
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n == 0:
            return b""
        if n > 16 * 1024 * 1024:
            raise ValueError("Range exceeds 16 MiB safety cap")
        start = self.pos
        with requests.get(
            self.url,
            headers={"Range": f"bytes={start}-{start + n - 1}", "If-Match": self.etag},
            timeout=60,
            stream=True,
        ) as response:
            response.raise_for_status()
            if (
                response.status_code != 206
                or response.headers.get("ETag") != self.etag
                or response.headers.get("Content-Range") != f"bytes {start}-{start + n - 1}/{self.size}"
            ):
                raise ValueError("Range or source identity mismatch")
            data = response.raw.read(n + 1)
        if len(data) != n:
            raise ValueError("Incomplete or oversized range")
        self.pos += n
        self.bytes_read += n
        self.requests += 1
        return data

    def member(self, info):
        # One bounded request per member instead of zipfile's three header/data reads.
        self.seek(info.header_offset)
        header = self.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Bad local ZIP header")
        n, e = struct.unpack("<HH", header[26:30])
        self.seek(info.header_offset)
        body = self.read(30 + n + e + info.compress_size)
        # ZIP decompression and CRC implemented explicitly for bounded payloads.
        import zlib

        raw = body[30 + n + e :]
        data = zlib.decompress(raw, -15) if info.compress_type == zipfile.ZIP_DEFLATED else raw
        if info.compress_type not in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
            raise ValueError("Unsupported codec")
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise ValueError("Member CRC mismatch")
        return data
