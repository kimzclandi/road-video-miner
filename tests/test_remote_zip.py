import io
from unittest.mock import patch

import pytest
from remote_zip import RemoteZip


class Response:
    status_code = 206
    headers = {"ETag": "fixed", "Content-Range": "bytes 0-2/10"}
    raw = io.BytesIO(b"abc")

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def reader():
    r = RemoteZip.__new__(RemoteZip)
    r.url = "https://example.invalid/archive.zip"
    r.etag = "fixed"
    r.size = 10
    r.pos = 0
    r.bytes_read = 0
    r.requests = 0
    return r


def test_reject_full_download_before_body_read():
    response = Response()
    response.status_code = 200
    with patch("remote_zip.requests.get", return_value=response), pytest.raises(ValueError, match="identity"):
        reader().read(3)
    assert response.raw.tell() == 0


def test_range_exact_and_truncated():
    response = Response()
    response.raw = io.BytesIO(b"abc")
    with patch("remote_zip.requests.get", return_value=response):
        assert reader().read(3) == b"abc"
    response.raw = io.BytesIO(b"ab")
    with (
        patch("remote_zip.requests.get", return_value=response),
        pytest.raises(ValueError, match="Incomplete"),
    ):
        reader().read(3)
