from pathlib import Path
from urllib.error import HTTPError

from pytest import raises

from py3dtiles.viewer import calculate_paths_and_url_suffixes
from tests.conftest import HttpInfo


def test_calculate_paths_and_url_suffixes(tmp_dir: Path) -> None:
    files = [
        tmp_dir / "foo/bar",
        tmp_dir / "test2/tileset.json",
        tmp_dir / "test4/prefix/tileset_2.json",
        tmp_dir / "test3/path/tileset.json",
        tmp_dir / "baz/another",
        tmp_dir / "not/existing/too/bad",
    ]
    # 0, 1 and 2 ok
    Path(files[0]).mkdir(parents=True)
    Path(files[0], "tileset.json").touch()
    Path(files[1]).parent.mkdir(parents=True)
    Path(files[1]).touch()
    Path(files[2]).parent.mkdir(parents=True)
    Path(files[2]).touch()
    # test3 and 4: missing tileset.json
    Path(files[3]).parent.mkdir(parents=True)
    Path(files[4]).mkdir(parents=True)
    # test 5: missing a folder

    res = calculate_paths_and_url_suffixes(files)
    assert res == [
        (tmp_dir / "foo/bar", "0/tileset.json"),
        (tmp_dir / "test2", "1/tileset.json"),
        (tmp_dir / "test4/prefix", "2/tileset_2.json"),
    ]


def test_requests(test_viewer_server: HttpInfo) -> None:
    import urllib.request

    http_info = test_viewer_server
    url = f"http://{http_info.host}:{http_info.port}"

    # homepage
    r = urllib.request.urlopen(url)
    assert r.status == 200
    assert r.getheader("Content-type") == "text/html; charset=utf-8"
    assert (
        r.read()
        == b'<h1>Available Files</h1><ul><li><a href="/0">0 (tmp/simple1_ifc)</a></li><li><a href="/1">1 (tmp/1)</a></li></ul>'
    )

    # folder page
    r = urllib.request.urlopen(f"{url}/0")
    assert r.status == 200
    assert r.getheader("Content-type") == "text/html; charset=utf-8"
    assert (
        r.read()
        == b'<h1>Available Files in tmp/simple1_ifc</h1><ul><li><a href="/0/geometry">tmp/simple1_ifc/geometry</a></li><li><a href="/0/tileset.json">tmp/simple1_ifc/tileset.json</a></li></ul>'
    )

    r = urllib.request.urlopen(f"{url}/1")
    assert r.status == 200
    assert r.getheader("Content-type") == "text/html; charset=utf-8"
    assert (
        r.read()
        == b'<h1>Available Files in tmp/1</h1><ul><li><a href="/1/points">tmp/1/points</a></li><li><a href="/1/preview.pnts">tmp/1/preview.pnts</a></li><li><a href="/1/tileset.json">tmp/1/tileset.json</a></li></ul>'
    )

    with raises(HTTPError, match="HTTP Error 404: File not found"):
        r = urllib.request.urlopen(f"{url}/2")

    # an existing tileset
    r = urllib.request.urlopen(f"{url}/0/tileset.json")
    assert r.status == 200
    assert r.getheader("Content-type") == "application/json"
    r = urllib.request.urlopen(f"{url}/1/tileset.json")
    assert r.status == 200
    assert r.getheader("Content-type") == "application/json"

    # a non-existing tileset
    with raises(HTTPError, match="HTTP Error 404: File not found"):
        r = urllib.request.urlopen(f"{url}/0/tileset2.json")
    with raises(HTTPError, match="HTTP Error 404: File not found"):
        r = urllib.request.urlopen(f"{url}/2/tileset.json")

    # an existing binary file
    r = urllib.request.urlopen(f"{url}/1/preview.pnts")
    assert r.status == 200
    assert r.getheader("Content-type") == "application/octet-stream"
    r = urllib.request.urlopen(f"{url}/0/geometry/1.b3dm")
    assert r.status == 200
    assert r.getheader("Content-type") == "application/octet-stream"

    # a non existing binary file
    with raises(HTTPError, match="HTTP Error 404: File not found"):
        r = urllib.request.urlopen(f"{url}/0/r.pnts")
        assert r.read() == "bar"
