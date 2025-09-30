import argparse
import functools
import http.server
import sys
import webbrowser
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote


class CustomHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args: Any, paths: list[Path], **kwargs: Any) -> None:
        self.paths = paths if paths is not None else []
        super().__init__(*args, **kwargs)

    def _make_homepage(self) -> bytes:
        links = "".join(
            f'<li><a href="/{i}">{i} ({path})</a></li>'
            for i, path in enumerate(self.paths)
        )
        return f"<h1>Available Files</h1><ul>{links}</ul>".encode()

    def do_GET(self) -> None:
        if self.path == "/":
            # Serve a simple HTML page with links
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self._make_homepage())
        else:
            try:
                parts = self.path.split("/")
                idx = int(parts[1])
                basepath = Path(self.paths[idx])
                filepath = basepath / Path(*parts[2:])
                if filepath.is_dir():
                    # Serve a simple HTML page with links
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=utf-8")
                    self.end_headers()
                    links = "".join(
                        f'<li><a href="/{idx}/{path.relative_to(basepath)}">{path}</a></li>'
                        for _, path in enumerate(sorted(filepath.iterdir()))
                    )
                    # we have checked that the filepath exists and is_dir() above
                    # let's consider it enough sanitization for this use case
                    # SONAR-IGNORE
                    self.wfile.write(
                        f"<h1>Available Files in {quote(str(filepath))}</h1><ul>{links}</ul>".encode()
                    )
                    # END-SONAR-IGNORE
                else:
                    if filepath.exists():
                        self.send_response(200)
                        if filepath.suffix == ".json":
                            self.send_header("Content-Type", "application/json")
                        else:
                            self.send_header(
                                "Content-Disposition",
                                f'attachment; filename="{quote(filepath.name)}"',
                            )
                            self.send_header("Content-type", "application/octet-stream")
                        self.end_headers()
                        with filepath.open("rb") as f:
                            self.wfile.write(f.read())
                    else:
                        self.send_error(404, "File not found")
            except (ValueError, IndexError):
                self.send_error(404, "File not found")
            except Exception as e:
                print(e, file=sys.stderr)
                self.send_error(500, str(e))

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def _init_parser(
    subparser: "argparse._SubParsersAction[Any]",
) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparser.add_parser(
        "view",
        help="View tilesets in your browser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--giro3d-base",
        default="https://giro3d.org/examples",
        help="The base for the giro3d viewer",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="List of tileset.json to view in the browser.",
    )
    return parser


def open_giro3d(giro3d_base: str, tileset_url: str) -> None:
    url = f"{giro3d_base}/3dtiles_simple_viewer.html?tileset_url={tileset_url}"
    webbrowser.open(url)


def calculate_paths_and_url_suffixes(
    files: Iterable[str | Path],
) -> list[tuple[Path, str]]:
    paths_and_urls_suffixes = []
    i = 0  # noqa # no idea why flake-simplify flags this, as they advertise supporting for-loop with continue
    for file in files:
        path = Path(file)
        if path.is_file():
            path_to_serve = path.parent
            tileset_path = path
        elif path.is_dir():
            potential_tileset_path = path / "tileset.json"
            print(f"WARNING: {path} is a directory, trying {potential_tileset_path}")
            if potential_tileset_path.is_file():
                print(f"-> {potential_tileset_path} found")
                path_to_serve = path
                tileset_path = potential_tileset_path
            else:
                print("WARNING: not found, skipping")
                continue
        else:
            print(f"WARNING: {file} does not exist, skipping")
            continue
        paths_and_urls_suffixes.append((path_to_serve, f"{i}/{tileset_path.name}"))
        i += 1
    return paths_and_urls_suffixes


def _get_handler_class(paths: list[Path]) -> functools.partial[CustomHandler]:
    return functools.partial(CustomHandler, paths=paths)


def _main(args: argparse.Namespace) -> None:
    paths_and_urls_suffixes = calculate_paths_and_url_suffixes(args.files)

    if len(paths_and_urls_suffixes) == 0:
        print("No tileset to serve, exiting")
        sys.exit(1)
    # The api of these RequestHandler is not simple and not nice :-/
    paths = [path for (path, _) in paths_and_urls_suffixes]
    handler_cls = _get_handler_class(paths)
    with http.server.HTTPServer(("", 0), handler_cls) as httpd:
        print(
            f"Serving {paths} files at http://{httpd.server_name}:{httpd.server_port}"
        )
        print("Now launching your browser!")
        for _, url_suffix in paths_and_urls_suffixes:
            # this is perfectly fine in our context, we can't nor we need having https here
            # SONAR-IGNORE
            tileset_url = f"http://{httpd.server_name}:{httpd.server_port}/{url_suffix}"
            # END-SONAR-IGNORE
            # technically, there is a race condition here because we open
            # giro3d before httpd is started that being said, in practice, this
            # takes so much time (opening a browser page through xdg-open, loading the page, javascript executes,
            # then launches the request...), that I think it won't be a problem
            open_giro3d(args.giro3d_base, tileset_url)
        httpd.serve_forever()
