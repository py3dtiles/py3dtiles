import argparse

from py3dtiles import convert, export, info, merger, viewer
from py3dtiles.constants import SpecVersion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read/write 3dtiles files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub_parsers = parser.add_subparsers(dest="command")

    # init subparsers
    command_parsers = [
        convert._init_parser(sub_parsers),
        info._init_parser(sub_parsers),
        merger._init_parser(sub_parsers),
        export._init_parser(sub_parsers),
        viewer._init_parser(sub_parsers),
    ]
    # add the verbose argument for all sub-parsers so that it is after the command.
    for command_parser in command_parsers:
        command_parser.add_argument("--verbose", "-v", action="count", default=0)
        command_parser.add_argument(
            "--spec-version",
            default="1.0",
            choices=[version.value for version in SpecVersion],
            help="Which 3dtiles spec version py3dtiles is allowed to use. If >=1.1, py3dtiles will produce gltf files instead of pnts or b3dm.",
        )

    args = parser.parse_args()

    if args.command == "convert":
        convert._main(args)
    elif args.command == "info":
        info._main(args)
    elif args.command == "merge":
        merger._main(args)
    elif args.command == "export":
        export._main(args)
    elif args.command == "view":
        viewer._main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
