import sys

from .config import build_convert_parser, build_main_parser
from .pipeline import run_batch_mode, run_convert_mode, run_tracking


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if len(argv) > 0 and argv[0] == "convert":
        convert_parser = build_convert_parser()
        convert_args = convert_parser.parse_args(argv[1:])
        run_convert_mode(convert_args)
        return

    parser = build_main_parser()
    if len(argv) == 0:
        parser.print_help(sys.stderr)
        return

    args = parser.parse_args(argv)
    if args.runs_config:
        run_batch_mode(args)
    else:
        run_tracking(args)
