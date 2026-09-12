import argparse
import sys


def main(args=None):
    parser = argparse.ArgumentParser(prog="raschlab")
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--con", required=True)
    analyze_parser.add_argument("--data")
    analyze_parser.add_argument("--labels")
    analyze_parser.add_argument("--out")
    analyze_parser.add_argument("--format")
    analyze_parser.add_argument("--digits")

    parsed = parser.parse_args(args)
    if parsed.command == "analyze":
        print("not implemented yet")
        sys.exit(2)


if __name__ == "__main__":
    main()
