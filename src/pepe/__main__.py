import logging
import sys
from pepe.parse_arguments import parse_arguments

logger = logging.getLogger("pepe.__main__")


def main():
    args = parse_arguments()

    if args.check_model:
        from pepe.model_selecter import report_model

        report_model(args.check_model, trust_remote_code=args.trust_remote_code)
        return

    missing = []
    if not args.model_name:
        missing.append("--model_name")
    if not args.fasta_path:
        missing.append("--fasta_path")
    if not args.output_path:
        missing.append("--output_path")
    if missing:
        print(
            f"Error: the following arguments are required: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Lazy import to avoid loading heavy dependencies during --help
    from pepe.model_selecter import select_model

    selected_model = select_model(
        args.model_name, trust_remote_code=args.trust_remote_code
    )

    embedder = selected_model(args)

    logger.info("Embedder initialized")

    embedder.run()

    logger.info("All outputs saved.")


if __name__ == "__main__":
    main()
