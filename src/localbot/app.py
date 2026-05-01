"""Entry point and Discord client setup."""
import asyncio
import logging

log = logging.getLogger(__name__)


def main() -> None:
    """Start the bot. Full implementation coming soon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("LocalBot starting up — implementation in progress.")
    print("LocalBot skeleton loaded. Full bot implementation is not yet complete.")


if __name__ == "__main__":
    main()
