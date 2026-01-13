from __future__ import annotations

import logging

from esm.config import ESMConfig

logger = logging.getLogger(__name__)


def run_select(cfg: ESMConfig) -> None:
    from esm.offline.stage2 import select_library

    logger.info("Stage II selection: start")
    select_library(cfg)
    logger.info("Stage II selection: done")


