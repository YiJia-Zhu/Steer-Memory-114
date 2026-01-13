from __future__ import annotations

import logging

from esm.config import ESMConfig

logger = logging.getLogger(__name__)


def run_memory(cfg: ESMConfig) -> None:
    from esm.offline.stage3 import build_memory

    logger.info("Stage III memory: start")
    build_memory(cfg)
    logger.info("Stage III memory: done")


