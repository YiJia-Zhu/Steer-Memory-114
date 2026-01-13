from __future__ import annotations

import logging

from esm.config import ESMConfig

logger = logging.getLogger(__name__)


def run_mine(cfg: ESMConfig) -> None:
    from esm.offline.stage1 import mine_candidates

    logger.info("Stage I mining: start")
    mine_candidates(cfg)
    logger.info("Stage I mining: done")


