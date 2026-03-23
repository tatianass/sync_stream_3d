import logging
from typing import Tuple

import great_expectations as gx
import pandas as pd

# Ignore type hints for GX 1.x since it doesn't depend on us
from great_expectations.expectations import (
    ExpectColumnValuesToBeBetween,  # type: ignore
    ExpectColumnValuesToBeUnique,  # type: ignore
    ExpectColumnValuesToMatchRegex,  # type: ignore
    ExpectColumnValuesToNotBeNull,  # type: ignore
)

logger = logging.getLogger(__name__)


class QualityGuard:
    @staticmethod
    def validate_index(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validates the index using the GX 1.x Batch-Validation pattern.
        """
        try:
            # 1. Get the ephemeral context
            context = gx.get_context()

            # 2. Retrieve a Batch directly from the DataFrame
            # The 'pandas_default' data source is built-in for GX 1.x
            batch = context.data_sources.pandas_default.read_dataframe(df)

            # 3. Define your individual Expectations
            # We wrap these in a list to validate them all at once
            expectations = [
                ExpectColumnValuesToNotBeNull(column="timestamp_ns"),
                ExpectColumnValuesToBeUnique(column="chunk_id"),
                ExpectColumnValuesToMatchRegex(
                    column="semantic_tags", regex=r"^\{.*\}$"
                ),
                ExpectColumnValuesToBeBetween(
                    column="pose_x", min_value=-10000, max_value=10000
                ),
            ]

            # 4. Run Validation
            # In GX 1.x, batch.validate can take an Expectation or a Suite
            all_success = True
            total_evaluated = len(expectations)
            successful_count = 0

            for exp in expectations:
                result = batch.validate(exp)
                if result.success:
                    successful_count += 1
                else:
                    all_success = False
                    logger.warning(
                        f"Expectation failed: {exp.__class__.__name__} on {exp.column}"
                    )

            summary = f"GX Check: {successful_count}/{total_evaluated} passed."
            return all_success, summary

        except Exception as e:
            logger.error(f"GX Validation Error: {e}")
            return False, f"Validation Failed: {str(e)}"


if __name__ == "__main__":
    # Example usage
    df = pd.read_parquet("output/index.parquet")
    success, report = QualityGuard.validate_index(df)
    if success:
        logger.info("Data validation passed. " + report)
    else:
        logger.warning("Data validation failed. " + report)
        raise ValueError(report)
