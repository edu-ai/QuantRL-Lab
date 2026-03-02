from typing import Any, Dict, List, Union

from quantrl_lab.alpha_research.models import AlphaResult


def results_to_pipeline_config(
    results: List[AlphaResult], top_n: int = 5, metric: str = "ic", deduplicate: bool = False
) -> List[Union[str, Dict[str, Any]]]:
    """
    Convert Alpha Research results into a DataPipeline configuration.

    Filters results to find the best performing indicators and formats them
    into the configuration structure expected by the DataPipeline's
    TechnicalIndicatorStep.

    Args:
        results (List[AlphaResult]): List of completed alpha research results.
        top_n (int): Number of top indicators to select. Defaults to 5.
        metric (str): Metric to use for ranking ("ic", "sharpe_ratio",
            "annual_return"). Defaults to "ic" (Information Coefficient).
        deduplicate (bool): When True, keep only the best-scoring result per
            indicator name. Useful when the same indicator was tested with
            multiple parameter sets (e.g., RSI window=14 and window=21) and
            you only want one entry per indicator type in the pipeline config.
            Defaults to False (all top_n results are included regardless of
            name).

    Returns:
        List[Union[str, Dict[str, Any]]]: A list of indicator configurations
        compatible with TechnicalIndicatorStep.
        Example: [{"RSI": {"window": 14}}, {"SMA": {"window": 50}}]
    """
    if not results:
        return []

    # Filter for successful jobs only
    completed_jobs = [r for r in results if r.status == "completed" and r.metrics]

    if not completed_jobs:
        return []

    # Sort by metric descending (missing metric defaults to -inf)
    sorted_results = sorted(completed_jobs, key=lambda x: x.metrics.get(metric, float("-inf")), reverse=True)

    if deduplicate:
        # Keep only the highest-scoring result per indicator name.
        # sorted_results is already in descending order, so the first
        # occurrence of each name is the best.
        seen: set = set()
        deduped = []
        for r in sorted_results:
            if r.job.indicator_name not in seen:
                seen.add(r.job.indicator_name)
                deduped.append(r)
        sorted_results = deduped

    # Select top N
    top_results = sorted_results[:top_n]

    pipeline_config = []
    for result in top_results:
        indicator_name = result.job.indicator_name
        params = result.job.indicator_params

        # Format: {"IndicatorName": {params}}
        # Dict format is always used when params are non-empty so the exact
        # research params (not registry defaults) are preserved downstream.
        if params:
            pipeline_config.append({indicator_name: params})
        else:
            pipeline_config.append(indicator_name)

    return pipeline_config
