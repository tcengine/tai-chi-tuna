from tai_chi_tuna.flow.to_enrich import execute_enrich
from tai_chi_tuna.config import PhaseConfig
import pandas as pd

example_df = pd.DataFrame(
    {
        "a": [1, 2, 3, 4, 5],
        "b": [1, 2, 3, 4, 5],
    })


def test_no_enrich():
    result_df = execute_enrich(example_df, PhaseConfig(), [])
    assert result_df.equals(example_df)