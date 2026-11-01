import pandas as pd


def with_analysis_features(frame):
    result = frame.copy()
    result["Water_Cement_Ratio"] = result.Water / result.Cement
    result["Age Group"] = pd.cut(result.Age, [0, 7, 28, 90, float("inf")],
                                 labels=["1–7 days", "8–28 days", "29–90 days", ">90 days"])
    return result
