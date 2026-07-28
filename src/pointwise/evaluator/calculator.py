import pandas as pd
from msr.constants import *
from .metric import *


class Calculator(object):
    def __init__(
        self, 
        cutoff: list,
        user_col: str=DEFAULT_USER_COL,
        item_col: str=DEFAULT_ITEM_COL,
        rating_col: str=DEFAULT_RATING_COL,
        prediction_col: str=DEFAULT_PREDICTION_COL,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.prediction_col = prediction_col

    def __call__(
        self, 
        result: pd.DataFrame, 
    ) -> pd.DataFrame:
        rating_true, rating_pred = self.sep(result)

        eval = [
            self.calc(
                rating_true=rating_true, 
                rating_pred=rating_pred, 
                k=k,
            )
            for k in self.cutoff
        ]

        return pd.DataFrame(eval)

    def sep(
        self, 
        result: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        TRUE_COL_LIST = [self.user_col, self.item_col, self.rating_col]
        PRED_COL_LIST = [self.user_col, self.item_col, self.prediction_col]

        rating_true = (
            result[TRUE_COL_LIST]
            [result[self.rating_col]==1]
            .sort_values(
                by=self.user_col, 
                ascending=True,
            )
        )

        rating_pred = (
            result[PRED_COL_LIST]
            .sort_values(
                by=[self.user_col, self.prediction_col], 
                ascending=[True, False], 
                kind='stable',
            ).groupby(self.user_col)
        )

        return rating_true, rating_pred

    def calc(
        self, 
        rating_true: pd.DataFrame, 
        rating_pred: pd.DataFrame, 
        k: int,
    ) -> dict[str, int]:
        kwargs = dict(
            rating_true=rating_true,
            rating_pred=rating_pred.head(k),
            col_user=self.user_col,
            col_item=self.item_col,
            col_rating=self.rating_col,
            col_prediction=self.prediction_col,
            k=k,
        )
        return dict(
            k=k,
            hit_ratio=hit_ratio_at_k(**kwargs), 
            precision=precision_at_k(**kwargs), 
            recall=recall_at_k(**kwargs), 
            map=map_at_k(**kwargs), 
            ndcg=ndcg_at_k(**kwargs),
        )