from abc import ABC, abstractmethod
from collections import Iterable
from multiprocessing import Pool
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from openea.modules.load.kgs import KGs
from sklearn.preprocessing import MinMaxScaler

from attribute_features import AttributeFeatureCombination
from distance_measures import DistanceMeasure
from similarity.create_training import create_similarity_frame_on_demand
from similarity.similarities import calculate_on_demand
from similarity_measures import SimilarityMeasure


class PairToVec:
    def __init__(
        self,
        embeddings: np.ndarray,
        kgs: KGs,
        name: str,
        attr_combination: AttributeFeatureCombination,
        embedding_measures: List[Union[DistanceMeasure, SimilarityMeasure]],
    ):
        self.embeddings = embeddings
        self.kgs = kgs
        self.name = name
        self.attr_combination = attr_combination
        self.embedding_measures = embedding_measures
        self.all_sims = None
        self.all_keys = None
        self.min_max = None
        self.cols = None

    def __call__(self, e1: int, e2: int) -> np.ndarray:
        if (e1, e2) in self.all_sims.index:
            sim = self.all_sims.loc[(e1, e2)].fillna(0)
        else:
            sim = self._calculate_on_demand(e1, e2).fillna(0)
        sim_vec = np.asarray([sim.get(k, 0.0) for k in self.all_keys])
        return sim_vec

    def dimension(self) -> int:
        return len(self.all_keys)

    def prepare(self, all_pairs):
        self.all_sims, self.min_max, self.cols = self._calculate_all_sims(all_pairs)
        print(len(self.all_sims.columns))
        self.all_sims = self.all_sims.dropna(
            axis=1, how="all", thresh=int(0.1 * len(self.all_sims))
        )
        print(len(self.all_sims.columns))
        self.all_sims.sort_index(inplace=True)
        self.all_keys = self.all_sims.columns

    def _calculate_on_demand(self, e1_index, e2_index):
        comparisons = self._calculate_pair_comparisons(e1_index, e2_index)
        sim_frame = pd.DataFrame.from_dict(comparisons, orient="index", dtype="float32")
        df, _ = self._create_normalized_sim_from_dist_cols(
            sim_frame, self.cols, self.min_max
        )
        return df

    def _calculate_all_sims(self, all_pairs):
        measures_to_normalize = [
            type(m).__name__
            for m in self.attr_combination.all_measures + self.embedding_measures
            if isinstance(m, DistanceMeasure)
        ]
        # TODO: parallelize
        with Pool() as pool:
            comparison_list = pool.starmap(self._calculate_pair_comparisons, [e[:2] for e in all_pairs])
        # for pair in all_pairs:
        #     comparisons[pair] = self._calculate_pair_comparisons(pair[0], pair[1])
        comparisons = {(e[0], e[1]): comp for e, comp in zip(all_pairs, comparison_list)}
        print("Finished calculation from given links")
        return self._create_labeled_similarity_frame(comparisons, measures_to_normalize)

    def _create_labeled_similarity_frame(
        self,
        comparisons: dict,
        measures_to_normalize: List[str],
        min_max: MinMaxScaler = None,
        cols: List[str] = None,
    ) -> Tuple[pd.DataFrame, MinMaxScaler, List[str]]:
        """
        Creates pandas DataFrame with the comparisons and labels (if labels for tuple are present)
        Distances will be normalized to similarities
        :param comparisons: dictionary of dictionaries of similarities per entity tuple
        :return: SparseDataFrame with labels if available
        """
        # create similarity frame
        sim_frame = pd.DataFrame.from_dict(comparisons, orient="index", dtype="float32")
        if len(measures_to_normalize) == 0:
            return sim_frame, min_max, cols

        print("Normalizing dataframe...")
        if cols is None:
            cols = [
                c
                for c in sim_frame.columns
                if any(m in c for m in measures_to_normalize)
            ]
        df, min_max = self._create_normalized_sim_from_dist_cols(
            sim_frame, cols, min_max
        )
        return df, min_max, cols

    def _create_normalized_sim_from_dist_cols(
        self, df: pd.DataFrame, cols: List[str], min_max: MinMaxScaler = None
    ) -> Tuple[pd.DataFrame, MinMaxScaler]:
        if len(cols) == 0 or len(df) == 0:
            return df, min_max
        min_max = min_max or MinMaxScaler().fit(df[cols])
        if all(c in df for c in cols):
            df[cols] = min_max.transform(df[cols])
        else:
            cols_in_df = [c for c in cols if c in df]
            other_cols = [c for c in cols if c not in df]
            if len(cols_in_df) > 0:
                df[cols_in_df] = min_max.transform(df[cols_in_df])
            if len(other_cols) > 0:
                df[other_cols] = [[1.0] * len(other_cols)] * len(df)
        df[cols] = 1 - df[cols]
        return df, min_max

    def _calculate_pair_comparisons(self, e1_index: np.int64, e2_index: np.int64):
        values = {}
        aligned_attrs = self.attr_combination.align_attributes(e1_index, e2_index)
        for a in aligned_attrs:
            key = f"{a.k1}:{a.k2}"
            for measure in a.measures:
                name = f"{type(measure).__name__}.{key}"
                comp_value = measure(a.v1, a.v2)
                if isinstance(comp_value, Iterable):
                    values.update({f"{name}:{i}": v for i, v in enumerate(comp_value)})
                else:
                    values[name] = comp_value

        for measure in self.embedding_measures:
            name = type(measure).__name__
            comp_value = measure(self.embeddings[e1_index], self.embeddings[e2_index])
            if isinstance(comp_value, Iterable):
                values.update({f"{name}:{i}": v for i, v in enumerate(comp_value)})
            else:
                values[name] = comp_value
        return values
