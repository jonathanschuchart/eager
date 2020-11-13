import json
from collections import Iterable
from multiprocessing import Pool
from typing import List, Tuple, Union
import os
from os.path import exists

import joblib
import numpy as np
import pandas as pd
from openea.modules.load.kgs import KGs
from sklearn.preprocessing import MinMaxScaler

from attribute_features import AttributeFeatureCombination
from distance_measures import DistanceMeasure
from similarity_measures import SimilarityMeasure


class PairToVec:
    def __init__(
        self,
        embeddings: np.ndarray,
        kgs: KGs,
        name: str,
        attr_combination: AttributeFeatureCombination,
        embedding_measures: List[Union[DistanceMeasure, SimilarityMeasure]],
        all_sims: pd.DataFrame = None,
        min_max: MinMaxScaler = None,
        scale_cols: List[str] = None,
    ):
        self.embeddings = embeddings
        self.kgs = kgs
        self.name = name
        self.attr_combination = attr_combination
        self.embedding_measures = embedding_measures
        self.all_sims = all_sims
        self.all_keys = self.all_sims.columns if self.all_sims is not None else None
        self.min_max = min_max
        self.cols = scale_cols

    def __call__(self, e1: int, e2: int) -> np.ndarray:
        if (e1, e2) in self.all_sims.index:
            sim = self.all_sims.loc[(e1, e2)].fillna(0.0)
        else:
            sim = self._calculate_on_demand(e1, e2)
        sim_vec = np.asarray([sim.get(k, 0.0) for k in self.all_keys])
        return sim_vec

    def dimension(self) -> int:
        return len(self.all_keys)

    def prepare(self, all_pairs):
        self.all_sims, self.min_max, self.cols = self._calculate_all_sims(all_pairs)
        self.all_sims = self.all_sims.dropna(
            axis=1, how="all", thresh=int(0.1 * len(self.all_sims))
        )
        self.all_sims.sort_index(inplace=True)
        self.all_keys = self.all_sims.columns

    def set_prepared(self, all_sims: pd.DataFrame, min_max: MinMaxScaler, scale_cols: List[str]):
        self.all_sims = all_sims
        self.min_max = min_max
        self.cols = scale_cols
        self.all_keys = all_sims.columns

    def save(self, folder):
        self.all_sims.to_parquet(os.path.join(folder, f"{self.name}-all_sims.parquet"))
        joblib.dump(self.min_max, os.path.join(folder, f"{self.name}-min_max.pkl"))
        with open(os.path.join(folder, f"{self.name}-scale_cols.json"), "w") as f:
            json.dump(self.cols, f)
        joblib.dump(
            self.attr_combination, os.path.join(folder, f"{self.name}-attr_combs.pkl")
        )
        joblib.dump(
            self.embedding_measures,
            os.path.join(folder, f"{self.name}-emb_measures.pkl"),
        )

    @staticmethod
    def load(embeddings, kgs, folder, name):
        all_sims_file_name = os.path.join(folder, f"{name}-all_sims.parquet")
        if not exists(all_sims_file_name):
            return None
        all_sims = pd.read_parquet(all_sims_file_name)
        min_max_file_name = os.path.join(folder, f"{name}-min_max.pkl")
        if exists(min_max_file_name):
            min_max = joblib.load(os.path.join(folder, f"{name}-min_max.pkl"))
        else:
            min_max = None
        scale_cols_file_name = os.path.join(folder, f"{name}-scale_cols.json")
        if exists(scale_cols_file_name):
            with open(scale_cols_file_name) as f:
                cols = json.load(f)
        else:
            cols = []
        attr_comb_file_name = os.path.join(folder, f"{name}-attr_combs.pkl")
        emb_measure_file_name = os.path.join(folder, f"{name}-emb_measures.pkl")
        if not exists(attr_comb_file_name) or not exists(emb_measure_file_name):
            return None
        attr_combination = joblib.load(attr_comb_file_name)
        embedding_measures = joblib.load(emb_measure_file_name)
        return PairToVec(
            embeddings,
            kgs,
            name,
            attr_combination,
            embedding_measures,
            all_sims,
            min_max,
            cols,
        )


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
        with Pool() as pool:
            comparison_list = pool.starmap(
                self._calculate_pair_comparisons, [e[:2] for e in all_pairs]
            )
        comparisons = {
            (e[0], e[1]): comp for e, comp in zip(all_pairs, comparison_list)
        }

        # comparisons = {(pair[0], pair[1]): self._calculate_pair_comparisons(pair[0], pair[1]) for pair in all_pairs}
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
        if cols is None or len(cols) == 0 or df is None or len(df) == 0:
            return df, min_max
        min_max = min_max or MinMaxScaler().fit(df[cols])
        if all(c in df for c in cols):
            df[cols] = min_max.transform(df[cols])
        else:
            cols = [c for c in cols if c in df]
            if len(cols) > 0:
                df[cols] = min_max.transform(df[cols])
        df[cols] = 1 - df[cols]
        return df, min_max

    def _calculate_pair_comparisons(self, e1_index: np.int64, e2_index: np.int64):
        values = {}
        aligned_attrs = self.attr_combination.align_entity_attributes(
            e1_index, e2_index
        )
        for a in aligned_attrs:
            key = ":".join(sorted((str(a.k1), str(a.k2))))
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
