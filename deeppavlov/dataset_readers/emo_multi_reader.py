# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress


@register('emo_multi_reader')
class EmoMultiReader(DatasetReader):

    def read(self,
            data_path: str,
            *args, **kwargs) -> Dict[str, List[Tuple[Tuple[str, str], int]]]:

        """
        Reads BoolQ dataset from files.

        Args:
            data_path: A path to a folder with dataset files.

        Returns:
            dataset: items of the dataset [(question, passage), label]
        """

        data_path = expand_path(data_path)
        if not data_path.exists():
            data_path.mkdir(parents=True)

        dataset = {}

        for filename in ['valid.tsv', 'train.tsv', 'test.tsv']:
            dataset[filename.split('.')[0]] = self._build_data(data_path / filename)

        return dataset

    @staticmethod
    def _build_data(data_path: Path) -> List[Tuple[Tuple[str, str], int]]:
        data = {}
        df = pd.read_csv(data_path, sep="\t")
        for _, row in df.iterrows():
            data[row["utt_1"], row["utt_2"]] = row["emotion"]

        return list(data.items())