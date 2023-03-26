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
from typing import Dict, Any, Optional

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress


@register('arabic_squad_reader')
class ArabicSquadReader(DatasetReader):
    """
    Downloads dataset files and prepares train/valid split.

    SQuAD:
    Stanford Question Answering Dataset
    https://rajpurkar.github.io/SQuAD-explorer/
    
    SQuAD2.0:
    Stanford Question Answering Dataset, version 2.0
    https://rajpurkar.github.io/SQuAD-explorer/

    SberSQuAD:
    Dataset from SDSJ Task B
    https://www.sdsj.ru/ru/contest.html

    MultiSQuAD:
    SQuAD dataset with additional contexts retrieved (by tfidf) from original Wikipedia article.

    MultiSQuADRetr:
    SQuAD dataset with additional contexts retrieved by tfidf document ranker from full Wikipedia.

    """

    def read(self, dir_path: str, dataset: Optional[str] = 'SQuAD', url: Optional[str] = None, *args, **kwargs) \
            -> Dict[str, Dict[str, Any]]:
        """

        Args:
            dir_path: path to save data
            dataset: default dataset names: ``'SQuAD'``, ``'SberSQuAD'`` or ``'MultiSQuAD'``
            url: link to archive with dataset, use url argument if non-default dataset is used

        Returns:
            dataset split on train/valid

        Raises:
            RuntimeError: if `dataset` is not one of these: ``'SQuAD'``, ``'SberSQuAD'``, ``'MultiSQuAD'``.
        """
    

        dataset = {}
        with open(f"{dir_path}/train.json",'r', encoding='utf8') as fp:
            data = json.load(fp)
            dataset["train"] = data

        with open(f"{dir_path}/valid.json",'r', encoding='utf8') as fp:
            data = json.load(fp)
            dataset["valid"] = data
        
        return dataset
