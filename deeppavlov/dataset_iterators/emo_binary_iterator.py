#  Copyright 2020 Neural Networks and Deep Learning lab, MIPT
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

from typing import List, Tuple, Any, Union

from datasets import Dataset
import pandas as pd

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('emo_binary_iterator')
class EmoBinaryDatasetIterator(DataLearningIterator):
    """Dataset iterator for HuggingFace Datasets."""

    def preprocess(self,
                data: Dataset,
                features: Union[str, List[str]],
                label: str = 'label',
                use_label_name: bool = False,
                *args, **kwargs) -> List[Tuple[Any, Any]]:
        """Extracts features and labels from HuggingFace Dataset

        Args:
            data: instance of HuggingFace Dataset
            features: Dataset fields names to be extracted as features
            label: Dataset field name to be used as label.
            use_label_name: Use actual label name instead of its index (0, 1, ...). Defaults to True.

        Returns:
            List[Tuple[Any, Any]]: list of pairs of extracted features and labels
        """
        # print(f"data: {data}")
        dataset = []
        for example in data:
            # print(example)
            if isinstance(features, str):
                feat = example[features]
            elif isinstance(features, list):
                # feat = [str(x) for x in example[0]]
                feat = tuple(str(x) for x in example[0])
                # feat = tuple(example[f] for f in features)
            else:
                raise RuntimeError(f"features should be str or list, but found: {features}")
            lb = str(example[1])
            if use_label_name and lb != -1:
                # -1 label is used if there is no label (test set)
                lb = data.info.features[label].names[lb]
            dataset += [(feat, lb)]

        return dataset