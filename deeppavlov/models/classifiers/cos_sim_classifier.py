# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwaredata
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from typing import List, Tuple, Union
from scipy.sparse import vstack
from scipy.sparse import csr_matrix

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.file import save_pickle
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.commands.utils import expand_path, make_all_dirs
from deeppavlov.core.models.serializable import Serializable
from numpy import linalg
from scipy.sparse.linalg import norm as sparse_norm

logger = get_logger(__name__)


@register("cos_sim_classifier")
class CosineSimilarityClassifier(Estimator, Serializable):
    """
    Classifier based on cosine similarity between vectorized sentences

    Parameters:
        save_path: path to save the model
        load_path: path to load the model

    Returns:
        None
    """

    def __init__(self, top_n: int = 1, save_path: str = None, load_path: str = None, **kwargs) -> None:
        self.save_path = save_path
        self.load_path = load_path
        self.top_n = top_n
        if kwargs['mode'] != 'train':
            self.load()

    def __call__(self, q_vects: Union[csr_matrix, List]) -> Tuple[List[str], List[int]]:
        """Found most similar answer for input vectorized question

        Parameters:
            q_vects: vectorized questions

        Returns:
            Tuple of Answer and Score
        """

        if isinstance(q_vects[0], csr_matrix):
            norm = sparse_norm(q_vects) * sparse_norm(self.x_train_features, axis=1)
            cos_similarities = np.array(q_vects.dot(self.x_train_features.T).todense())/norm
        elif isinstance(q_vects[0], np.ndarray):
            q_vects = np.array(q_vects)
            norm = linalg.norm(q_vects)*linalg.norm(self.x_train_features, axis=1)
            cos_similarities = q_vects.dot(self.x_train_features.T)/norm
        elif q_vects[0] is None:
            cos_similarities = np.zeros(len(self.x_train_features))
        else:
            raise NotImplementedError('Not implemented this type of vectors')

        # get cosine similarity for each class
        y_labels = np.unique(self.y_train)
        labels_scores = np.zeros((len(cos_similarities), len(y_labels)))
        for i, label in enumerate(y_labels):
            labels_scores[:, i] = np.max([cos_similarities[:, i] for i, value in enumerate(self.y_train) if value == label], axis=0)

        # normalize for each class
        labels_scores = labels_scores/np.sum(labels_scores, axis=1)
        answer_ids = np.argsort(labels_scores)[:, -self.top_n:]

        # generate top_n asnwers and scores
        answers = []
        scores = []
        for i in range(len(answer_ids)):
            answers.append([y_labels[id] for id in answer_ids[i, ::-1]])
            scores.append([np.round(labels_scores[i, id], 2) for id in answer_ids[i, ::-1]])

        return answers, scores

    def fit(self, x_train_vects: Tuple[Union[csr_matrix, List]], y_train: Tuple[str]) -> None:
        """Train classifier

        Parameters:
            x_train_vects: vectorized question for train dataset
            y_train: answers for train dataset

        Returns:
            None
        """
        if len(x_train_vects) != 0:
            if isinstance(x_train_vects[0], csr_matrix):
                self.x_train_features = vstack(list(x_train_vects))
            elif isinstance(x_train_vects[0], np.ndarray):
                self.x_train_features = np.vstack(list(x_train_vects))
            else:
                raise NotImplementedError('Not implemented this type of vectors')
        else:
            raise ValueError("Train vectors can't be empty")

        self.y_train = list(y_train)

    def save(self) -> None:
        """Save classifier parameters"""
        logger.info("Saving faq_model to {}".format(self.save_path))
        path = expand_path(self.save_path)
        make_all_dirs(path)
        save_pickle((self.x_train_features, self.y_train), path)

    def load(self) -> None:
        """Load classifier parameters"""
        logger.info("Loading faq_model from {}".format(self.load_path))
        self.x_train_features, self.y_train = load_pickle(expand_path(self.load_path))