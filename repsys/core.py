import logging
from typing import Dict

from repsys.dataset import Dataset
from repsys.evaluators import DatasetEvaluator
from repsys.model import Model
from repsys.server import run_server

logger = logging.getLogger(__name__)


def split_dataset(dataset: Dataset, output_path: str):
    logger.info("Creating splits of the input data ...")
    dataset.split()

    logger.info(f"Saving the split into '{output_path}'")
    dataset.save(output_path)


def fit_models(models: Dict[str, Model], dataset: Dataset, training: bool):
    for model in models.values():
        logger.info(f"Fitting model '{model.name()}' ...")
        model.update_dataset(dataset)
        model.fit(training=training)


def start_server(models: Dict[str, Model], dataset: Dataset, split_path: str, dataset_eval_path: str, port: int):
    dataset.load(split_path)
    fit_models(models, dataset, training=False)

    dataset_evaluator = DatasetEvaluator()
    dataset_evaluator.update_dataset(dataset)

    if dataset_eval_path is not None:
        logger.info('Loading dataset evaluations ...')
        dataset_evaluator.load(dataset_eval_path)

    run_server(port, models, dataset, dataset_evaluator)


def train_models(models: Dict[str, Model], dataset: Dataset, split_path: str):
    dataset.load(split_path)
    fit_models(models, dataset, training=True)


def evaluate_dataset(dataset: Dataset, split_path: str, output_path: str):
    dataset.load(split_path)

    evaluator = DatasetEvaluator()
    evaluator.update_dataset(dataset)
    evaluator.compute_embeddings('train')
    evaluator.compute_embeddings('validation')
    evaluator.save(output_path)

# def evaluate_models(models: Dict[str, Model], dataset: Dataset, data_type: str, output_path: str):
#     create_tmp_dir()
#
#     evaluator = ModelEvaluator()
#     evaluator.update_dataset(dataset)
#     for model in models.values():
#         logger.info(f"Evaluating model '{model.name()}' ...")
#
#         if data_type == "test":
#             evaluator.test_model_eval(model)
#         else:
#             evaluator.vad_model_eval(model)
#
#         evaluator.print()
#
#         eval_dir = os.path.join(tmp_dir_path(), model.name())
#         create_dir(eval_dir)
#         evaluator.save(eval_dir)
#
#     zip_dir(output_path, tmp_dir_path())
#     remove_tmp_dir()
