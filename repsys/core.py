import logging
from typing import Dict

from repsys.dataset import Dataset
from repsys.evaluators import DatasetEvaluator, ModelEvaluator
from repsys.model import Model
from repsys.server import run_server

logger = logging.getLogger(__name__)


def split_dataset(dataset: Dataset, output_path: str):
    logger.info("Creating splits of the input data ...")
    dataset.split()

    logger.info(f"Saving the split into '{output_path}'")
    dataset.save(output_path)

    logger.warning("DON'T FORGET TO RETRAIN YOUR MODELS! ")


def fit_models(models: Dict[str, Model], dataset: Dataset, training: bool):
    for model in models.values():
        logger.info(f"Fitting model '{model.name()}' ...")
        model.update_dataset(dataset)
        model.fit(training=training)


def start_server(models: Dict[str, Model], dataset: Dataset, split_path: str, dataset_eval_path: str,
                 models_eval_path: str):
    dataset.load(split_path)
    fit_models(models, dataset, training=False)

    dataset_evaluator = DatasetEvaluator()
    dataset_evaluator.update_dataset(dataset)

    models_evaluator = ModelEvaluator()
    models_evaluator.update_dataset(dataset)

    if models_eval_path is not None:
        logger.info('Loading models evaluations ...')
        models_evaluator.load(models_eval_path)

    if dataset_eval_path is not None:
        logger.info('Loading dataset evaluations ...')
        dataset_evaluator.load(dataset_eval_path)

    run_server(models, dataset, dataset_evaluator, models_evaluator)


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


def evaluate_models(models: Dict[str, Model], dataset: Dataset, split_path: str, split_type: str, output_path: str):
    dataset.load(split_path)

    fit_models(models, dataset, training=False)

    evaluator = ModelEvaluator()
    evaluator.update_dataset(dataset)

    for model in models.values():
        logger.info(f"Evaluating model '{model.name()}' ...")

        evaluator.evaluate(model, split_type)
        evaluator.save(output_path)
