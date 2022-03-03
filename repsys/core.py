import logging
from typing import Dict

from repsys.config import Config
from repsys.dataset import Dataset
from repsys.evaluators import DatasetEvaluator, ModelEvaluator
from repsys.model import Model
from repsys.server import run_server

logger = logging.getLogger(__name__)


def split_dataset(config: Config, dataset: Dataset):
    logger.info("Creating splits of the input data ...")
    dataset.fit(config.dataset.train_split_prop, config.dataset.test_holdout_prop, config.dataset.min_user_interacts,
                config.dataset.min_item_interacts, config.seed)

    logger.info(f"Saving the split into '{config.checkpoints_dir}'")
    dataset.save(config.checkpoints_dir)

    logger.warning("DON'T FORGET TO RETRAIN YOUR MODELS! ")


def fit_models(models: Dict[str, Model], dataset: Dataset, training: bool):
    for model in models.values():
        logger.info(f"Fitting model '{model.name()}' ...")
        model.update_dataset(dataset)
        model.fit(training=training)


def start_server(config: Config, models: Dict[str, Model], dataset: Dataset):
    dataset.load(config.checkpoints_dir)
    fit_models(models, dataset, training=False)

    dataset_eval_train = DatasetEvaluator(dataset, split='train')
    dataset_eval_train.load(config.checkpoints_dir)

    dataset_eval_vad = DatasetEvaluator(dataset, split='validation')
    dataset_eval_vad.load(config.checkpoints_dir)

    dataset_eval = {
        'train': dataset_eval_train,
        'validation': dataset_eval_vad
    }

    model_eval = ModelEvaluator(dataset)
    model_eval.load(config.checkpoints_dir, list(models.keys()), history=2)

    run_server(config, models, dataset, dataset_eval, model_eval)


def train_models(config: Config, models: Dict[str, Model], dataset: Dataset, model_name: str = None):
    dataset.load(config.checkpoints_dir)

    if model_name is not None:
        model = models.get(model_name)
        models = {model_name: model}

    fit_models(models, dataset, training=True)


def evaluate_dataset(config: Config, dataset: Dataset, method: str):
    dataset.load(config.checkpoints_dir)

    for split in ['train', 'validation']:
        evaluator = DatasetEvaluator(dataset, split)
        evaluator.compute_embeddings(method=method, max_samples=10000)
        evaluator.save(config.checkpoints_dir)


def evaluate_models(config: Config, models: Dict[str, Model], dataset: Dataset, split_type: str, model_name: str):
    dataset.load(config.checkpoints_dir)

    if model_name is not None:
        models = {model_name: models.get(model_name)}

    fit_models(models, dataset, training=False)

    evaluator = ModelEvaluator(dataset)

    for model in models.values():
        logger.info(f"Evaluating model '{model.name()}' ...")
        evaluator.evaluate(model, split_type)

    evaluator.print()
    evaluator.save(config.checkpoints_dir)
