import logging
from typing import Dict

from repsys.config import Config
from repsys.dataset import Dataset
from repsys.evaluators import DatasetEvaluator, ModelEvaluator
from repsys.model import Model
from repsys.server import run_server

logger = logging.getLogger(__name__)


def split_dataset(config: Config, dataset: Dataset):
    logger.info("Creating train/validation/test split")
    dataset.fit(
        config.dataset.train_split_prop,
        config.dataset.test_holdout_prop,
        config.dataset.min_user_interacts,
        config.dataset.min_item_interacts,
        config.seed,
    )

    logger.info(f"Saving splits into '{config.checkpoints_dir}'")
    dataset.save(config.checkpoints_dir)

    logger.info("Splitting successfully finished")
    logger.warning("DON'T FORGET TO RETRAIN YOUR MODELS! ")


def fit_models(models: Dict[str, Model], dataset: Dataset, config: Config, training: bool = False):
    for model in models.values():
        if training:
            logger.info(f"Training '{model.name()}' model")
        else:
            logger.info(f"Fitting '{model.name()}' model")

        model.config = config
        model.dataset = dataset
        model.fit(training=training)


def start_server(config: Config, models: Dict[str, Model], dataset: Dataset):
    logger.info("Starting web application server")

    dataset.load(config.checkpoints_dir)
    fit_models(models, dataset, config)

    logger.info("Loading dataset evaluation")
    dataset_eval = DatasetEvaluator(dataset)
    dataset_eval.load(config.checkpoints_dir)

    logger.info("Loading models evaluation")
    model_eval = ModelEvaluator(dataset)
    model_eval.load(config.checkpoints_dir, list(models.keys()), load_prev=True)

    run_server(config, models, dataset, dataset_eval, model_eval)


def train_models(config: Config, models: Dict[str, Model], dataset: Dataset, model_name: str = None):
    logger.info("Training implemented models")

    dataset.load(config.checkpoints_dir)

    if model_name is not None:
        models = {model_name: models.get(model_name)}

    fit_models(models, dataset, config, training=True)


def evaluate_dataset(
    config: Config,
    models: Dict[str, Model],
    dataset: Dataset,
    method: str,
    model_name: str,
):
    logger.info(f"Evaluating implemented dataset using '{method}' method")

    dataset.load(config.checkpoints_dir)

    if model_name is not None:
        models = {model_name: models.get(model_name)}

    fit_models(models, dataset, config)

    evaluator = DatasetEvaluator(dataset)

    model = models.get(model_name)

    logger.info(f"Computing embeddings")
    evaluator.compute_user_embeddings("train", method, model, max_samples=10000)
    evaluator.compute_user_embeddings("validation", method, model)

    evaluator.compute_item_embeddings(method, model)

    evaluator.save(config.checkpoints_dir)


def evaluate_models(
    config: Config,
    models: Dict[str, Model],
    dataset: Dataset,
    split_type: str,
    model_name: str,
):
    logger.info("Evaluating implemented models")

    dataset.load(config.checkpoints_dir)

    if model_name is not None:
        models = {model_name: models.get(model_name)}

    fit_models(models, dataset, config)

    evaluator = ModelEvaluator(
        dataset,
        precision_recall_k=config.eval.precision_recall_k,
        ndcg_k=config.eval.ndcg_k,
        coverage_k=config.eval.coverage_k,
        diversity_k=config.eval.diversity_k,
        novelty_k=config.eval.novelty_k,
        coverage_lt_k=config.eval.coverage_lt_k,
        percentage_lt_k=config.eval.percentage_lt_k,
    )

    for model in models.values():
        logger.info(f"Evaluating '{model.name()}' model")
        evaluator.evaluate(model, split_type)

    evaluator.print()
    evaluator.save(config.checkpoints_dir)
