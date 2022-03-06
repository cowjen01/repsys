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
    dataset.fit(config.dataset.train_split_prop, config.dataset.test_holdout_prop, config.dataset.min_user_interacts,
                config.dataset.min_item_interacts, config.seed)

    logger.info(f"Saving splits into '{config.checkpoints_dir}'")
    dataset.save(config.checkpoints_dir)

    logger.info("Splitting successfully finished")
    logger.warning("DON'T FORGET TO RETRAIN YOUR MODELS! ")


def fit_models(models: Dict[str, Model], dataset: Dataset, config: Config, training: bool):
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
    fit_models(models, dataset, config, training=False)

    logger.info("Loading embeddings (train split)")
    dataset_eval_train = DatasetEvaluator(dataset, split='train')
    dataset_eval_train.load(config.checkpoints_dir)

    logger.info("Loading embeddings (validation split)")
    dataset_eval_vad = DatasetEvaluator(dataset, split='validation')
    dataset_eval_vad.load(config.checkpoints_dir)

    dataset_eval = {
        'train': dataset_eval_train,
        'validation': dataset_eval_vad
    }

    logger.info("Loading models evaluation")
    model_eval = ModelEvaluator(dataset)
    model_eval.load(config.checkpoints_dir, list(models.keys()), history=2)

    run_server(config, models, dataset, dataset_eval, model_eval)


def train_models(config: Config, models: Dict[str, Model], dataset: Dataset, model_name: str = None):
    logger.info("Training implemented models")

    if model_name is not None:
        model = models.get(model_name)
        models = {model_name: model}

    dataset.load(config.checkpoints_dir)

    fit_models(models, dataset, config, training=True)


def evaluate_dataset(config: Config, dataset: Dataset, method: str):
    logger.info("Evaluating implemented dataset")

    dataset.load(config.checkpoints_dir)

    for split in ['train', 'validation']:
        evaluator = DatasetEvaluator(dataset, split)

        logger.info(f"Computing embeddings ({split} split)")
        evaluator.compute_embeddings(method=method, max_samples=10000)
        evaluator.save(config.checkpoints_dir)


def evaluate_models(config: Config, models: Dict[str, Model], dataset: Dataset, split_type: str, model_name: str):
    logger.info("Evaluating implemented models")

    dataset.load(config.checkpoints_dir)

    if model_name is not None:
        models = {model_name: models.get(model_name)}

    fit_models(models, dataset, config, training=False)

    evaluator = ModelEvaluator(dataset, rp_k=config.eval.rp_k, ndcg_k=config.eval.ndcg_k,
                               coverage_k=config.eval.coverage_k)
    evaluator.load(config.checkpoints_dir, list(models.keys()), history=1)

    for model in models.values():
        logger.info(f"Evaluating '{model.name()}' model")
        evaluator.evaluate(model, split_type)

    evaluator.print()
    evaluator.save(config.checkpoints_dir)
