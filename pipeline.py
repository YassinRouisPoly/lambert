# region IMPORTS ===
import shutil, os, logging, uuid, subprocess, torch
from mlflow.store.db_migrations.versions.bd07f7e963c5_create_index_on_run_uuid import depends_on
from torch.utils.data import Dataset, Subset
from transformers import (
    CamembertTokenizer, AutoModel, AutoTokenizer
)
from sklearn.model_selection import train_test_split
from zenml import step, pipeline
from typing import Annotated, Tuple
from libs.types import TextDataset, compute_metrics
from torch.utils.data import Dataset
from transformers import (
    CamembertTokenizer,
    CamembertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import pandas as pd
# endregion


@step #region all:pull_dvc
def pull_dvc():
    try:
        subprocess.run(["dvc", "pull"], check=True)
        return True
    except FileNotFoundError:
        print("Pas de DVC")
        return False
#endregion
@step # region model:load_dataset
def load_datasets(version="latest", dataset_split=0.01, base_model="camembert-base", depends_on=None) -> Tuple[
    Annotated[Subset, "dataset_train"],
    Annotated[Subset, "dataset_test"]
]:
    dataset_file = "./datasets/" + version + "/data.csv"
    tokenizer = CamembertTokenizer.from_pretrained(base_model)

    dataset = TextDataset(dataset_file, tokenizer)

    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=dataset_split, random_state=42)

    dataset_train = Subset(dataset, train_idx)
    dataset_test = Subset(dataset, test_idx)

    return dataset_train, dataset_test
# endregion
@step #region model:train_model
def train_model(
    train_dataset: Annotated[Subset, "dataset_train"],
    eval_dataset: Annotated[Subset, "dataset_test"],
    output_dir="./cache/output",
    base_model="camembert-base",
    epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
):
    logger = logging.getLogger(__name__)
    model = CamembertForMaskedLM.from_pretrained(base_model)
    tokenizer = CamembertTokenizer.from_pretrained(base_model)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Préparation du Data Collator")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    logger.info("Préparation du modèle [1/2]")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
        eval_strategy="epoch",
        eval_steps=100
    )
    logger.info("Préparation du modèle [2/2]")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    logger.info("Entrainement ... (cela peut prendre un moment)")
    trainer.train()
    model_cache = str(uuid.uuid4())
    model.save_pretrained("./cache/trains/"+model_cache+"/model")
    tokenizer.save_pretrained("./cache/trains/"+model_cache+"/tokenizer")
    return model_cache
# endregion
@step #region model:save_model
def save_cache(cache_uuid: str, version: str, overwrite=False):

    shutil.copytree("./cache/trains/"+cache_uuid+"/model", "./models/"+version+"/model", dirs_exist_ok=overwrite)
    shutil.copytree("./cache/trains/"+cache_uuid+"/tokenizer", "./models/"+version+"/tokenizer", dirs_exist_ok=overwrite)
    return True
# endregion

#region (data cleaner)
def cleaner(text, col):
    text_col = text[col]
    text_col = text_col.str.strip()
    text_col = text_col.str.replace(r"L\.\s?[0-9\-]+", "[article]")
    text_col = text_col.str.replace(r"[0-9]+", "[nombre]")
    text_col = text_col.str.replace(r"[\t\n]+", "", regex=True)
    text_col = text_col.str.replace(r"[^a-zA-Z0-9À-ÿ\s\-\']", "", regex=True)
    return text_col
#endregion

@step #region data:load_raw_dataset
def load_raw_dataset(dataset_path = "./datasets/original/cold-french-law.csv", depends_on=None):
    raw_df = pd.read_csv(dataset_path)
    return raw_df
#endregion
@step #region data:extract_lines
def extract_lines(df, max_lines=5000):
    return df.head(max_lines)
#endregion
@step #region data:treat_dataset
def treat_dataset(df, target_column="article_contenu_text"):
    df[target_column] = cleaner(df, target_column)
    df[target_column] = df[target_column].str.split(r"(?<![A-Z])\.")
    df = df.explode(target_column).reset_index(drop=True)

    return df
# endregion
@step #region data:save_dataset
def save_dataset(df, version = "v0.1-beta", update_latest=True):
    os.makedirs("./datasets/"+version+"/", exist_ok=True)
    df.to_csv("./datasets/"+version+"/data.csv")
    if update_latest:
        df.to_csv("./datasets/latest/data.csv")
    return True
#endregion
@step #region all:dvc_push_models
def dvc_push_models(clear_cache_with=None, depends_on=None):
    try:
        subprocess.run(["dvc", "add", "models"], check=True)
        subprocess.run(["dvc", "commit"], check=True)
        subprocess.run(["dvc", "push"], check=True)
        return True
    except FileNotFoundError:
        print("Pas de DVC")
        return False
#endregion
@step #region all:dvc_push_datasets
def dvc_push_datasets(clear_cache_with=None, depends_on=None):
    try:
        subprocess.run(["dvc", "add", "datasets"], check=True)
        subprocess.run(["dvc", "commit"], check=True)
        subprocess.run(["dvc", "push"], check=True)
        return True
    except FileNotFoundError:
        print("Pas de DVC")
        return False
#endregion

@pipeline #region dataset_pipeline
def pipeline_dataset(version):
    res = pull_dvc()
    if res:
        df = load_raw_dataset(depends_on=res)
        df_r = extract_lines(df)
        df_t = treat_dataset(df_r)
        _ = save_dataset(df_t, version=version)
        dvc_push_datasets(depends_on=_, clear_cache_with=version)


def _get_model(model_cache):
    return CamembertForMaskedLM.from_pretrained("./cache/trains/"+model_cache+"/model")

def _eval_model(trainer):
    return trainer.evaluate()

@pipeline # region pipeline_test
def pipeline_test():
    res = pull_dvc()
    if res:
        dataset_train, dataset_test = load_datasets(version="latest", depends_on=res)
        model_cache = train_model(
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            output_dir="./cache/output",
            base_model="camembert-base",
            epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
        )
        res = save_cache(model_cache, "test", overwrite=True)
        dvc_push_models(depends_on=res, clear_cache_with=uuid.uuid4())
# enregion

if __name__ == "__main__":
    # pipeline_test()
    pipeline_dataset(version="v1")
