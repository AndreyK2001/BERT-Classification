import torch

from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import BertForSequenceClassification
import numpy as np
import pandas as pd
import os


def load_data(language="en"):
    label_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    data = pd.concat(
        [
            pd.read_excel(
                f"/content/readme/dataset/{language}/readme_{language}_train.xlsx"
            ),
            pd.read_excel(
                f"/content/readme/dataset/{language}/readme_{language}_val.xlsx"
            ),
            pd.read_excel(
                f"/content/readme/dataset/{language}/readme_{language}_test.xlsx"
            ),
        ],
        axis=0,
    )

    return data["Sentence"][
        data["Sentence"].apply(lambda x: isinstance(x, str))
    ].values, data["Rating"][
        data["Sentence"].apply(lambda x: isinstance(x, str))
    ].replace(label_dict).values


class ReadMePipeline:
    def __init__(self, model_language="en"):
        link_dict = {
            "en": "https://disk.yandex.com/d/UN5pmzv66OM1Pg",
            "ar": "https://disk.yandex.com/d/_xiXiuYJN6L4ww",
            "ru": "https://disk.yandex.com/d/n9euMqTL1xkDMw",
            "hi": "https://disk.yandex.com/d/sF9kPoIVUQnSIg",
            "fr": "https://disk.yandex.com/d/AFgnXRBNAvc3Rw",
        }
        os.system(
            f'wget "https://getfile.dokpub.com/yandex/get/{link_dict[model_language]}" -O ReadMe_{model_language}'
        )

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=6,
            output_attentions=False,
            output_hidden_states=True,
        )

        stete_dict = torch.load(f"./ReadMe_{model_language}", map_location="cpu")
        self.model.load_state_dict(stete_dict)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        model = self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased", do_lower_case=True
        )

    def get_neurons_activations(self, dataloader):
        self.model.eval()

        # Подготовка списка для хранения усредненных скрытых состояний по слоям
        avg_hidens_per_layer = [
            [] for _ in range(13)
        ]  # предполагается, что у модели 12 слоев + входной слой

        for batch in dataloader:
            batch = tuple(b.to(self.device) for b in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Создание маски для исключения [PAD] токенов (где input_ids равно 0)
            mask = (
                inputs["input_ids"] != 0
            )  # Создается маска размером как input_ids, True где не PAD
            mask = mask.unsqueeze(-1).expand_as(
                outputs["hidden_states"][0]
            )  # Расширение маски до размера скрытых состояний

            for layer_idx, layer_hidden_states in enumerate(outputs["hidden_states"]):
                # Применение маски к скрытым состояниям
                masked_hidden_states = layer_hidden_states * mask.float()
                # Вычисление суммы и количества не-pad токенов для усреднения
                sum_hidden_states = masked_hidden_states.sum(
                    dim=1
                )  # Сумма по оси токенов
                non_pad_tokens = mask.sum(dim=1)  # Количество не-pad токенов
                # Усреднение скрытых состояний, исключая pad-токены
                avg_hidden_states = sum_hidden_states / non_pad_tokens.clamp(
                    min=1
                )  # Избегание деления на 0
                avg_hidens_per_layer[layer_idx].append(avg_hidden_states)

        # Собираем усредненные скрытые состояния по всему датасету для каждого слоя
        avg_hidens_stacked_per_layer = [
            torch.cat(layer_avg_hidens, dim=0)
            for layer_avg_hidens in avg_hidens_per_layer
        ]

        # Стекинг усредненных скрытых состояний для всех слоев
        all_avg_hidens_tensor = torch.stack(avg_hidens_stacked_per_layer)

        return all_avg_hidens_tensor  # [layers, batch, neurons]

    def predict_dataloader(self, dataloader):
        self.model.eval()

        predictions = []

        for batch in dataloader:
            batch = tuple(b.to(self.device) for b in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits

            logits = logits.detach().cpu().numpy()
            predictions.append(logits)

        predictions = np.concatenate(predictions, axis=0)

        return predictions

    def predict_proba(self, data):
        encoded_data_predict = self.tokenizer.batch_encode_plus(
            data,
            add_special_tokens=True,
            return_attention_mask=True,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        input_ids_predict = encoded_data_predict["input_ids"]
        attention_masks_predict = encoded_data_predict["attention_mask"]
        dataset_predict = TensorDataset(input_ids_predict, attention_masks_predict)

        batch_size = 10
        dataloader_predict = DataLoader(
            dataset_predict,
            sampler=SequentialSampler(dataset_predict),
            batch_size=batch_size,
        )

        predictions = self.predict_dataloader(dataloader_predict)

        return predictions

    def predict(self, data):
        predictions = self.predict_proba(data)
        return np.argmax(predictions, axis=1).flatten()
