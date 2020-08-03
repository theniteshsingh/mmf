# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, Dict, Type

import torch
from torch import Tensor, nn
from transformers.modeling_bert import BertPooler, BertPredictionHeadTransform

from mmf.common.registry import registry
from mmf.common.typings import DictConfig
from mmf.models.transformers.base import (
    BaseTransformer,
    BaseTransformerConfigType,
    BaseTransformerInput,
)
from mmf.modules.encoders import MultiModalEncoderBase


class ImageEncoder(MultiModalEncoderBase):
    """Extends the MultiModalEncoderBase class which builds the encoder based on
    the config parameters. We can set the type of image encoder(resnet50, resnet152,
    resnext50 etc) and other parameters like num of features, type of pooling etc.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def build(self):
        self.encoder = self._build_modal_encoder(self.config.image_encoder)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class MMFTransformerEmbeddings(nn.Module):
    """Embedding class takes two types of modalities(image and text), each can
    have their input id, position id and segment id. We generate embeddings of
    dimension config.hidden_size for each and then first add the three embeddings
    for each modality to have a modality specific embedding. We then concat the
    modality specific embeddings to have a joint embedding.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        transformer: Type[nn.Module],
        img_dim: int,
        img_pos_dim: int,
    ):
        super().__init__()

        # Text Embeddings
        self.word_embeddings = transformer.embeddings.word_embeddings
        self.position_embeddings = transformer.embeddings.position_embeddings
        self.layer_norm = transformer.embeddings.LayerNorm
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Image Embeddings
        self.img_embeddings = nn.Sequential(
            nn.Linear(img_dim, config.hidden_size),
            torch.nn.LayerNorm(config.hidden_size, eps=1e-12),
        )
        self.img_pos_embeddings = nn.Sequential(
            nn.Linear(img_pos_dim, config.hidden_size),
        )
        self.img_layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.img_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Token Type Embeddings
        self.token_type_embeddings = transformer.embeddings.token_type_embeddings

    def forward(
        self,
        input_ids: Dict[str, Tensor],
        position_ids: Dict[str, Tensor],
        segment_ids: Dict[str, Tensor],
    ) -> Tensor:
        ## Calculate text embeddings for word, position, segment type
        words_embeddings = self.word_embeddings(input_ids["text"])
        # Add position ids for text tokens
        if "text" not in position_ids:
            position_ids["text"] = input_ids["text"].new_tensor(
                torch.arange(0, input_ids["text"].size(1), dtype=torch.long)
                .unsqueeze(0)
                .expand(input_ids["text"].size(0), input_ids["text"].size(1))
            )
        position_embeddings = self.position_embeddings(position_ids["text"])
        if "text" not in segment_ids:
            segment_ids["text"] = torch.zeros_like(input_ids["text"])
        txt_type_embeddings = self.token_type_embeddings(segment_ids["text"])

        txt_embeddings = self.layer_norm(
            words_embeddings + position_embeddings + txt_type_embeddings
        )
        txt_embeddings = self.dropout(txt_embeddings)

        ## Calculate image embeddings for feature, position, segment type
        transformed_input = self.img_embeddings(input_ids["image"])
        img_embeddings = transformed_input
        if "image" in position_ids:
            transformed_pos = self.position_embeddings(position_ids["image"])
            img_embeddings += transformed_pos

        if "image" not in segment_ids:
            segment_ids["image"] = torch.zeros_like(
                input_ids["image"][:, :, 0], dtype=torch.long
            )
        img_type_embeddings = self.token_type_embeddings(segment_ids["image"])
        img_embeddings += img_type_embeddings

        img_embeddings = self.img_dropout(self.img_layer_norm(img_embeddings))

        return torch.cat([txt_embeddings, img_embeddings], dim=1)


@registry.register_model("mmf_transformer")
class MMFTransformer(BaseTransformer):
    def __init__(self, config: BaseTransformerConfigType):
        super().__init__(config)

    @classmethod
    def config_path(cls) -> str:
        return "configs/models/mmf_transformer/defaults.yaml"

    def build_encoders(self):
        self.image_encoder = ImageEncoder(self.config)

    def build_embeddings(self):
        """Here we initialize the embedding class we will use for multiple
        modalities (here just text and image). For the text embeeddings we will use the
        pretrained weights from the trasnformer model rather than training from scratch.
        """
        self.embeddings = MMFTransformerEmbeddings(
            self.transformer_config,
            self.transformer,
            self.config.visual_embedding_dim,
            self.config.visual_position_dim,
        )

    def build_heads(self):
        """Here we initialize the classifier head. It takes the output of the
        transformer encoder and passes it through a pooler (we use the pooler from BERT
        model), then dropout, BertPredictionHeadTransform (which is a liner layer,
        followed by activation and layer norm) and lastly a linear layer projecting the
        hidden output to classification labels.
        """
        self.classifier = nn.Sequential(
            BertPooler(self.transformer_config),
            nn.Dropout(self.transformer_config.hidden_dropout_prob),
            BertPredictionHeadTransform(self.transformer_config),
            nn.Linear(self.transformer_config.hidden_size, self.config.num_labels),
        )

    def preprocess_sample(self, sample_list: Dict[str, Any]) -> BaseTransformerInput:
        """Here we preprocess the sample list elements and form a BaseTransformerInput
        type object. This object standardizes how we represent multiple modalities.
        Check the definition of this dataclass in BaseTransformer.
        """

        # Input IDs (or text tokens/image features)
        input_ids: Dict[str, Tensor] = {}
        input_ids["text"] = sample_list.input_ids
        if "image_feature_0" in sample_list:
            input_ids["image"] = sample_list.image_feature_0
        elif "image" in sample_list:
            input_ids["image"] = self.image_encoder(sample_list.image)

        # Position IDs
        position_ids: Dict[str, Tensor] = {}
        position_ids["image"] = input_ids["image"].new_tensor(
            torch.arange(0, input_ids["image"].size(1), dtype=torch.long)
            .unsqueeze(0)
            .expand(input_ids["image"].size(0), input_ids["image"].size(1)),
            dtype=torch.long,
        )

        # Segment IDs
        segment_ids: Dict[str, Tensor] = {}
        segment_ids["text"] = sample_list.segment_ids

        # Masks
        masks: Dict[str, Tensor] = {}
        masks["text"] = sample_list.input_mask
        if "image_mask" in sample_list:
            masks["image"] = sample_list.image_mask
        else:
            masks["image"] = torch.ones_like(
                input_ids["image"][:, :, 0], dtype=torch.long
            )

        return BaseTransformerInput(input_ids, position_ids, segment_ids, masks)

    def forward(self, sample_list: Dict[str, Any]) -> Dict[str, Tensor]:
        # Sample preprocess
        output = self.preprocess_sample(sample_list)

        # Transformer Input Embeddings
        embedding_output = self.embeddings(
            input_ids=output.input_ids,
            position_ids=output.position_ids,
            segment_ids=output.segment_ids,
        )

        # Transformer Attention mask
        # concat the attention masks for text and image
        attention_mask = torch.cat(
            (output.masks["text"], output.masks["image"]), dim=-1
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Transformer Encoder
        encoded_layers = self.transformer.encoder(
            embedding_output,  # combined embedding
            extended_attention_mask,  # combined attention mask
            [None] * len(self.transformer.encoder.layer),  # head masks
        )

        # Transformer Heads
        head_output = self.classifier(encoded_layers[0])

        # Postprocess outputs
        return self.postprocess_output(head_output)

    def postprocess_output(self, output: Tensor) -> Dict[str, Tensor]:
        """Here we postprocess the output from the classifier head and reshape it.
        This will be used to calculate losses and metrics in mmf.
        """
        output_dict = {}
        output_dict["scores"] = output.contiguous().view(-1, self.config.num_labels)
        return output_dict
