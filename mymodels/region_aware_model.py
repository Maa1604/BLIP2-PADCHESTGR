import torch
import torch.nn as nn
from typing import Optional, Union
from transformers import Blip2ForConditionalGeneration, Blip2Config
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput



#https://huggingface.co/docs/transformers/model_doc/blip-2#transformers.Blip2Model.forward
#https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L1590
class RegionBlip2ForConditionalGeneration(Blip2ForConditionalGeneration):
    """
    BLIP-2 with extra 'region tokens' that are concatenated to the query tokens for the Q-Former.
    Only the first num_query_tokens outputs (Z^) are projected to the language model.
    """

    def __init__(self, config: Blip2Config):
        super().__init__(config)
        # An embedding table for region tokens in Q-Former hidden size (not LM hidden size).
        # We reuse the LM vocab size so you can tokenize the region string with the processor's tokenizer.
        self.region_token_embed = nn.Embedding(
            num_embeddings=config.text_config.vocab_size,
            embedding_dim=config.qformer_config.hidden_size,
        )

    # ---- internal helper ----
    def _build_qformer_inputs(
        self,
        batch_size: int,
        region_input_ids: Optional[torch.LongTensor],
        device: torch.device,
    ):
        """
        Returns:
            query_and_region_embeds: [B, Z_len + W_len, H_q]
            z_len: int (= num_query_tokens)
        """
        z_len = self.config.num_query_tokens

        # Z: learned query tokens (kept in FP32 per base class)
        Z = self.query_tokens.expand(batch_size, -1, -1)  # [B, Z_len, H_q]

        # W: region tokens (optional)
        if region_input_ids is not None and region_input_ids.numel() > 0:
            W = self.region_token_embed(region_input_ids.to(device))  # [B, W_len, H_q]
            query_and_region = torch.cat([Z, W], dim=1)               # [B, Z_len + W_len, H_q]
        else:
            query_and_region = Z

        return query_and_region, z_len

    # ---- override: image features with region ----
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = False,
        region_input_ids: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes images to LM-ready embeddings. If region_input_ids is provided, concatenate
        region token embeddings to query tokens before the Q-Former and later keep only Z^.
        """
        # 1) Vision encoder
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
        )
        image_embeds = vision_outputs[0]  # [B, V_seq, H_v]

        # 2) Q-Former with [Z, W] as query_embeds
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_and_region_embeds, z_len = self._build_qformer_inputs(
            batch_size=image_embeds.size(0),
            region_input_ids=region_input_ids,
            device=image_embeds.device,
        )

        q_outputs = self.qformer(
            query_embeds=query_and_region_embeds,             # [B, Z+W, H_q]
            encoder_hidden_states=image_embeds,               # I
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        q_out = q_outputs[0]  # [B, Z+W, H_q]

        # Keep only Z^ (first num_query_tokens positions)
        z_out = q_out[:, :z_len, :]  # [B, Z, H_q]

        # Q-Former is fp32; downcast to match vision dtype if needed
        if z_out.dtype != image_embeds.dtype:
            z_out = z_out.to(image_embeds.dtype)

        # 3) Project Z^ to LM hidden size
        lm_inputs = self.language_projection(z_out)  # [B, Z, H_lm]

        if return_dict:
            return lm_inputs, vision_outputs, q_outputs
        return lm_inputs

    # ---- override: forward ----
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        region_input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Get only Z^ projected embeddings, plus debug outputs if desired
        lm_img_embeds, vision_outputs, qformer_outputs = self.get_image_features(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
            region_input_ids=region_input_ids,
        )

        # Text side
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Replace special image placeholders with the projected Z^ embeddings
        lm_img_embeds = lm_img_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
        inputs_embeds = inputs_embeds.to(lm_img_embeds.device).masked_scatter(
            special_image_mask, lm_img_embeds
        )

        # Run LM (unchanged)
        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )
            logits = outputs[0]
            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)
                loss_fct = nn.CrossEntropyLoss(reduction="mean")
                loss = loss_fct(
                    shift_logits.view(-1, self.config.text_config.vocab_size),
                    shift_labels.view(-1)
                )
        else:
            kwargs["return_dict"] = True
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                **kwargs,
            )
            loss = outputs.loss
            logits = outputs.logits

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=qformer_outputs,
            language_model_outputs=outputs,
        )

    # ---- override: generate ----
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
        region_input_ids: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        # accelerate prep if needed
        if hasattr(self, "hf_device_map"):
            self._preprocess_accelerate()

        B = pixel_values.shape[0]

        # Vision
        image_embeds = self.vision_model(
            pixel_values,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        ).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # Q-Former with [Z, W]
        query_and_region_embeds, z_len = self._build_qformer_inputs(
            batch_size=B,
            region_input_ids=region_input_ids,
            device=image_embeds.device,
        )
        q_outputs = self.qformer(
            query_embeds=query_and_region_embeds,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        q_out = q_outputs.last_hidden_state  # [B, Z+W, H_q]

        # Keep only Z^ and project
        z_out = q_out[:, :z_len, :]
        if z_out.dtype != image_embeds.dtype:
            z_out = z_out.to(image_embeds.dtype)
        lm_img_embeds = self.language_projection(z_out)  # [B, Z, H_lm]

        # Build initial text stream exactly like HF base
        if inputs_embeds is None:
            if input_ids is None:
                image_tokens = [self.config.image_token_index] * self.config.num_query_tokens
                start_tokens = image_tokens + [self.config.text_config.bos_token_id]
                input_ids = torch.tensor([start_tokens], dtype=torch.long, device=image_embeds.device).repeat(B, 1)
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Replace special image placeholders with Z^ projected embeddings
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        lm_img_embeds = lm_img_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, lm_img_embeds)

        inputs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        if not self.language_model.config.is_encoder_decoder:
            inputs["input_ids"] = input_ids

        return self.language_model.generate(**inputs, **generate_kwargs)
