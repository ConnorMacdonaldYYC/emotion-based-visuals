import os

import anthropic
import torch
from diffusers import AutoPipelineForText2Image
from dotenv import load_dotenv
from PIL import Image
from regex import P
from transformers import pipeline

from latentblending.blending_engine import BlendingEngine

load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


class EmotionBasedImageGenerator:
    def __init__(self, base_prompt: str, max_word_count: int, device: str):
        self.base_prompt = base_prompt
        self.max_word_count = max_word_count
        self.magic_prompt_pipeline = pipeline(
            "text-generation",
            model="Gustavosta/MagicPrompt-Stable-Diffusion",
            device=device,
            framework="pt",
        )
        self.image_generation_pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        self.image_generation_pipeline.to(device)
        self.blending_engine = BlendingEngine(self.image_generation_pipeline)

    def generate_enhanced_prompt(self, emotion_list: list[str]) -> str:
        prompt = self._create_image_prompt_from_emotions(
            self.base_prompt, emotion_list, self.max_word_count
        )
        return self._enhance_prompt_with_magic_prompt(prompt)

    def generate_image_from_emotions(
        self, emotion_list: list[str]
    ) -> list[Image.Image]:

        prompt = self.generate_enhanced_prompt(emotion_list)
        print(f"PROMPT {prompt} ------------------------------")
        return self._generate_blended_images(prompt)

    def _generate_blended_images(self, prompt: str) -> list[Image.Image]:
        if self.blending_engine.prompt1 is None or self.blending_engine.prompt1 == "":
            self.blending_engine.set_prompt1(prompt)
            return [self.blending_engine.compute_latents1(return_image=True)]

        self.blending_engine.set_prompt2(prompt)
        transition_images = self.blending_engine.run_transition(True)
        self.blending_engine.swap_forward()
        return transition_images

    def _create_image_prompt_from_emotions(
        self, description_prompt: str, emotion_list: list[str], max_word_count: int
    ) -> str:
        # Initialize the Anthropic client
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

        # Construct the message to Claude
        message = f"Given the following description: '{description_prompt}' and the list of emotions: {emotion_list}, create an image prompt that incorporates these elements. The prompt should be detailed and vivid, suitable for an image generation AI. With a maximum length of {max_word_count} words. Return only the image prompt, no intro, no explanation, no commentary, just the prompt."

        # Send the message to Claude and get the response
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": message}],
            max_tokens=max_word_count * 10,
        )

        # Extract the generated prompt from Claude's response
        generated_prompt = response.content[0].text

        return generated_prompt

    def _enhance_prompt_with_magic_prompt(self, prompt: str) -> str:

        # Generate enhanced prompt
        enhanced_prompt = self.magic_prompt_pipeline(
            prompt,
            max_length=len(prompt.split()) + 50,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            clean_up_tokenization_spaces=True,
            truncation=True,
        )

        # Extract the generated text
        enhanced_prompt = enhanced_prompt[0]["generated_text"]

        return enhanced_prompt
