#!/usr/bin/env python

from __future__ import annotations

import os
import random

import gradio as gr
import numpy as np
import PIL.Image
import torch
from diffusers import DiffusionPipeline

DESCRIPTION = '# SD-XL'
if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>在CPU 🥶上运行～这个演示不能在CPU上运行。</p>'

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv(
    'CACHE_EXAMPLES') == '1'
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '1024'))
USE_TORCH_COMPILE = os.getenv('USE_TORCH_COMPILE') == '1'
ENABLE_CPU_OFFLOAD = os.getenv('ENABLE_CPU_OFFLOAD') == '1'
ENABLE_REFINER = os.getenv('ENABLE_REFINER', '1') == '1'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    pipe = DiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant='fp16')
    if ENABLE_REFINER:
        refiner = DiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-refiner-1.0',
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant='fp16')

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
        if ENABLE_REFINER:
            refiner.enable_model_cpu_offload()
    else:
        pipe.to(device)
        if ENABLE_REFINER:
            refiner.to(device)

    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet,
                                  mode='reduce-overhead',
                                  fullgraph=True)
        if ENABLE_REFINER:
            refiner.unet = torch.compile(refiner.unet,
                                         mode='reduce-overhead',
                                         fullgraph=True)
else:
    pipe = None
    refiner = None


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def generate(prompt: str,
             negative_prompt: str = '',
             prompt_2: str = '',
             negative_prompt_2: str = '',
             use_negative_prompt: bool = False,
             use_prompt_2: bool = False,
             use_negative_prompt_2: bool = False,
             seed: int = 0,
             width: int = 1024,
             height: int = 1024,
             guidance_scale_base: float = 5.0,
             guidance_scale_refiner: float = 5.0,
             num_inference_steps_base: int = 50,
             num_inference_steps_refiner: int = 50,
             apply_refiner: bool = False) -> PIL.Image.Image:
    generator = torch.Generator().manual_seed(seed)

    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    if not use_prompt_2:
        prompt_2 = None  # type: ignore
    if not use_negative_prompt_2:
        negative_prompt_2 = None  # type: ignore

    if not apply_refiner:
        return pipe(prompt=prompt,
                    negative_prompt=negative_prompt,
                    prompt_2=prompt_2,
                    negative_prompt_2=negative_prompt_2,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale_base,
                    num_inference_steps=num_inference_steps_base,
                    generator=generator,
                    output_type='pil').images[0]
    else:
        latents = pipe(prompt=prompt,
                       negative_prompt=negative_prompt,
                       prompt_2=prompt_2,
                       negative_prompt_2=negative_prompt_2,
                       width=width,
                       height=height,
                       guidance_scale=guidance_scale_base,
                       num_inference_steps=num_inference_steps_base,
                       generator=generator,
                       output_type='latent').images
        image = refiner(prompt=prompt,
                        negative_prompt=negative_prompt,
                        prompt_2=prompt_2,
                        negative_prompt_2=negative_prompt_2,
                        guidance_scale=guidance_scale_refiner,
                        num_inference_steps=num_inference_steps_refiner,
                        image=latents,
                        generator=generator).images[0]
        return image


examples = [
    'Astronaut in a jungle, cold color palette, muted colors, detailed, 8k',
    'An astronaut riding a green horse',
]

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value='Duplicate Space for private use',
                       elem_id='duplicate-button',
                       visible=os.getenv('SHOW_DUPLICATE_BUTTON') == '0')
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label='正面提示词',
                show_label=False,
                max_lines=1,
                placeholder='输入提示词（最好用英文表达）',
                container=False,
            )
            run_button = gr.Button('生成', scale=0)
        result = gr.Image(label='Result', show_label=False)
    with gr.Accordion('高级选项', open=False):
        with gr.Row():
            use_negative_prompt = gr.Checkbox(label='开启负面提示词',
                                              value=False)
            use_prompt_2 = gr.Checkbox(label='开启正面提示词 2', value=False)
            use_negative_prompt_2 = gr.Checkbox(label='开启负面提示词 2',
                                                value=False)
        negative_prompt = gr.Text(
            label='负面提示词',
            max_lines=1,
            placeholder='输入负面提示词',
            visible=False,
        )
        prompt_2 = gr.Text(
            label='正面提示词 2',
            max_lines=1,
            placeholder='输入提示词',
            visible=False,
        )
        negative_prompt_2 = gr.Text(
            label='负面提示词 2',
            max_lines=1,
            placeholder='输入负面提示词',
            visible=False,
        )

        seed = gr.Slider(label='种子',
                         minimum=0,
                         maximum=MAX_SEED,
                         step=1,
                         value=0)
        randomize_seed = gr.Checkbox(label='随机种子', value=True)
        with gr.Row():
            width = gr.Slider(
                label='宽度',
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )
            height = gr.Slider(
                label='高度',
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )
        apply_refiner = gr.Checkbox(label='开启细化',
                                    value=False,
                                    visible=ENABLE_REFINER)
        with gr.Row():
            guidance_scale_base = gr.Slider(label='推理程度',
                                            minimum=1,
                                            maximum=20,
                                            step=0.1,
                                            value=5.0)
            num_inference_steps_base = gr.Slider(
                label='推理步数',
                minimum=10,
                maximum=100,
                step=1,
                value=50)
        with gr.Row(visible=False) as refiner_params:
            guidance_scale_refiner = gr.Slider(
                label='细化程度',
                minimum=1,
                maximum=20,
                step=0.1,
                value=5.0)
            num_inference_steps_refiner = gr.Slider(
                label='细化步数',
                minimum=10,
                maximum=100,
                step=1,
                value=50)

    gr.Examples(examples=examples,
                inputs=prompt,
                outputs=result,
                fn=generate,
                cache_examples=CACHE_EXAMPLES)

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        queue=False,
        api_name=False,
    )
    use_prompt_2.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_prompt_2,
        outputs=prompt_2,
        queue=False,
        api_name=False,
    )
    use_negative_prompt_2.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt_2,
        outputs=negative_prompt_2,
        queue=False,
        api_name=False,
    )
    apply_refiner.change(
        fn=lambda x: gr.update(visible=x),
        inputs=apply_refiner,
        outputs=refiner_params,
        queue=False,
        api_name=False,
    )

    inputs = [
        prompt,
        negative_prompt,
        prompt_2,
        negative_prompt_2,
        use_negative_prompt,
        use_prompt_2,
        use_negative_prompt_2,
        seed,
        width,
        height,
        guidance_scale_base,
        guidance_scale_refiner,
        num_inference_steps_base,
        num_inference_steps_refiner,
        apply_refiner,
    ]
    prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name='run',
    )
    negative_prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
    prompt_2.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
    negative_prompt_2.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
    run_button.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
demo.queue(max_size=20).launch()
