# This code partially references https://github.com/gligen/diffusers/blob/0e09db9d7150126e327ff93cf91857b00f624ee0/examples/research_projects/mulit_token_textual_inversion/multi_token_clip.py
from ptp_inversion import *
from multi_token_clip import MultiTokenCLIPTokenizer
from transformers import CLIPTextModel
import os
from PIL import Image
from datetime import datetime
import time  

import ddim_inversion
import argparse




def add_tokens(tokenizer, text_encoder, placeholder_token, num_vec_per_token=1, initializer_token=None, use_neg=False):
    """
    Add tokens to the tokenizer and set the initial value of token embeddings
    """
    tokenizer.add_placeholder_tokens(placeholder_token, num_vec_per_token=num_vec_per_token)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    placeholder_token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
    print(f"number of placeholder tokens are: {len(placeholder_token_ids)}")
    if initializer_token:
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            
            token_embeds[placeholder_token_id] = token_embeds[token_ids[i * len(token_ids)//num_vec_per_token]]
            if use_neg:
                token_embeds[placeholder_token_id] += torch.randn_like(token_embeds[placeholder_token_id])*1e-3
    else:
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
                token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[placeholder_token_id])
                
                
def load_multitoken_tokenizer(tokenizer, text_encoder, pos_learned_embeds_dict, pos_placeholder_token):
    num_vec_pos_token = pos_learned_embeds_dict[pos_placeholder_token].shape[0]
    add_tokens(tokenizer, text_encoder, pos_placeholder_token, num_vec_per_token=num_vec_pos_token)
    pos_placeholder_token_ids = tokenizer.encode(pos_placeholder_token, add_special_tokens=False)
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for i, placeholder_token_id in enumerate(pos_placeholder_token_ids):
        token_embeds[placeholder_token_id] = pos_learned_embeds_dict[pos_placeholder_token][i]

def image_grid_two_groups(imgs_group1, imgs_group2):
    assert len(imgs_group1) == len(imgs_group2)
    w, h = imgs_group1[0].size
    grid = Image.new('RGB', size=(2*w, len(imgs_group1)*h))
    grid_w, grid_h = grid.size
    for i in range(len(imgs_group1)):
        grid.paste(imgs_group1[i], box=(0, i*h))
        grid.paste(imgs_group2[i], box=(w, i*h))
    return grid

def image_grid_three_groups(imgs_group1, imgs_group2, imgs_group3):
    
    w, h = imgs_group1[0].size
    grid = Image.new('RGB', size=(3*w, len(imgs_group1)*h))
    grid_w, grid_h = grid.size
    for i in range(len(imgs_group1)):
        grid.paste(imgs_group1[i], box=(0, i*h))
        grid.paste(imgs_group2[i], box=(w, i*h))
        grid.paste(imgs_group3[i], box=(2*w, i*h))
    return grid

def get_out_img_path(out_path, input_img_path):
    input_img_name = input_img_path.split('/')[-1].split(".")[-2]
    length = len(os.listdir(out_path))
    time_stamp = time.time()
    date_time = datetime.fromtimestamp(time_stamp)
    str_date_time = date_time.strftime("%Y-%m-%d-%H-%M")
    img_name = "%.4d_%s_%s.jpg"%(length, input_img_name, str_date_time)
    img_path = os.path.join(out_path, img_name)
    return img_path




def main(model_id, image_path, step_list, prompt_name_backward, forward_prompt, backward_prompts, out_img_save_path, content_name_pre, cross_attention_injection_ratio, self_attention_injection_ratio):
    
    #out_path = os.path.join(model_id, "vsiual_imgs")
    #if not os.path.exists(out_path):
    #    os.mkdir(out_path)
    #out_img_path = get_out_img_path(out_path, image_path)
    
    input_img_list = []
    output_img_list = []
    third_img_list = []
    
    pipe = ddim_inversion.DDIMStableDiffusionPipeline.from_pretrained("/home/liuzuhao/stable-diffusion-v1-5")
    pipe.scheduler = DDIMScheduler.from_config("/home/liuzuhao/stable-diffusion-v1-5/scheduler")
    pipe = pipe.to("cuda")
    
    for train_step in step_list:
        
        
        backward_embeds_dict_name = "_%s__%s.bin"%(prompt_name_backward, str(train_step))
        backward_placeholder_token = "<%s>"%prompt_name_backward
        backward_embeds_dict_path = os.path.join(model_id, backward_embeds_dict_name)
        backward_learned_embeds_dict = torch.load(backward_embeds_dict_path)
        
        text_encoder = CLIPTextModel.from_pretrained("/home/liuzuhao/stable-diffusion-v1-5", subfolder="text_encoder", revision=False)
        tokenizer = MultiTokenCLIPTokenizer.from_pretrained("/home/liuzuhao/stable-diffusion-v1-5", subfolder="tokenizer")
        
        load_multitoken_tokenizer(tokenizer, text_encoder, backward_learned_embeds_dict, backward_placeholder_token)
        
        
        
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        ldm_stable = StableDiffusionPipeline.from_pretrained("/home/liuzuhao/stable-diffusion-v1-5", scheduler=scheduler, tokenizer=tokenizer, text_encoder=text_encoder).to("cuda")
        
        ldm_stable.safety_checker = lambda images, clip_input: (images, False)
        inversion = Inversion(ldm_stable)
        
        inversion.init_prompt(forward_prompt)
        ptp_utils.register_attention_control(inversion.model, None)
        image_gt = load_512(image_path)


        text_embeddings = inversion.get_text_embedding(forward_prompt)
        image_latent = inversion.image2latent(image_gt)

        reversed_latent_list =inversion.forward_diffusion(image_latent,
                                    text_embeddings=text_embeddings,
                                    num_inference_steps=50,
                                    return_all=True
                                    )
        
        
        
        uncond_embeddings_list,latent_cur=inversion.null_optimization_path(
            reversed_latent_list,
            text_embeddings,
            num_inner_steps=50,
            epsilon=1e-5, # 0.001
            guidance_scale=7.5,
            num_ddim_steps=50)
        
        
        cross_replace_steps = {'default_': cross_attention_injection_ratio, }
        blend_word = None
        eq_params = {"words": (backward_placeholder_token,  ), "values": (0.5,)}  # amplify attention to the word "watercolor" by 5
    
    


        controller = make_controller(backward_prompts, tokenizer, False, cross_replace_steps, self_attention_injection_ratio, blend_word, eq_params)


        latents= inversion.backward_diffusion_ptp(
            backward_prompts,
            controller=controller,
            latent=reversed_latent_list[-1],
            num_inference_steps=50, guidance_scale=7.5,
            uncond_embeddings=uncond_embeddings_list
        )
        
        text_embeddings = torch.cat([inversion.get_text_embedding(""), inversion.get_text_embedding(backward_prompts[1])], dim=0)
        
        reconstructed_latents = pipe.backward_diffusion(
            latents=reversed_latent_list[-1],
            text_embeddings=text_embeddings,
            guidance_scale=3,
            num_inference_steps=50,
        )
        
        
        with torch.no_grad():
            images = ptp_utils.latent2image(ldm_stable.vae, latents.detach())
        input_img_list.append(Image.fromarray(images[0,:,:,:]))
        output_img_list.append(Image.fromarray(images[1,:,:,:]))
        third_img_list.append(pipe.latents_to_imgs(reconstructed_latents)[0])
        
    output_img_list[0].save(out_img_save_path)
    



def has_valid_extension(filename, valid_extensions):
    """
    Check if a file has a valid extension.
    Args:
        filename (str): The filename to check.
        valid_extensions (list): A list of valid extensions.

    """
    ext = os.path.splitext(filename)[1][1:]
    return ext.lower() in valid_extensions


def get_image_file(path):
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    img_files = [f for f in os.listdir(path) if f.lower().endswith(img_extensions)]
    return img_files[0]


def inversion(model_id, content_img_path, out_path_base, train_img_path, train_step, cross_attention_injection_ratio, self_attention_injection_ratio):
    """
    Implement multi-concept inversion.
    Args:
        train_step (int): The input training step.
    """
    extensions = ["jpg", "png"]
    concept_name = get_image_file(train_img_path)
    concept_name_prefix = concept_name.split('.')[0]
    for file in os.listdir(content_img_path):
        if has_valid_extension(file, extensions):
            input_img_path = os.path.join(content_img_path, file)
            content_name_prefix = file.split('.')[0]
            out_name = content_name_prefix + "_" + concept_name_prefix + ".png"
            out_img_save_path = os.path.join(out_path_base, out_name)
            prompt_name_backward = "s2"
            forward_prompt = ""
            backward_prompts = ["", "<s2>"]
            step_list = [train_step]
            main(model_id, input_img_path, step_list, prompt_name_backward, forward_prompt, backward_prompts, out_img_save_path, content_name_prefix, cross_attention_injection_ratio, self_attention_injection_ratio)
    return 0
            