import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


IMAGE_PATH = "https://upload.wikimedia.org/wikipedia/commons/c/c8/New_York_City_roofs_%284891620775%29.jpg"
PROMPT = "Write a long descriptive caption for this image in a formal tone"
MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
llava_model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype="float8", device_map=0 )
llava_model.eval()

image = Image.open(IMAGE_PATH)
convo = [
    {
        "role": "system", 
        "content": "You are a helpful image captioner."
    }, 
    {
        "role": "user", 
        "content": PROMPT
    }
]


convo_string = processor.apply_chat_template(convo, tokenize=False, return_tensor="pt")
assert isinstance(convo_string, str)
inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)




with torch.no_grad():

    generate_ids = llava_model.generate(
        **inputs, 
        max_new_tokens = 512, 
        do_sample = True, 
        suppress_tokens = None, 
        use_cache = True, 
        temperature = 0.6, 
        top_k = None, 
        top_p = 0.9

    )[0]



generate_ids = generate_ids[inputs["input_ids"].shape[1]:]


caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_space=False)
caption = caption.strip()
print(caption)