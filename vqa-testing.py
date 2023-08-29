from transformers import AutoProcessor, BlipForQuestionAnswering
import torch
import openai
from PIL import Image
import pandas as pd

openai.api_key = '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'


blip_processor_base = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model_base = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model_base.to(device)


def generate_answer_blip(processor, model, image, question):
    # prepare image + question
    inputs = processor(images=image, text=question, return_tensors="pt")
    
    generated_ids = model.generate(**inputs, max_length=100)
    generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)

    generated_ids = model.generate(**inputs, max_length=100)
    generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_answer

def openai_create(prompt):

   response = openai.Completion.create(
   model="text-davinci-003",
   prompt=prompt,
   temperature=0.6,
   max_tokens=256,
   top_p=1,
   frequency_penalty=0,
   presence_penalty=0.6,
)
   return response.choices[0].text

material_dict = {}
with open("new_material_list.txt", "r") as f1:
    results = f1.readlines()
    for line in results: 
        disposal = line.split("\'")[1].strip()
        material = line.split(" [")[0].strip()

        material_dict[material] = disposal
    f1.close()

data = {'expected' : [], 'VQA image description' : [], 'OpenAI output' : [], 'VQA output' : [], 'url' : []}

with open("other.txt", "r") as f2: 
    lines = f2.readlines()
    other_list = []
    for line in lines:
        splits = line.split("https://")
        material = splits[0].strip()
        urls = splits[1:]
        url_list = []

        for part in urls:
            other_list.append(material)
            url = "https://" + part.strip()
            url_list.append(url)
            
            name = material.lower().replace(" ", "-").replace("(", " ").replace(")", " ") + ".jpg" # might need to add the (1) and (2) etc. things later
            torch.hub.download_url_to_file(url, name)
            image = Image.open("{}.jpg".format(name))
            
            city = 'Boston, MA'
            # Long processing time
            material_identification = generate_answer_blip(blip_processor_base, blip_model_base, image, 'What is this object made out of?')
            object_identification = generate_answer_blip(blip_processor_base, blip_model_base, image, 'What is this object?')
            stain_identification = generate_answer_blip(blip_processor_base, blip_model_base, image, 'Is this object wet, oily, sharp, fragmented, whole, or dry?')

            answer_vqa = generate_answer_blip(blip_processor_base, blip_model_base, image, 'Should this item be disposed in ' + city + ', would it be classified as Regular Trash, Regular Trash - Bulky Items, Regular Recycling, Private Disposal, Hazardous Waste, Food Waste, Clothing and Textiles Drop-off, Leaf and Yard Waste, Electronics Recycling, Reusable, or should it require Special Instructions?')

            if(str(material_identification[0]) == str(object_identification[0])):
                item = str(stain_identification[0]) + ' ' + str(object_identification[0])
                answer_openai = openai_create('Should ' + item + ' be disposed in ' + city + ', would it be classified as Regular Trash, Regular Trash - Bulky Items, Regular Recycling, Private Disposal, Hazardous Waste, Food Waste, Clothing and Textiles Drop-off, Leaf and Yard Waste, Electronics Recycling, Reusable, or should it require Special Instructions?')

            else:
                item = str(stain_identification[0]) + ' ' + str(material_identification[0]) + ' ' + str(object_identification[0])
                answer_openai = openai_create('Should ' + item + ' be disposed in ' + city + ', would it be classified as Regular Trash, Regular Trash - Bulky Items, Regular Recycling, Private Disposal, Hazardous Waste, Food Waste, Clothing and Textiles Drop-off, Leaf and Yard Waste, Electronics Recycling, Reusable, Collection by Appointment, or should it require Special Instructions?')
            
            # output = answer_openai.split(".")[0] # split? Perhaps not for now
            
            # data[material] = [material_dict[material], item, answer_openai.strip(), answer_vqa]

            data['expected'].append(material_dict[material])
            data['VQA image description'].append(item)
            data['OpenAI output'].append(answer_openai.strip())
            data['VQA output'].append(answer_vqa)
            data['url'].append(url)

            print(item)
            print(answer_openai.strip())
            print(answer_vqa)


df = pd.DataFrame(data, index= other_list)
print(df)
df.to_csv("vqa_other_results.csv")