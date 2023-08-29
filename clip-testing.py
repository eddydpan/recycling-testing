import pandas as pd
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

results = {}
data = {'expected' : [], 'output' : [], 'output disposal' : [], 'probability' : [], 'url' : []}

material_list = []
material_dict = {}
with open("new_material_list.txt", "r") as f1:
    lines = f1.readlines()
    for line in lines: 
        material = line.split(" [")[0]
        material_list.append(material.strip())  # Trim any leading/trailing whitespace

        disposal = line.split("\'")[1]
        #disposal_list.append(disposal)
        material_dict[material] = disposal
    f1.close()

text = tokenizer(material_list) # sets the parameters to material_list


#####################################################################################################################

with open("other.txt", "r") as f2: 
    lines = f2.readlines()
    other_list = []
    for line in lines:
        splits = line.split("https://")
        material = splits[0].strip()
        urls = splits[1:]
        url_list = []

#####################################################################################################################

        for part in urls:
            other_list.append(material)
            url = "https://" + part.strip()
            url_list.append(url)
            name = material.lower().replace(" ", "-").replace("(", " ").replace(")", " ") + ".jpg" # might need to add the (1) and (2) etc. things later
            torch.hub.download_url_to_file(url, name)
            image = preprocess(Image.open(name)).unsqueeze(0)

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                probabilities = []
                counter = 0
                for row in text_probs:
                    for column in row:
                        results[float(column)] = [material_list[counter], material_dict[material_list[counter]]] #disposal_list[counter]
                        probabilities.append(float(column))
                        counter += 1

                sorted_probabilities = sorted(probabilities, reverse=True)
                max_probability = sorted_probabilities[0]

                # data[material] = [material_dict[material], results[max_probability][0], results[max_probability][1], max_probability, url]
                print(material_dict[material], " -:- ", results[max_probability][0], " -:- ", results[max_probability][1], " -:- ", max_probability, " -:- ", url)
                data['expected'].append(material_dict[material])
                data['output'].append(results[max_probability][0])
                data['output disposal'].append(results[max_probability][1])
                data['probability'].append(max_probability)
                data['url'].append(url)

                # print(data[material])
    f2.close()
df = pd.DataFrame(data, index= other_list)

print(df)
df.to_csv('clip_other_results.csv')