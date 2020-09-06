import string
import csv
import os 
 
path = os.path.dirname(os.getcwd())

# extract descriptions for images
def load_descriptions(filename):
    mapping = dict()

    with open(path+"/dataset/"+filename, 'r', encoding="utf8") as file:
        reader = csv.reader(file)
        for line in reader:
            if len(line) < 1:
                continue
            image_id, image_desc = line[0], line[1]
            image_id = image_id.split('.')[0]
            if image_id not in mapping:
                mapping[image_id] = list()
            mapping[image_id].append(image_desc)

    return mapping
 
def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)


# Creating set of videonames
def find_videoName():
    files = os.listdir(path+"/dataset/YouTubeClips")
    video_name = set()
    for f in files:
        video_name.add(f.split('.')[0])
    return video_name

# save descriptions to file, one per line
def save_descriptions(descriptions, filename, video_name):
    lines = list()
    for key, desc_list in descriptions.items():
        if key in video_name:
            for desc in desc_list:
                lines.append(key+' '+desc)
    data = '\n'.join(lines)
    file = open(path+"/dataset/"+filename, 'w', encoding="utf8")
    file.write(data)
    file.close()

filename = 'MSVD_description_cfile.csv'
descriptions = load_descriptions(filename)
# clean descriptions
clean_descriptions(descriptions)
# Storing the name of videos
video_name = find_videoName()
# save to file
save_descriptions(descriptions, 'cleaned_data.txt', video_name)