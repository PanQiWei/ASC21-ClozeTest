import warnings
warnings.filterwarnings(action='ignore')
from config import *

import os
import json
import nltk
import fnmatch

from multiprocessing import Process, cpu_count
try:
    from nltk.tokenize import sent_tokenize
except:
    nltk.download('all')
finally:
    from nltk.tokenize import sent_tokenize

def get_file_list(data_dir,expand=".json"):
    files = []
    for root, _, file_names in os.walk(data_dir):
        for filename in fnmatch.filter(file_names, "*"+expand):
            files.append(os.path.join(root, filename))
    return files

def gen_eda_data(dir, num_threads=cpu_count()-1, tool_path=ROOT+"/tools/eda_tool/code/augment.py", num_aug=1, alpha=0.05):
    all_files = get_file_list(dir)
    each_thread_files_num = int(len(all_files)/4)+1
    thread_list = []
    for i in range(num_threads):
        start = i*each_thread_files_num
        end = min((i+1)*each_thread_files_num, len(all_files))
        files = all_files[start:end]
        thread_list.append(
            Process(target=augment_articles, args=(files, tool_path, num_aug, alpha))
        )
    for threaad in thread_list:
        threaad.start()
    for threaad in thread_list:
        threaad.join()

def augment_articles(files, tool_path, num_aug, alpha):
    for file in files:
        try:
            augment_article(file, tool_path, num_aug, alpha)
        except Exception:
            continue

def augment_article(file, tool_path, num_aug, alpha):
    # get file name
    fname = os.path.split(file)[-1].split(".")[0]
    # load json file
    data = json.loads(open(file,'r').read())
    article = data['article']
    # split article into sentences
    sentences = sent_tokenize(article)
    for i in range(len(sentences)):
        label = fname+"-"+str(i)
        sentences[i] = sentences[i].replace("\n"," ")
        sentences[i] = label+"\t"+sentences[i]
    # save label-sentence pair into temp file
    augmented_file_dir = ROOT+"/data/raw/train_eda/"
    if not os.path.exists(augmented_file_dir):
        os.mkdir(augmented_file_dir)
    input_file = augmented_file_dir+fname+".txt"
    with open(input_file, "w") as f:
        for sentence in sentences:
            f.write(sentence+"\n")
    # augment sentences
    order = "python "+tool_path+" --input={} --num_aug={} --alpha={}".format(input_file, num_aug, alpha)
    os.system(order)
    # merge augmented sentences into articles
    label_sentence_pair = {}
    output_file = augmented_file_dir+"eda_"+fname+".txt"
    with open(output_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            label, sentence = [each for each in line.split("\t")]
            if not label in label_sentence_pair.keys():
                label_sentence_pair[label] = [sentence.strip()]
            else:
                label_sentence_pair[label].append(sentence.strip())
    label_list = [label for label in label_sentence_pair.keys()]
    aug_sentences_list = [label_sentence_pair[label] for label in label_list]

    article_list = []
    for i in range(num_aug+1):
        article = ""
        for each in aug_sentences_list:
            article += " "+each[i]
        article_list.append(article.strip())
    # merge article and options, ans and save
    for i in range(len(article_list)):
        json_data = {}
        json_data["article"] = article_list[i]
        json_data["options"] = data["options"]
        json_data["answers"] = data["answers"]
        with open(augmented_file_dir+fname+"_eda"+str(i)+".json","w") as f:
            json.dump(json_data, f)
    # delete all temp file
    temp_files = [input_file, output_file]
    for file in temp_files:
        os.remove(file)

if __name__ == "__main__":
    gen_eda_data(ROOT+"/data/raw/train")