import os, sys
import requests
import zipfile
import shutil
from argparse import ArgumentParser


# v2 (with improved sentence tokenization pipeline) created silver dataset google drive link for each language
# languages_map = {
#     'as': ['Assamese', '1rcXXecUDIIZqU70Ie29xfwdVXQuZFpnb'],
#     'bn': ['Bengali', '1rvErywG-H3Xxt5aSnbqxkfWkJhjSmEId'],
#     'en': ['English', '10R9tmaRYFt189b1a43y3dVYAtebd4ede'],
#     'gu': ['Gujarati', '1XGBBa83p7uov3CgMGJR7NO8I8SQRdzTO'],
#     'hi': ['Hindi', '1znNS7GXgdqOo36nuz1dU5o2kbEMrajYa'],
#     'kn': ['Kannada', '1Ra0EsCUmXbuo_5hPU_HO08rrehflRJrf'],
#     'ml': ['Malayalam', '1ygTZO1FbEUYKDrISCdQFVeiczzyHLU_k'],
#     'mr': ['Marathi', '1QiD8I1jR1WhS_FPMdJGrx6-i7UoJZbhX'],
#     'or': ['Odia', '1HhZslTjZuV_NT8M559dhwexP1Uql28EP'], 
#     'pa': ['Punjabi', '1R3PtADe3fgV6bsCl6RzkpFsZ_IBHdNHD'],
#     'ta': ['Tamil', '1_33CiJJ81RGEF7BeSv_FShjw6dW9zq7a'],
#     'te': ['Telugu', '1waCaRresYVkvATTJr-4KkZR-KZLfQDjf'],
# }

# v2.1 with qualifier (with improved sentence tokenization pipeline) created silver dataset google drive link for each language
languages_map = {
    'as': ['Assamese', '1GKZpodnOWfieuARkUUYeOTwd9-zkG1Zz'],
    'bn': ['Bengali', '1Yn62mB2rw8t0KUdDfabUzkNba1kbYSeL'],
    'en': ['English', '1PFjaOk63LcOIlXUS7jMHeZOIaJTpR5_P'],
    'gu': ['Gujarati', '15S0kWBVEU61VyMnaZlUXBvOtwr1SB6Jc'],
    'hi': ['Hindi', '1LtYSeOeCQ09Ng2jE-42BMnijJ4-FUCXN'],
    'kn': ['Kannada', '17AfI-CcDBLe0T9m5rSSjwaubpFUZOeml'],
    'ml': ['Malayalam', '1DBRU-_x9bgFRj6FISZCbTy3jg822YcIG'],
    'mr': ['Marathi', '1Y3EvuzcREH5vuwh0QZCjOQaa1terc7aJ'],
    'or': ['Odia', '1C-SURcLd39FnQCn2Kc780p76b0Rz8E4U'],
    'pa': ['Punjabi', '1EJZ3MxZUOTAYAND6vGtiMfafyaVPdQXf'],
    'ta': ['Tamil', '1HxDpo8Me5keu7-zh1QbdaqxD-AHh2gyA'],
    'te': ['Telugu', '1V2UMvhqO4Crsa8qx-rhfr3A01BQqiN2q'],
}

# # v3 (using the coverage threshold value of 0.65)
# languages_map = {
#     'as': ['Assamese', '1o8I4BX80LnaS_O7OPEJ-RQC-tP1m27Bt'],
#     'bn': ['Bengali', '12XMhNyLANfNnpSMJ4d46j55B_P8L1sYo'],
#     'en': ['English', '1Tz8qEFCYubl9fWCxGVEt7UETsARXgT1S'],
#     'gu': ['Gujarati', '1X4mNS7xYXYNRZOyATHD1wlyqpPE9_uiR'],
#     'hi': ['Hindi', '10U8o1sUvEYgXInP5in0iWFQot0TzNB_g'],
#     'kn': ['Kannada', '1jJc3yROg0ZVXliEoZEqP434yYaTpYTMI'],
#     'ml': ['Malayalam', '1957HdKffEyFVUa2SGoqVY7Gk82sN2KAn'],
#     'mr': ['Marathi', '1lwrwyTljKNXIQBPHg4-UKiTpKGyqklx-'],
#     'or': ['Odia', '1lsZ4_ReixeAqBXYUdeJyHpKZbcl9B73K'], 
#     'pa': ['Punjabi', '19CbK340Cjl49nnK74t_xFv7lYUS9pvQm'],
#     'ta': ['Tamil', '1NJ4vHL3-xA6ZHzfm3MrNq4OTNfOEEiw3'],
#     'te': ['Telugu', '1I6K9eevJIyoBoenxIofadxX2VhdDFaG4'],
# }

def handle_multiple_languages(lang):
    # check if multiple "," separated entries exists.
    lang_list = [x.strip().lower() for x in lang.split(',')]
    valid_languages = []
    for x in lang_list:
        if x not in languages_map:
            continue
        valid_languages.append(x)
    if len(valid_languages)!=len(lang_list):
        print("%d invalid languages identified" % (len(lang_list) - len(valid_languages)))
    print('successfully identified %d langauges: %s' % (len(valid_languages), valid_languages))
    valid_languages = sorted(valid_languages)
    return valid_languages

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def dataset_exists(dir_path):
    required_files = set(['train.jsonl', 'test.jsonl', 'val.jsonl'])
    existing_files = set()
    for dfile in os.listdir(os.path.abspath(dir_path)):
        if dfile in required_files:
            existing_files.add(dfile)
    missing_files = required_files.difference(existing_files)
    if len(missing_files):
        print("%s files are missing, downloading the files.." % missing_files)
    return len(missing_files)==0



if __name__ == "__main__":
    parser = ArgumentParser()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    default_dataset_path = os.path.join(base_dir, 'datasets')
    
    parser.add_argument("--store_path", type=str,
                        help="directory path to store processed data", default=default_dataset_path)
    parser.add_argument("--lang", type=str,
                        help="specify the language for data-to-text generation", default='hi')
    args  = parser.parse_args()

    valid_languages = handle_multiple_languages(args.lang)
    
    if len(valid_languages)==0:
        print('Invalid language(s) specified !!!')
        sys.exit(0)

    for seq_no, lang in enumerate(valid_languages):
        print('%d / %d working on language: %s (%s)' % (seq_no+1, len(valid_languages), languages_map[lang][0], lang))
        complete_path = os.path.join(args.store_path, lang)

        if os.path.exists(complete_path) and dataset_exists(complete_path):
            #remove the directory if exists
            print("All dataset files are present.")
            print('--'*30)
            continue

        print("creating processed data directory")
        os.makedirs(args.store_path, exist_ok=True)

        print("Downloading Processed Dataset...")
        file_store_path = os.path.join(args.store_path, "%s_silver_data.zip" % lang)

        download_file_from_google_drive(languages_map[lang][1], file_store_path)

        with zipfile.ZipFile(file_store_path, 'r') as zfile:
            zfile.extractall(args.store_path)
        #finally delete the zip file
        os.remove(file_store_path)
        print("Downloaded successfully.")
        print('--'*30)