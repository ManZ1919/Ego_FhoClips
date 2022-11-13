import json
import av
from tqdm import tqdm

path = "/media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/ego4d/new/v1"
anno_path = path + "/annotations/fho_oscc-pnr_val.json"
video_path = path + "/full_scale"
clips = json.load(open(anno_path, "r"))["clips"]
for clip in tqdm(clips):
    video = video_path + "/"+ str(clip["video_uid"]) + ".mp4"
    container = av.open(video)