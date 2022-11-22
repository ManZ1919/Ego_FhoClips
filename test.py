import os
import av

# if os.path.exists('video_path' + '.mp4'):
#     with av.open(video_path) as container:
#         for frame in _get_frames(
#                 frames_list,
#                 container,
#                 include_audio=False,
#                 audio_buffer_frames=0
#         ):
#             frame = frame.to_rgb().to_ndarray()
#             frames.append(frame)
#         print("real_frames:", frames)
#     return frames
# else:
#     print(False)
#     pass

# if os.path.exists('/media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/ego4d/new/v1/full_scale/542bd8e2-37a1-49be-8f4c-0fedeb1cc9f1'+'.mp4'):
#     print(True)
# else:
#     print(False)

# with av.open('/media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/ego4d/new/v1/full_scale/542bd8e2-37a1-49be-8f4c-0fedeb1cc9f1'+'.mp4') as container:
#     for frame in _get_frames(
#             frames_list,
#             container,
#             include_audio=False,
#             audio_buffer_frames=0
#     ):
#         frame = frame.to_rgb().to_ndarray()
#         frames.append(frame)
#     print("real_frames:", frames)
# return frames

# python -m train --cfg /home/dml/PycharmProjects/Ego4D/Ego4dBenchmark/hands-and-objects/state-change-localization-classification/i3d-resnet50/configs/2022-10-27_keyframe_loc_release1.yaml DATA.VIDEO_DIR_PATH /media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/ego4d/new/v1/full_scale DATA.CLIPS_SAVE_PATH /media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/ego4d/new/v1/positive_clips DATA.NO_SC_PATH /media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/ego4d/new/v1/negative_clips DATA.DATA.ANN_DIR /media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/ego4d/new/v1/annotations
# torch.multiprocessing.set_start_method('spawn')
# RuntimeError: Could not infer dtype of NoneType
import json
import av
from tqdm import tqdm

path = "/media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/dataset/ego4d/v1"
anno_path = path + "/annotations/fho_oscc-pnr_val.json"
video_path = path + "/full_scale"
clips = json.load(open(anno_path, "r"))["clips"]
for clip in tqdm(clips):
    video = video_path + "/"+ str(clip["video_uid"]) + ".mp4"
    container = av.open(video)


# python -m train --cfg configs/2022-10-27_keyframe_loc_release1.yaml DATA.VIDEO_DIR_PATH /media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/dataset/ego4d/v1/full_scale DATA.CLIPS_SAVE_PATH /media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/dataset/ego4d/v1/positive_clips DATA.NO_SC_PATH /media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/dataset/ego4d/v1/negative_clips DATA.ANN_DIR /media/dml/e5afa40a-df1a-4c60-8623-87e2a51c3a09/dataset/ego4d/v1/annotations


