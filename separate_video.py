from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os

base_name_list = ["./rawdatas/houmoto", "./rawdatas/uchida", "./rawdatas/koizumi"]
out_name_list = ["./datasets/houmoto", "./datasets/uchida", "./datasets/koizumi"]

def extract_and_split_audio(video_path, segment_length_ms, output_folder):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = video_path.replace('.mp4', '.wav')
    audio.write_audiofile(audio_path)
    video.close()
    
    sound = AudioSegment.from_file(audio_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i in range(0, len(sound), segment_length_ms):
        segment = sound[i:i + segment_length_ms]
        segment_file_path = f"{output_folder}/{os.path.basename(video_path).replace('.mp4', '')}_{i//1000}-{(i+segment_length_ms)//1000}.wav"
        segment.export(segment_file_path, format="wav")

def process_multiple_videos(folder_path, segment_length_ms, output_folder):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mp4'):
            video_path = os.path.join(folder_path, file_name)
            extract_and_split_audio(video_path, segment_length_ms, output_folder)

if __name__ == "__main__":
    for base_path, out_path in zip(base_name_list, out_name_list):
        process_multiple_videos(base_path, 10000, out_path)
