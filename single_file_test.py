import numpy as  np
import torch
import os, sys
from src import SAMPLE_RATE, WINDOW_LENGTH, E2E, HOP_LENGTH, to_local_average_cents
from src.rvc_modified import *
# from src.rvc_modified import infer_from_audio
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import torchaudio

def inference(file_path, model_path, device, save_path=None):
    if save_path is None:
        basename = os.path.basename(file_path)
        save_path = os.path.join(os.path.dirname(file_path), os.path.splitext(basename)[0] + '_rmvpe.txt')

    print(f'Inference the model with {file_path} and save the pitch to {save_path}')

    print(f'Load the model from {model_path}')
    # model = E2E(4, 1, (2, 2))
    # print(model)
    # model.load_state_dict(torch.load(model_path,
    #                               map_location=torch.device('cpu')))
    # model = model.to(device).eval()
    rmvpe = RMVPE(model_path, is_half=False, device="cpu")
    # rmvpe.model.to(device).eval()
    audio, sample_rate = torchaudio.load(file_path)

    # resample to 16000Hz
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
    audio = audio[0].to(device)
    # Chunk audio -> the input to the model should be 128 frames
    # pitch_pred = torch.zeros((audio.shape[0] // 128 + 1, 128, 360)).to(device)

    # Inference the model with progress bar
    # slices = range(0, pitch_pred.shape[0], 128)

    print(f'start inference the model with {file_path}')
    # print(type(model))
    # for start_steps in tqdm(slices):
    #
    #     # Get the start and end steps
    #     end_steps = start_steps + 127
    #     start = int(start_steps * HOP_LENGTH * SAMPLE_RATE / 1000)
    #     end = int(end_steps * HOP_LENGTH * SAMPLE_RATE / 1000) + WINDOW_LENGTH

        # Pad the audio if the end_steps is greater than the audio length
        # if end_steps >= audio.shape[0]:
        #     t_audio = F.pad(audio[start:end], (0, end - start - len(audio[start:end])), mode='constant')
        #     input = t_audio.reshape((-1, t_audio.shape[-1]))
        #     print(input.shape)
        #     # generate batch dimension
        #     input = input.unsqueeze(0)
        #     t_pitch_pred = model(input).squeeze(0)
        #     pitch_pred[start_steps:end_steps + 1] = t_pitch_pred[:pitch_pred.shape[0] - end_steps - 1]
        #
        # # If the end_steps is less than the audio length
        # else:
        #     t_audio = audio[start:end]
        #     input = t_audio.reshape((-1, t_audio.shape[-1]))
        #     print(input.shape)
        #     # generate batch dimension
        #     input = input.unsqueeze(0)
        #     t_pitch_pred = model(input).squeeze(0)
        #     pitch_pred[start_steps:end_steps + 1] = t_pitch_pred
    f0 = rmvpe.infer_from_audio(audio, 0.03)

    # write the pitch text file as the format of rows of (times, freqs)
    print(f0, f0.shape, np.max(f0), np.min(f0))
    # cents_pred = rmvpe.to_local_average_cents(f0, thred=0.03)
    # freq_pred = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
    time_slice = np.array([i*10/1000 for i in range(len(f0))])

    print(f'Save the pitch to {save_path}')
    with open(save_path, 'w') as f:
        for i in range(len(time_slice)):
            f.write(str(time_slice[i]) + '\t' + str(f0[i]) + '\n')

    print(f'all done!')
if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Inference the model')
    parser.add_argument('--file_path', type=str, help='Path to the audio file')
    parser.add_argument('--save_path', type=str, help='Path to save the pitch file')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--device', type=str, help='Device to run the model')
    args = parser.parse_args()

    # Run the inference
    inference(args.file_path, args.model_path, args.device, args.save_path)