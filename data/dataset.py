import json
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import torch
import os
import pandas as pd

class SCF(Dataset):
    def __init__(self, manifest_files, sample_duration=0.63, sample_rate=16000, n_fft=400, n_mels=64, win_length=400, hop_length=160, augment=False):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.augment = augment
        self.sample_duration = sample_duration

        # Load JSON files
        self.data = []
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as f:
                for line in f:
                    try:
                        manifest_entry = json.loads(line.strip())
                        audio_path = manifest_entry['audio_filepath']
                        duration = manifest_entry['duration']
                        offset = manifest_entry['offset']
                        label = 1 if manifest_entry['label'] == 'speech' else 0
                        self.data.append((audio_path, label, duration, offset))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            win_length=self.win_length,
            hop_length=self.hop_length
        )
        self.log_mel_spectrogram = T.AmplitudeToDB()

        self.specaugment = T.FrequencyMasking(freq_mask_param=15)
        self.timemask = T.TimeMasking(time_mask_param=25)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label, duration, offset = self.data[idx]

        if self.sample_duration == 0.63:
            # Load audio file with offset and duration
            waveform, _ = torchaudio.load(audio_path, frame_offset=int(offset * self.sample_rate), num_frames=int(duration * self.sample_rate))

            # Apply augmentation if specified
            if self.augment:
                waveform = self.augment_waveform(waveform)

            # Convert to log mel spectrogram
            mel_spec = self.mel_spectrogram(waveform)
            log_mel_spec = self.log_mel_spectrogram(mel_spec)

            if self.augment:
                log_mel_spec = self.augment_spectrogram(log_mel_spec)
            
            return log_mel_spec, label
        
        elif self.sample_duration == 0.32:
            # Split the duration into two halves
            half_duration = duration / 2

            waveform1, _ = torchaudio.load(audio_path, frame_offset=int(offset * self.sample_rate), num_frames=int(half_duration * self.sample_rate))                
            waveform2, _ = torchaudio.load(audio_path, frame_offset=int((offset + half_duration) * self.sample_rate), num_frames=int(half_duration * self.sample_rate))

            if self.augment:
                waveform1 = self.augment_waveform(waveform1)
                waveform2 = self.augment_waveform(waveform2)

            mel_spec1 = self.mel_spectrogram(waveform1)
            log_mel_spec1 = self.log_mel_spectrogram(mel_spec1)
            mel_spec2 = self.mel_spectrogram(waveform2)
            log_mel_spec2 = self.log_mel_spectrogram(mel_spec2)

            if self.augment:
                log_mel_spec1 = self.augment_spectrogram(log_mel_spec1)
                log_mel_spec2 = self.augment_spectrogram(log_mel_spec2)

            # Return both halves with the same label
            return (log_mel_spec1, label), (log_mel_spec2, label)
        
        elif self.sample_duration == 0.16:
            # Split the duration into four quarters
            quarter_duration = duration / 4

            waveform1, _ = torchaudio.load(audio_path, frame_offset=int(offset * self.sample_rate), num_frames=int(quarter_duration * self.sample_rate))
            waveform2, _ = torchaudio.load(audio_path, frame_offset=int((offset + quarter_duration) * self.sample_rate), num_frames=int(quarter_duration * self.sample_rate))
            waveform3, _ = torchaudio.load(audio_path, frame_offset=int((offset + 2 * quarter_duration) * self.sample_rate), num_frames=int(quarter_duration * self.sample_rate))
            waveform4, _ = torchaudio.load(audio_path, frame_offset=int((offset + 3 * quarter_duration) * self.sample_rate), num_frames=int(quarter_duration * self.sample_rate))

            if self.augment:
                waveform1 = self.augment_waveform(waveform1)
                waveform2 = self.augment_waveform(waveform2)
                waveform3 = self.augment_waveform(waveform3)
                waveform4 = self.augment_waveform(waveform4)
            
            mel_spec1 = self.mel_spectrogram(waveform1)
            log_mel_spec1 = self.log_mel_spectrogram(mel_spec1)
            mel_spec2 = self.mel_spectrogram(waveform2)
            log_mel_spec2 = self.log_mel_spectrogram(mel_spec2)
            mel_spec3 = self.mel_spectrogram(waveform3)
            log_mel_spec3 = self.log_mel_spectrogram(mel_spec3)
            mel_spec4 = self.mel_spectrogram(waveform4)
            log_mel_spec4 = self.log_mel_spectrogram(mel_spec4)

            if self.augment:
                log_mel_spec1 = self.augment_spectrogram(log_mel_spec1)
                log_mel_spec2 = self.augment_spectrogram(log_mel_spec2)
                log_mel_spec3 = self.augment_spectrogram(log_mel_spec3)
                log_mel_spec4 = self.augment_spectrogram(log_mel_spec4)

            # Return both halves with the same label
            return (log_mel_spec1, label), (log_mel_spec2, label), (log_mel_spec3, label), (log_mel_spec4, label)
        
        elif self.sample_duration == 0.025:
            num_frames = int(self.sample_duration * self.sample_rate)
            num_segments = int(duration / self.sample_duration)

            segments = []
            for i in range(num_segments):
                waveform, _ = torchaudio.load(audio_path, frame_offset=int((offset + i * self.sample_duration) * self.sample_rate), num_frames=num_frames)

                if self.augment:
                    waveform = self.augment_waveform(waveform)

                mel_spec = self.mel_spectrogram(waveform)
                log_mel_spec = self.log_mel_spectrogram(mel_spec)

                if self.augment:
                    log_mel_spec = self.augment_spectrogram(log_mel_spec)

                segments.append((log_mel_spec, label))

            return tuple(segments)

    def augment_waveform(self, waveform):
        """
        Augment the waveform with time shift and white noise.
        """
        # Time shift perturbation with a probability of 80%
        if random.random() < 0.8:
            shift = random.randint(-5, 5)  # Time shift between -5ms to 5ms
            waveform = torch.roll(waveform, shifts=shift, dims=-1)
        
        # Add white noise ranging from 90dB to -46dB
        noise_level = random.uniform(-46, 90)
        noise = torch.randn(waveform.size()) * (10 ** (noise_level / 20.0))
        waveform = waveform + noise
        
        return waveform

    def augment_spectrogram(self, log_mel_spec):
        """
        Apply SpecAugment and SpecCutout to the log mel spectrogram.
        """
        # Apply SpecAugment with time and frequency masks
        log_mel_spec = self.timemask(log_mel_spec)
        log_mel_spec = self.specaugment(log_mel_spec)
        
        # Dynamically adjust cutout size based on the spectrogram's actual size
        cutout_width = min(25, log_mel_spec.size(-1))  # Ensure cutout doesn't exceed time dimension
        cutout_height = min(15, log_mel_spec.size(-2))  # Ensure cutout doesn't exceed frequency dimension
        
        # Apply SpecCutout (masking rectangular areas)
        cutout = torch.zeros((log_mel_spec.size(0), cutout_height, cutout_width))  # Adjust the cutout to have 3 dimensions
        
        # Randomly select cutout position, ensuring the cutout fits within the spectrogram
        cutout_x = random.randint(0, log_mel_spec.size(-1) - cutout_width)
        cutout_y = random.randint(0, log_mel_spec.size(-2) - cutout_height)
        
        # Replace the cutout section with zeros
        log_mel_spec[:, cutout_y:cutout_y + cutout_height, cutout_x:cutout_x + cutout_width] = cutout

        return log_mel_spec

class SCF_ESC(Dataset):
    def __init__(self, manifest_files, noise_csv, noise_audio_dir, sample_duration=0.63, sample_rate=16000, n_fft=400, n_mels=64, win_length=400, hop_length=160, augment=True):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.augment = augment
        self.sample_duration = sample_duration
        self.noise_zero_power_count = 0

        # Load noise data
        self.noise_data = pd.read_csv(noise_csv)
        self.noise_audio_dir = noise_audio_dir

        # Load JSON files
        self.data = []
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as f:
                for line in f:
                    try:
                        manifest_entry = json.loads(line.strip())
                        audio_path = manifest_entry['audio_filepath']
                        duration = manifest_entry['duration']
                        offset = manifest_entry['offset']
                        label = 1 if manifest_entry['label'] == 'speech' else 0
                        self.data.append((audio_path, label, duration, offset))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            win_length=self.win_length,
            hop_length=self.hop_length
        )
        self.log_mel_spectrogram = T.AmplitudeToDB()

        self.specaugment = T.FrequencyMasking(freq_mask_param=15)
        self.timemask = T.TimeMasking(time_mask_param=25)

    def _load_noise(self):
        # Randomly select a noise file
        noise_row = self.noise_data.sample(n=1).iloc[0]
        noise_filename = noise_row['filename']
        noise_path = os.path.join(self.noise_audio_dir, noise_filename)
        noise_waveform, _ = torchaudio.load(noise_path)
        return noise_waveform

    def _random_snr(self):
        # Randomly select an SNR from [-10, -5, 0, 5, 10]
        snr_values = [-10, -5, 0, 5, 10]
        return random.choice(snr_values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label, duration, offset = self.data[idx]

        # Load the clean waveform
        waveform, _ = torchaudio.load(audio_path, frame_offset=int(offset * self.sample_rate), num_frames=int(duration * self.sample_rate))

        # Load noise waveform
        noise_waveform = self._load_noise()

        # Ensure noise and clean speech have the same length
        if noise_waveform.size(1) < waveform.size(1):
            # Repeat noise until it's long enough
            repeat_times = (waveform.size(1) // noise_waveform.size(1)) + 1
            noise_waveform = noise_waveform.repeat(1, repeat_times)[:, :waveform.size(1)]
        else:
            # Randomly crop noise to match the length of clean speech
            start = random.randint(0, noise_waveform.size(1) - waveform.size(1))
            noise_waveform = noise_waveform[:, start:start + waveform.size(1)]

        # Randomly select an SNR
        self.snr = self._random_snr()

        # Calculate SNR
        clean_power = torch.norm(waveform) ** 2
        noise_power = torch.norm(noise_waveform) ** 2
        desired_noise_power = clean_power / (10 ** (self.snr / 10))
        if noise_power > 0:
            noise_scaling_factor = (desired_noise_power / noise_power) ** 0.5
            noise_waveform = noise_waveform * noise_scaling_factor
        else:
            self.noise_zero_power_count += 1
            noise_waveform = torch.zeros_like(waveform)

        # Mix clean speech and noise
        mixed_waveform = waveform + noise_waveform

        # Apply augmentations if needed
        if self.augment:
            mixed_waveform = self.augment_waveform(mixed_waveform)

        if self.sample_duration == 0.63:
            # Convert to log mel spectrogram
            mel_spec = self.mel_spectrogram(mixed_waveform)
            log_mel_spec = self.log_mel_spectrogram(mel_spec)

            if self.augment:
                log_mel_spec = self.augment_spectrogram(log_mel_spec)

            return log_mel_spec, label

        elif self.sample_duration == 0.32:
            # Split the duration into two halves
            half_duration = duration / 2

            waveform1, _ = torchaudio.load(audio_path, frame_offset=int(offset * self.sample_rate), num_frames=int(half_duration * self.sample_rate))                
            waveform2, _ = torchaudio.load(audio_path, frame_offset=int((offset + half_duration) * self.sample_rate), num_frames=int(half_duration * self.sample_rate))

            if self.augment:
                waveform1 = self.augment_waveform(waveform1)
                waveform2 = self.augment_waveform(waveform2)

            mel_spec1 = self.mel_spectrogram(waveform1)
            log_mel_spec1 = self.log_mel_spectrogram(mel_spec1)
            mel_spec2 = self.mel_spectrogram(waveform2)
            log_mel_spec2 = self.log_mel_spectrogram(mel_spec2)

            if self.augment:
                log_mel_spec1 = self.augment_spectrogram(log_mel_spec1)
                log_mel_spec2 = self.augment_spectrogram(log_mel_spec2)

            # Return both halves with the same label
            return (log_mel_spec1, label), (log_mel_spec2, label)
        
        elif self.sample_duration == 0.16:
            # Split the duration into four quarters
            quarter_duration = duration / 4

            waveform1, _ = torchaudio.load(audio_path, frame_offset=int(offset * self.sample_rate), num_frames=int(quarter_duration * self.sample_rate))
            waveform2, _ = torchaudio.load(audio_path, frame_offset=int((offset + quarter_duration) * self.sample_rate), num_frames=int(quarter_duration * self.sample_rate))
            waveform3, _ = torchaudio.load(audio_path, frame_offset=int((offset + 2 * quarter_duration) * self.sample_rate), num_frames=int(quarter_duration * self.sample_rate))
            waveform4, _ = torchaudio.load(audio_path, frame_offset=int((offset + 3 * quarter_duration) * self.sample_rate), num_frames=int(quarter_duration * self.sample_rate))

            if self.augment:
                waveform1 = self.augment_waveform(waveform1)
                waveform2 = self.augment_waveform(waveform2)
                waveform3 = self.augment_waveform(waveform3)
                waveform4 = self.augment_waveform(waveform4)
            
            mel_spec1 = self.mel_spectrogram(waveform1)
            log_mel_spec1 = self.log_mel_spectrogram(mel_spec1)
            mel_spec2 = self.mel_spectrogram(waveform2)
            log_mel_spec2 = self.log_mel_spectrogram(mel_spec2)
            mel_spec3 = self.mel_spectrogram(waveform3)
            log_mel_spec3 = self.log_mel_spectrogram(mel_spec3)
            mel_spec4 = self.mel_spectrogram(waveform4)
            log_mel_spec4 = self.log_mel_spectrogram(mel_spec4)

            if self.augment:
                log_mel_spec1 = self.augment_spectrogram(log_mel_spec1)
                log_mel_spec2 = self.augment_spectrogram(log_mel_spec2)
                log_mel_spec3 = self.augment_spectrogram(log_mel_spec3)
                log_mel_spec4 = self.augment_spectrogram(log_mel_spec4)

            # Return both halves with the same label
            return (log_mel_spec1, label), (log_mel_spec2, label), (log_mel_spec3, label), (log_mel_spec4, label)
        
        elif self.sample_duration == 0.025:
            num_frames = int(self.sample_duration * self.sample_rate)
            num_segments = int(duration / self.sample_duration)

            segments = []
            for i in range(num_segments):
                waveform, _ = torchaudio.load(audio_path, frame_offset=int((offset + i * self.sample_duration) * self.sample_rate), num_frames=num_frames)

                if self.augment:
                    waveform = self.augment_waveform(waveform)

                mel_spec = self.mel_spectrogram(waveform)
                log_mel_spec = self.log_mel_spectrogram(mel_spec)

                if self.augment:
                    log_mel_spec = self.augment_spectrogram(log_mel_spec)

                segments.append((log_mel_spec, label))

            return tuple(segments)

    def augment_waveform(self, waveform):
        # Time shift perturbation with a probability of 80%
        if random.random() < 0.8:
            shift = random.randint(-5, 5)  # Time shift between -5ms to 5ms
            waveform = torch.roll(waveform, shifts=shift, dims=-1)
        
        # Add white noise ranging from 90dB to -46dB
        noise_level = random.uniform(-46, 90)
        noise = torch.randn(waveform.size()) * (10 ** (noise_level / 20.0))
        waveform = waveform + noise
        return waveform

    def augment_spectrogram(self, log_mel_spec):
        # Apply SpecAugment and SpecCutout to log mel spectrogram
        log_mel_spec = self.timemask(log_mel_spec)
        log_mel_spec = self.specaugment(log_mel_spec)
        
        return log_mel_spec

    def get_noise_zero_power_count(self):
        return self.noise_zero_power_count

class AVA(Dataset):
    def __init__(self, root_dir, sample_duration=0.63, overlap=0.125, sample_rate=16000, n_fft=400, n_mels=64, win_length=400, hop_length=160):
        self.root_dir = root_dir
        self.audio_paths = []
        self.labels = []
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.sample_duration = sample_duration
        self.min_duration_samples = int(self.sample_duration * sample_rate)
        self.overlap = overlap
        self.step_size = int(self.min_duration_samples * (1 - self.overlap))

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            win_length=self.win_length,
            hop_length=self.hop_length
        )
        self.log_mel_spectrogram = T.AmplitudeToDB()
        self._prepare_dataset()

    def _prepare_dataset(self):
        label_mapping = {
            'NO_SPEECH': 0,
            'CLEAN_SPEECH': 1,
            'SPEECH_WITH_MUSIC': 1,
            'SPEECH_WITH_NOISE': 1
        }

        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                label = label_mapping.get(folder_name, -1)
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.wav'):
                        self.audio_paths.append(os.path.join(folder_path, file_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        waveform, _ = torchaudio.load(audio_path)

        if waveform.size(1) < self.min_duration_samples:
            pad_length = self.min_duration_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length), mode='constant', value=0)

        num_samples = waveform.size(1)
        segments = []

        start = 0
        while start + self.min_duration_samples <= num_samples:
            segment = waveform[:, start:start + self.min_duration_samples]

            if segment.size(1) < self.min_duration_samples:
                pad_length = self.min_duration_samples - segment.size(1)
                segment = torch.nn.functional.pad(segment, (0, pad_length), mode='constant', value=0)

            mel_spec = self.mel_spectrogram(segment)
            log_mel_spec = self.log_mel_spectrogram(mel_spec)

            segments.append((segment, log_mel_spec, label))

            start += self.step_size

        if start < num_samples:
            remainder = waveform[:, start:]

            if remainder.size(1) < self.min_duration_samples:
                pad_length = self.min_duration_samples - remainder.size(1)
                remainder = torch.nn.functional.pad(remainder, (0, pad_length), mode='constant', value=0)

            mel_spec = self.mel_spectrogram(remainder)
            log_mel_spec = self.log_mel_spectrogram(mel_spec)

            segments.append((remainder, log_mel_spec, label))

        return tuple(segments)

class AVA_ESC(Dataset):
    def __init__(self, ava_dir, esc_csv, esc_audio_dir, snr=10, sample_duration=0.63, overlap=0.125, sample_rate=16000, n_fft=400, n_mels=64, win_length=400, hop_length=160):
        self.ava = AVA(ava_dir, sample_duration, overlap, sample_rate, n_fft, n_mels, win_length, hop_length)
        self.esc_csv = esc_csv
        self.esc_audio_dir = esc_audio_dir
        self.snr = snr
        self.noise_zero_power_count = 0

        # Load ESC-50 noise data
        self.noise_data = pd.read_csv(self.esc_csv)

    def _load_noise(self):
        # Randomly select a noise file
        noise_row = self.noise_data.sample(n=1).iloc[0]
        noise_filename = noise_row['filename']
        noise_path = os.path.join(self.esc_audio_dir, noise_filename)
        noise_waveform, _ = torchaudio.load(noise_path)
        return noise_waveform

    def __len__(self):
        return len(self.ava)

    def __getitem__(self, idx):
        segments = self.ava[idx]

        mixed_segments = []

        for clean_waveform, log_mel_spec, label in segments:
            # Load noise
            noise_waveform = self._load_noise()

            # Ensure noise and clean speech have the same length
            if noise_waveform.size(1) < clean_waveform.size(1):
                # Repeat noise until it's long enough
                repeat_times = (clean_waveform.size(1) // noise_waveform.size(1)) + 1
                noise_waveform = noise_waveform.repeat(1, repeat_times)[:, :clean_waveform.size(1)]
            else:
                # Randomly crop noise to match the length of clean speech
                start = random.randint(0, noise_waveform.size(1) - clean_waveform.size(1))
                noise_waveform = noise_waveform[:, start:start + clean_waveform.size(1)]

            # Calculate signal-to-noise ratio (SNR)
            clean_power = torch.norm(clean_waveform) ** 2
            noise_power = torch.norm(noise_waveform) ** 2
            desired_noise_power = clean_power / (10 ** (self.snr / 10))
            if noise_power > 0:
                noise_scaling_factor = (desired_noise_power / noise_power) ** 0.5
                noise_waveform = noise_waveform * noise_scaling_factor
            else:
                self.noise_zero_power_count += 1
                # If noise has zero power, use silence instead
                noise_waveform = torch.zeros_like(clean_waveform)

            # Mix clean speech and noise
            mixed_waveform = clean_waveform + noise_waveform

            # Process into mel spectrogram
            mel_spec = self.ava.mel_spectrogram(mixed_waveform)
            log_mel_spec = self.ava.log_mel_spectrogram(mel_spec)

            # Store the processed segment
            mixed_segments.append((log_mel_spec, label))

        return tuple(mixed_segments)

    def get_noise_zero_power_count(self):
        return self.noise_zero_power_count

if __name__ == "__main__":
    train_manifests = ['./data/manifest/balanced_background_training_manifest.json', './data/manifest/balanced_speech_training_manifest.json']
    test_dir = '/share/nas165/aaronelyu/Datasets/AVA-speech/'

    train_dataset = SCF(train_manifests, sample_duration=0.63)
    print(f"Training dataset size: {len(train_dataset)}")
    sample = train_dataset[0]
    # print(sample)
    for i, (log_mel_spec, label) in enumerate(train_dataset):
        print(f"Segment {i+1}: log_mel_spec shape: {log_mel_spec.shape}, label: {label}")

    # test_dataset = AVA(test_dir)
    # print(f"Test dataset size: {len(test_dataset)}")
    # sample = test_dataset[0]
    # print(sample)
    # for i, (_, log_mel_spec, label) in enumerate(sample):
    #     print(f"Segment {i+1}: log_mel_spec shape: {log_mel_spec.shape}, label: {label}")

    # train_noise_dataset = SCF_ESC(
    #     manifest_files=train_manifests,
    #     noise_csv='./data/ESC-50/esc50.csv',
    #     noise_audio_dir='./data/ESC-50/audio/',
    #     sample_duration=0.63,
    # )
    # print(f"Train noise dataset size: {len(train_noise_dataset)}")
    # sample, label = train_noise_dataset[0]
    # print(f"Sample shape: {sample.shape}, Label: {label}")

    # test_snr_dataset = AVA_ESC(
    #     ava_dir=test_dir,
    #     esc_csv='./data/ESC-50/esc50.csv',
    #     esc_audio_dir='./data/ESC-50/audio/',
    #     snr=0,
    # )
    # print(f"Test snr dataset size: {len(test_snr_dataset)}")
    # sample = test_snr_dataset[3]
    # for i, (log_mel_spec, label) in enumerate(sample):
    #     print(f"Segment {i+1}: log_mel_spec shape: {log_mel_spec.shape}, label: {label}")
