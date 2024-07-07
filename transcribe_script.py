import os
import csv
import argparse
import nemo.collections.asr as nemo_asr


def transcribe_audio(checkpoint_path, data_dir, output_csv='transcriptions.csv', batch_size=4):
    # Restore the ASR model from the checkpoint
    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(checkpoint_path)

    # List all .wav files in the directory
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

    # Prepare the list of audio paths
    audio_paths = [os.path.join(data_dir, wav) for wav in wav_files]

    # Transcribe the audio files in batches
    transcriptions = []
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        transcripts = asr_model.transcribe(audio=batch_paths, batch_size=len(batch_paths))
        transcriptions.extend(transcripts)

    # Prepare data for CSV
    csv_data = []
    for wav, transcript in zip(wav_files, transcriptions):
        audio_name = os.path.splitext(wav)[0]
        csv_data.append([audio_name, transcript])

    # Write to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['audio', 'transcript'])
        writer.writerows(csv_data)

    print(f"Transcriptions saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe audio files and save to CSV.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the .nemo ASR model checkpoint file.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the .wav files.')
    parser.add_argument('--output_csv', type=str, default='transcriptions.csv', help='Output CSV file name.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for transcription.')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the transcribe function with parsed arguments
    transcribe_audio(args.checkpoint_path, args.data_dir, args.output_csv, args.batch_size)
