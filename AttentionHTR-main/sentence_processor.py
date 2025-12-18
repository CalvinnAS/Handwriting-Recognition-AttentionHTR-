import cv2
import numpy as np
from PIL import Image
import os
import subprocess
import shutil
import sys

class SentenceHTRPipeline:

    def __init__(self, model_dir='./model', model_path='saved_models/AttentionHTR-General-sensitive.pth', python_cmd='python'):
        self.model_dir = model_dir
        self.model_path = model_path
        self.python_cmd = python_cmd

        self.create_lmdb_script = os.path.join(model_dir, 'create_lmdb_dataset.py')
        self.test_script = os.path.join(model_dir, 'test.py')


    def segment_words(self, image_path, output_dir, min_word_width=10):
        os.makedirs(output_dir, exist_ok=True)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


        print(f"Processing image: {image_path}")
        print(f"Image shape: {img.shape}")

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        vertical_projection = np.sum(binary, axis=0)

        threshold = np.max(vertical_projection) * 0.1
        text_columns = vertical_projection > threshold

        words_boundaries = []
        in_word = False
        start = 0

        for i, has_text in enumerate(text_columns):
            if has_text and not in_word:
                start = i
                in_word = True
            elif not has_text and in_word:
                if i - start > min_word_width:
                    words_boundaries.append((start, i))
                in_word = False

        if in_word and len(text_columns) - start > min_word_width:
            words_boundaries.append((start, len(text_columns)))

        merged_boundaries = []
        if len(words_boundaries) > 0:
            current_start, current_end = words_boundaries[0]

            for i in range(1, len(words_boundaries)):
                next_start, next_end = words_boundaries[i]
                gap = next_start - current_end
                avg_word_width = (current_end - current_start + next_end - next_start) / 2

                if gap < avg_word_width * 0.3:
                    current_end = next_end
                else:
                    merged_boundaries.append((current_start, current_end))
                    current_start, current_end = next_start, next_end

            merged_boundaries.append((current_start, current_end))

        print(f"Ditemukan {len(merged_boundaries)} kata")

        word_info = []
        padding = 5

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for idx, (start, end) in enumerate(merged_boundaries):
            left = max(0, start - padding)
            right = min(img.shape[1], end + padding)

            word_img = img[:, left:right]

            word_filename = f"{base_name}_word_{idx+1:02d}.png"
            word_path = os.path.join(output_dir, word_filename)

            cv2.imwrite(word_path, word_img)

            word_info.append((word_filename, "UNKNOWN"))

            print(f"  Saved: {word_filename}")

        return word_info

    def create_ground_truth_file(self, word_info, gt_file_path):

        with open(gt_file_path, 'w', encoding='utf-8') as f:
            for filename, text in word_info:
                f.write(f"{filename}\t{text}\n")

        print(f"\nGround truth file created: {gt_file_path}")

    def create_lmdb_dataset(self, input_dir, gt_file, output_dir):
        print(f"\n{'='*60}")
        print("Creating LMDB dataset")
        print(f"{'='*60}")

        input_dir_abs = os.path.abspath(input_dir)
        gt_file_abs = os.path.abspath(gt_file)
        output_dir_abs = os.path.abspath(output_dir)

        cmd = [
            sys.executable,
            'create_lmdb_dataset.py',
            '--inputPath', input_dir_abs,
            '--gtFile', gt_file_abs,
            '--outputPath', output_dir_abs
        ]

        print(f"Working directory: {self.model_dir}")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=self.model_dir, capture_output=True, text=True)

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Failed to create LMDB dataset: {result.stderr}")

        print(result.stdout)
        print(f"LMDB dataset created: {output_dir}")

    def run_prediction(self, lmdb_path, case_sensitive=True):
        print(f"\n{'=' * 60}")
        print("Running prediction with AttentionHTR")
        print(f"{'=' * 60}")

        lmdb_path_abs = os.path.abspath(lmdb_path)

        cmd = [
            sys.executable,
            'test.py',
            '--eval_data', lmdb_path_abs,
            '--Transformation', 'TPS',
            '--FeatureExtraction', 'ResNet',
            '--SequenceModeling', 'BiLSTM',
            '--Prediction', 'Attn',
            '--saved_model', self.model_path,
            '--workers', '0'
        ]

        if case_sensitive:
            cmd.append('--sensitive')

        print(f"Working directory: {self.model_dir}")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=self.model_dir, capture_output=True, text=True)

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Failed to run prediction: {result.stderr}")

        print(result.stdout)

        return self.parse_predictions_from_stdout(result.stdout)

    def parse_predictions_from_stdout(self, stdout_text):
        predictions = []
        lines = stdout_text.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('GT:') and 'Prediction:' in line:
                parts = line.split('Prediction:')
                if len(parts) >= 2:
                    predicted_text = parts[1].strip()
                    predictions.append(predicted_text)

        return predictions
    def parse_predictions(self, log_file):
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Prediction log tidak ditemukan: {log_file}")

        predictions = []

        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith('Accuracy') or line.startswith('Norm ED'):
                    continue

                parts = line.split('\t')
                if len(parts) >= 3:
                    predicted_text = parts[2].strip()
                    predictions.append(predicted_text)

        return predictions

    def process_sentence_image(self, image_path, work_dir='temp_processing', case_sensitive=True, cleanup=True):
        os.makedirs(work_dir, exist_ok=True)

        words_dir = os.path.join(work_dir, 'words')
        gt_file = os.path.join(work_dir, 'gt.txt')
        lmdb_dir = os.path.join(work_dir, 'lmdb_dataset')

        try:
            print(f"\n{'='*60}")
            print("Segmenting words from sentence image")
            print(f"{'='*60}")
            word_info = self.segment_words(image_path, words_dir)

            if len(word_info) == 0:
                raise ValueError("Tidak ada kata yang terdeteksi dalam gambar")

            print(f"\n{'='*60}")
            print("Creating ground truth file")
            print(f"{'='*60}")
            self.create_ground_truth_file(word_info, gt_file)

            print(f"\n{'='*60}")
            print("Creating LMDB dataset")
            print(f"{'='*60}")
            self.create_lmdb_dataset(words_dir, gt_file, lmdb_dir)

            print(f"\n{'='*60}")
            print("Running prediction")
            print(f"{'='*60}")
            predictions = self.run_prediction(lmdb_dir, case_sensitive)

            sentence = ' '.join(predictions)

            print(f"\n{'='*60}")
            print("FINAL RESULT")
            print(f"{'='*60}")
            print(f"Recognized words: {predictions}")
            print(f"Full sentence: {sentence}")
            print(f"{'='*60}\n")

            return sentence

        except Exception as e:
            print(f"\nError during processing: {e}")
            raise

        finally:
            if cleanup:
                print("\nCleaning up temporary files...")
                if os.path.exists(work_dir):
                    shutil.rmtree(work_dir)
                print("Cleanup completed.")


# Usage

def example_single_image():
    pipeline = SentenceHTRPipeline(
        model_dir='./model',
        model_path='saved_models/AttentionHTR-General-sensitive.pth',
        python_cmd='python'
    )

    image_path = 'image1.png'

    try:
        sentence = pipeline.process_sentence_image(
            image_path=image_path,
            work_dir='temp_processing',
            case_sensitive=True,
            cleanup=True
        )

        print(f"\nFinal Result: {sentence}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def example_batch_processing():

    pipeline = SentenceHTRPipeline(
        model_dir='./model',
        model_path='saved_models/AttentionHTR-General-sensitive.pth',
        python_cmd='python'
    )

    image_files = [
        'sentence1.png',
        'sentence2.png',
        'sentence3.png'
    ]

    results = {}

    for img_path in image_files:
        try:
            print(f"\n\nProcessing: {img_path}")
            sentence = pipeline.process_sentence_image(img_path)
            results[img_path] = sentence
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            results[img_path] = None

    print("\n\n" + "="*60)
    print("BATCH PROCESSING RESULTS")
    print("="*60)
    for img_path, sentence in results.items():
        print(f"{img_path}:")
        print(f"  → {sentence}\n")


if __name__ == "__main__":
    example_single_image()
