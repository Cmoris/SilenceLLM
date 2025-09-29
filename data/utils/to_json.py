import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

class DataProcessor:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.df = pd.read_csv(config['annotation_path'])
        self.video_path = Path(config['video_path'])
        self.audio_path = Path(config['audio_path'])
        
        # Define prompts and classes
        self.json_classes = [
            'stopped', 
            'thinking'
        ]
        self.prompt_v = """Output only "stopped" or "thinking" based on the silent section at the end of the video. Please judge whether user has stopped speech or is thinking."""
        self.prompt_a = """Output only "stopped" or "thinking" based on the silent section at the end of the video. Please judge whether user has stopped speech or is thinking."""
        self.prompt_mm = """Output only "stopped" or "thinking" based on the silent section at the end of the video. Please judge whether user has stopped speech or is thinking."""
        
        # Date ranges for train/test splits
        self.date_ranges = {
            'train': (datetime.strptime("20231106", "%Y%m%d"), datetime.strptime("20240118", "%Y%m%d")),
            'test': (datetime.strptime("20240119", "%Y%m%d"), datetime.strptime("20240124", "%Y%m%d"))
        }

    def create_conversation_data(self, item_id: int, label: int) -> Tuple[Dict, Dict, Dict, Dict]:
        """Create video, audio, multimodal, and text conversation data for a single item."""
        audio_file = str(self.audio_path / f"{item_id}.wav")
        video_file = str(self.video_path / f"{item_id}.mp4")
        
        
        # Audio conversation data
        audio_data = {
            "id": item_id,
            "audio": audio_file,
            "conversations": [
                {
                    "from": "human",
                    "value": f"{self.prompt_a}"
                },
                {
                    "from": "gpt",
                    "value": self.json_classes[label]
                }
            ]
        }
        
        # Video conversation data
        video_data = {
            "id": item_id,
            "video": video_file,
            "conversations": [
                {
                    "from": "human",
                    "value": f"{self.prompt_v}"
                },
                {
                    "from": "gpt",
                    "value": self.json_classes[label]
                }
            ]
        }
        
        # Multimodal (video + audio) conversation data
        multimodal_data = {
            "id": item_id,
            "video": video_file,
            "audio": audio_file,
            "conversations": [
                {
                    "from": "human",
                    "value": f"{self.prompt_mm}"
                },
                {
                    "from": "gpt",
                    "value": self.json_classes[label]
                }
            ]
        }
        
        
        return video_data, audio_data, multimodal_data

    def get_date_split(self, filename: str) -> str:
        """Determine which split (train/test) a file belongs to based on its date."""
        try:
            date_str = filename[:8]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            
            for split_name, (start_date, end_date) in self.date_ranges.items():
                if start_date <= file_date <= end_date:
                    return split_name
            return 'test'  # Default to test if not in any range
        except ValueError:
            return 'test'  # Default to test if date parsing fails

    def process_data(self) -> Dict[str, List]:
        """Process all data and split into train/test sets."""
        splits = {'train': [], 'test': []}
        
        ids = self.df["file_name"].values
        labels = self.df.iloc[:, 2:].values
        
        for i, (item_id, label_row) in enumerate(zip(ids, labels)):
            # Skip if no valid label
            if np.sum(label_row) == 0:
                continue
                
            label = np.argmax(label_row)
            
            # Create conversation data
            video_data, audio_data, multimodal_data = self.create_conversation_data(item_id, label)
            
            # Determine split based on date
            filename = f"{item_id}.mp4"
            split_name = self.get_date_split(filename)
            
            # Add to appropriate split
            splits[split_name].append([video_data, audio_data, multimodal_data])
        
        return splits

    def save_splits(self, splits: Dict[str, List]) -> None:
        """Save the processed data splits to JSON files."""
        output_dir = Path(self.config['output_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract individual modalities for each split
        for split_name, data_list in splits.items():
            if not data_list:  # Skip empty splits
                continue
                
            video_data = [item[0] for item in data_list]
            audio_data = [item[1] for item in data_list]
            multimodal_data = [item[2] for item in data_list]  # New multimodal data
            
            # Save files
            self._save_json_file(output_dir / f"config_v_{split_name}.json", video_data)
            self._save_json_file(output_dir / f"config_a_{split_name}.json", audio_data)
            self._save_json_file(output_dir / f"config_mm_{split_name}.json", multimodal_data)  # Multimodal
            
            # Also save combined data
            self._save_json_file(output_dir / f"config_combined_{split_name}.json", data_list)
        
        # Save train files without suffix for backward compatibility
        if splits['train']:
            train_video = [item[0] for item in splits['train']]
            train_audio = [item[1] for item in splits['train']]
            train_multimodal = [item[2] for item in splits['train']]  # New multimodal
            
            self._save_json_file(output_dir / "config_v.json", train_video)
            self._save_json_file(output_dir / "config_a.json", train_audio)
            self._save_json_file(output_dir / "config_mm.json", train_multimodal)  # Multimodal
        
        self._print_statistics(splits)

    def _save_json_file(self, filepath: Path, data: List) -> None:
        """Save data to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _print_statistics(self, splits: Dict[str, List]) -> None:
        """Print dataset statistics."""
        total = sum(len(data) for data in splits.values())
        if total > 0:
            print(f"Dataset statistics:")
            for split_name, data in splits.items():
                count = len(data)
                percentage = (count / total) * 100
                print(f"  {split_name}: {count} ({percentage:.1f}%)")

def main():
    # Configuration
    config = {
        'annotation_path': "/n/work1/muyun/Dataset/datasets_v3/annotation.csv",
        'video_path': "/n/work1/muyun/Dataset/datasets_v3/videos",
        'audio_path': "/n/work1/muyun/Dataset/datasets_v3/audios",
        'output_path': "/n/work1/muyun/Dataset/datasets_v3/try"
    }
    
    # Process data
    processor = DataProcessor(config)
    splits = processor.process_data()
    processor.save_splits(splits)

if __name__ == "__main__":
    main()