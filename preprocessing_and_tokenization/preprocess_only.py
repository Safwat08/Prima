import argparse
import json
from pathlib import Path

import SimpleITK as sitk
import torch
import yaml

from tools.models import ModelLoader
from tools.mrcommondataset import MrVoxelDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess NIfTI dataset into PRIMA token files.')
    parser.add_argument('--config',
                        required=True,
                        help='Path to config_input_data.yaml (or JSON).')
    parser.add_argument('--dataset_dir',
                        default=None,
                        help='Input dataset root with t1c/ and t2w/ folders.')
    parser.add_argument('--output_dir',
                        default=None,
                        help='Output root where data/, datajson.json are written.')
    parser.add_argument('--tokenizer_model_config',
                        default=None,
                        help='Path to tokenizer model config (YAML or JSON).')
    parser.add_argument('--vqvaename',
                        default='TOKENIZER',
                        help='Name for output emb folder under each series.')
    parser.add_argument('--max_tokens_per_chunk',
                        type=int,
                        default=400,
                        help='Chunk size for VQ-VAE encode.')
    parser.add_argument('--study_description',
                        default='',
                        help='Default study description in datajson.')
    parser.add_argument('--report_text',
                        default='',
                        help='Default report text in datajson.')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for VQ-VAE inference.')
    return parser.parse_args()


def load_config(path):
    p = Path(path)
    with open(p, 'r') as f:
        if p.suffix in ('.yaml', '.yml'):
            return yaml.safe_load(f)
        return json.load(f)


def study_id_from_path(path: Path):
    # Handles both .nii and .nii.gz
    if path.name.endswith('.nii.gz'):
        return path.name[:-7]
    return path.stem


def find_series_files(dataset_dir: Path):
    series_map = {}
    for series_name in ('t1c', 't2w'):
        series_dir = dataset_dir / series_name
        if not series_dir.exists():
            continue
        files = list(series_dir.glob('*.nii')) + list(series_dir.glob('*.nii.gz'))
        for fp in files:
            sid = study_id_from_path(fp)
            if sid not in series_map:
                series_map[sid] = {}
            series_map[sid][series_name] = fp
    return series_map


def encode_series(vqvae, image, device, max_tokens_per_chunk):
    dataset = MrVoxelDataset([image])
    tokens, ser_emb_meta = dataset[0]
    if not isinstance(tokens, torch.Tensor) or tokens.numel() == 0:
        return None, ser_emb_meta

    # tokens: [N, D, H, W] -> VQ-VAE expects [B, C, D, H, W]
    num_tokens = tokens.shape[0]
    chunks = []
    for i in range(0, num_tokens, max_tokens_per_chunk):
        chunk = tokens[i:i + max_tokens_per_chunk].unsqueeze(1).to(device)
        with torch.no_grad():
            emb = vqvae.encode(chunk).detach().cpu()
        chunks.append(emb)
    embeddings = torch.cat(chunks, dim=0)
    return embeddings, ser_emb_meta


def main():
    args = parse_args()
    input_config = load_config(args.config)

    def cfg_value(key, default=None):
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            return arg_val
        return input_config.get(key, default)

    dataset_dir_v = cfg_value('dataset_dir')
    output_dir_v = cfg_value('output_dir')
    tokenizer_model_config_v = cfg_value('tokenizer_model_config')
    if dataset_dir_v is None or output_dir_v is None or tokenizer_model_config_v is None:
        raise ValueError(
            'Missing required settings. Provide dataset_dir, output_dir, and '
            'tokenizer_model_config in config or CLI.')

    dataset_dir = Path(dataset_dir_v)
    output_dir = Path(output_dir_v)
    output_data_dir = output_dir / 'data'
    output_data_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_config = load_config(tokenizer_model_config_v)
    vqvae = ModelLoader.load_vqvae_model(tokenizer_config)
    device = cfg_value('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    max_tokens_per_chunk = int(cfg_value('max_tokens_per_chunk', 400))
    vqvaename = cfg_value('vqvaename', 'TOKENIZER')
    report_text = cfg_value('report_text', '')
    study_description = cfg_value('study_description', '')

    vqvae = vqvae.to(device).eval()

    series_map = find_series_files(dataset_dir)
    datajson = []

    for study_id in sorted(series_map.keys()):
        study_out_dir = output_data_dir / study_id
        study_out_dir.mkdir(parents=True, exist_ok=True)
        series_entries = []

        for series_name in ('t1c', 't2w'):
            if series_name not in series_map[study_id]:
                continue
            nii_path = series_map[study_id][series_name]
            image = sitk.ReadImage(str(nii_path))
            embeddings, ser_emb_meta = encode_series(vqvae, image, device,
                                                     max_tokens_per_chunk)
            if embeddings is None:
                continue

            emb_dir = study_out_dir / series_name / 'emb' / vqvaename
            stacked_dir = emb_dir / 'stacked'
            stacked_dir.mkdir(parents=True, exist_ok=True)

            torch.save(embeddings, stacked_dir / 'stacked.pt')
            with open(emb_dir / 'emb_meta.json', 'w') as f:
                json.dump(ser_emb_meta, f, indent=2)

            # Keep legacy six-number field for compatibility.
            series_entries.append([series_name, [0, 0, 0, 0, 0, 0]])

        if len(series_entries) == 0:
            continue

        datajson.append([
            str(study_out_dir),
            series_entries,
            report_text,
            study_description,
        ])

    with open(output_dir / 'datajson.json', 'w') as f:
        json.dump(datajson, f, indent=2)

    print(f'Wrote {len(datajson)} studies to {output_dir / "datajson.json"}')


if __name__ == '__main__':
    main()
