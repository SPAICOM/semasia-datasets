import sys

from datasets import load_dataset


def inspect_split(ds, name: str, n: int = 3):
    print(f'\n--- Inspecting split: {name} ---')
    print(f'Length: {len(ds)}')
    print('Features:')
    for k, v in ds.features.items():
        print(f'  - {k}: {v}')

    print(f'\nFirst {n} samples:')
    for i in range(min(n, len(ds))):
        row = ds[i]
        emb = row.get('embedding')

        print(f'\nSample {i}')
        print(f'  label: {row.get("label")}')
        print(f'  model_name: {row.get("model_name")}')

        if emb is not None:
            print(f'  embedding: shape={len(emb)}, dtype={type(emb[0])}')
        else:
            print('  embedding: MISSING')


def main(
    repo_id: str = 'spaicom-lab/semantic-cifar100',
    config: str = 'vit_small_patch14_dinov2.lvd142m',
):
    print(f'Testing dataset: {repo_id}')
    print(f'Config: {config}')

    try:
        ds_train = load_dataset(repo_id, config, split='train')
        ds_test = load_dataset(repo_id, config, split='test')
    except Exception:
        print('\n[ERROR] Failed to load dataset')
        raise

    inspect_split(ds_train, 'train')
    inspect_split(ds_test, 'test')

    # Minimal semantic checks
    assert 'embedding' in ds_train.features, "Missing 'embedding' column"

    print('\n[OK] Dataset loaded and validated successfully.')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
