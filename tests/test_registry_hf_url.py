"""Tests of ``refresh_hf_url`` (no network: the Hub check is replaced by a stub)."""

import polars as pl

from src.utils.timm_metadata import _hf_hub_id, refresh_hf_url

M1 = 'resnet50.a1_in1k'
M2 = 'vit_base_patch16_224.augreg_in21k'
URL1 = 'https://huggingface.co/timm/resnet50.a1_in1k'


def test_hf_hub_id_comes_from_timm_config():
    assert _hf_hub_id(M1) == 'timm/resnet50.a1_in1k'


def test_missing_column_is_built_second_and_checked():
    df = pl.DataFrame({'model_name': [M1, M2], 'x': [1, 2]})
    out = refresh_hf_url(df, checker=lambda repo: repo == 'timm/resnet50.a1_in1k')
    assert out.columns == ['model_name', 'hf_url', 'x']
    assert out.schema['hf_url'] == pl.String
    # unreachable repository -> null, never a guessed URL
    assert out['hf_url'].to_list() == [URL1, None]
    assert out.drop('hf_url').equals(df)


def test_stored_values_kept_for_models_not_verified():
    df = pl.DataFrame({'model_name': [M1, M2], 'hf_url': [None, 'stored']})

    def never(_):
        raise AssertionError('the Hub must not be queried')

    out = refresh_hf_url(df, verify=None, checker=never)
    assert out['hf_url'].to_list() == [None, 'stored']
    assert out.columns == ['model_name', 'hf_url']


def test_only_requested_models_are_verified_and_column_stays_second():
    df = pl.DataFrame(
        {'model_name': [M1, M2], 'hf_url': ['old', 'stored'], 'y': [1, 2]}
    )
    seen = []

    def ok(repo):
        seen.append(repo)
        return True

    out = refresh_hf_url(df, verify={M1}, checker=ok)
    assert seen == ['timm/resnet50.a1_in1k']
    assert out['hf_url'].to_list() == [URL1, 'stored']
    assert out.columns == ['model_name', 'hf_url', 'y']
