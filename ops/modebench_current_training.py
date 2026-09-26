"""Complete neutral Python L3 launch defaults: admitted data and tested native runtime."""
from pathlib import Path
import modebench_current_data as data

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'var/artifacts/python_level3_cli_recovery_20260912'
RUNTIME_IDENTITIES={
    'e122':'33302f4c695a212fd967b7475b59e2244db9d81e7352ae68d8e37c1d56769310',
    'e124':'7e2e076037a3d22c4dec5dcae036901c9cabbe3f1e2b43c8bd5593d11fd6e004',
}


def neutral_python_training_environment(environment, *, campaign):
    """Preserve the native recipe while binding its tested 0.5B or 7B successor.

    Select e122 for the registered Qwen-0.5B recipe or e124 for its Qwen-7B
    counterpart. This prepares configuration only; ordinary campaign resource
    and qualification gates still decide whether a job may be released.
    """
    if campaign not in RUNTIME_IDENTITIES:
        raise ValueError('Select the registered e122 or e124 native runtime')
    runtime=ART/'runtime_v2'/campaign
    manifest=runtime/'CLI_AMENDMENT_IDENTITY.json'
    expected=RUNTIME_IDENTITIES[campaign]
    if data.digest(manifest)!=expected:
        raise ValueError('Neutral Python runtime identity changed')
    for relative,digest in data.read(manifest)['inventory_sha256'].items():
        if data.digest(runtime/relative)!=digest:
            raise ValueError('Neutral Python runtime source changed: '+relative)
    result=data.training_environment(3,'python_factors',environment)
    result.update(OAT_ZERO_SOURCE_ROOT=str(runtime/'src'),
                  OAT_ZERO_OPS_SNAPSHOT_ROOT=str(runtime/'ops'),
                  OAT_ZERO_MODEBENCH_RUNTIME_IDENTITY_SHA256=expected)
    return result
