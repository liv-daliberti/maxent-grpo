import json
from pathlib import Path
import tempfile
import unittest

import prepare_real_domains_resume_20260921 as resume


class ResumePreparationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / 'original'
        self.bundle = self.original / 'bundle'
        self.training = self.original / 'training'
        self.checkpoint = self.training / 'checkpoint-16'
        self.adapter = self.checkpoint / 'adapter'
        self.adapter.mkdir(parents=True)
        self.runner = self.bundle / 'ops/train_real_domains_pilot_20260921.py'
        self.module = self.bundle / 'src/oat_drgrpo/example.py'
        self.runner.parent.mkdir(parents=True)
        self.module.parent.mkdir(parents=True)
        self.runner.write_text('TRAINER = True\n')
        self.module.write_text('ADAPTER = True\n')
        self.model = self.root / ('a' * 40)
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        self.config = {'model': str(self.model), 'model_revision': 'a' * 40, 'updates': 32}
        self.config_path = self.original / 'config.json'
        self.config_path.write_text(json.dumps(self.config))
        frozen = {'schema': 'real-domains-frozen-job-20260921-v1',
                  'request': {'arm': 'remax', 'entrypoint': self.runner.name},
                  'config_sha256': resume.digest(self.config_path),
                  'files': [{'snapshot': str(p), 'sha256': resume.digest(p)} for p in [self.runner, self.module]]}
        (self.original / 'identity.json').write_text(json.dumps(frozen))
        self.training_identity = {'arm': 'remax', 'input_config_sha256': resume.digest(self.config_path),
                                  'config': self.config, 'config_sha256': resume.canonical_hash(self.config),
                                  'runner_sha256': resume.digest(self.runner), 'adapter_module': str(self.module),
                                  'adapter_module_sha256': resume.digest(self.module),
                                  'model_config_sha256': resume.digest(self.model / 'config.json')}
        (self.training / 'identity.json').write_text(json.dumps(self.training_identity))
        (self.checkpoint / 'bank.json').write_text('{}')
        (self.checkpoint / 'training.pt').write_bytes(b'sealed optimizer state')
        (self.adapter / 'adapter_config.json').write_text('{}')
        (self.adapter / 'adapter_model.safetensors').write_bytes(b'sealed adapter weights')
        self.seal = {'arm': 'remax', 'config_sha256': resume.canonical_hash(self.config), 'completed_updates': 16,
                     'training_state_sha256': resume.digest(self.checkpoint / 'training.pt'),
                     'bank_sha256': resume.digest(self.checkpoint / 'bank.json'),
                     'adapter_files': {p.name: resume.digest(p) for p in self.adapter.iterdir()}}
        self.save_seal()

    def save_seal(self):
        (self.checkpoint / 'complete.json').write_text(json.dumps(self.seal))

    def validate(self):
        return resume.validate_original(self.original, self.checkpoint, 'remax')

    def test_validates_original_paths_without_rewriting(self):
        before = self.config_path.read_bytes()
        result = self.validate()
        self.assertEqual(result['completed_updates_before_resume'], 16)
        self.assertEqual(result['original_target_updates'], 32)
        self.assertEqual(result['config_path'], str(self.config_path))
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_rejects_wrong_arm(self):
        with self.assertRaisesRegex(ValueError, 'arm differs'):
            resume.validate_original(self.original, self.checkpoint, 'maxrl')

    def test_rejects_changed_original_config(self):
        self.config_path.write_text(self.config_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'configuration drift'):
            self.validate()

    def test_rejects_changed_frozen_source(self):
        self.module.write_text('ADAPTER = False\n')
        with self.assertRaisesRegex(ValueError, 'dependency drift'):
            self.validate()

    def test_rejects_config_mismatched_seal(self):
        self.seal['config_sha256'] = '0' * 64
        self.save_seal()
        with self.assertRaisesRegex(ValueError, 'seal configuration'):
            self.validate()

    def test_rejects_changed_optimizer_state(self):
        (self.checkpoint / 'training.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'state or bank drift'):
            self.validate()

    def test_rejects_unsealed_extra_adapter_file(self):
        (self.adapter / 'extra_config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'file set differs'):
            self.validate()

    def test_rejects_completed_checkpoint(self):
        self.seal['completed_updates'] = 32
        self.save_seal()
        with self.assertRaisesRegex(ValueError, 'must precede'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
