import importlib
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'ops/exp_scaling'),str(ROOT/'src')]

class NeutralCalibrationTests(unittest.TestCase):
    def test_generator_does_not_mutate_historical_catalog(self):
        import modebench_level3_python_v7 as old
        before=(old.PROFILE, old.CASE_WINDOWS, old.MINIMUM_BANDS, old.catalog(0))
        import modebench_level3_python_neutral_v2 as new
        self.assertEqual(before,(old.PROFILE, old.CASE_WINDOWS, old.MINIMUM_BANDS, old.catalog(0)))
        self.assertNotEqual(old.PROFILE,new.PROFILE)
        self.assertTrue(any(n<60 for n in new.catalog(0)[0]))
        self.assertFalse(any(n<60 for n in old.catalog(0)[0]))

    def test_exact_neutral_interface_preserves_decoding(self):
        import evaluate_modebench_level3_neutral as new
        import modebench_current_contract as current
        for field in ('sample_count','temperature','top_p','max_tokens','max_model_len','dtype','syntax_profile','seed_policy'):
            self.assertEqual(new.frozen_interface('python_factors')[field], new._BASE_INTERFACE('python_factors')[field])
        problem='Write one pure Python function with the exact form lambda n: EXPR.'
        self.assertEqual(new.prompt_messages('python_factors',problem,current.CURRENT),current.make_messages(3,'python_factors',{'problem':problem}))
        self.assertNotIn('Test small divisors',new.prompt_messages('python_factors',problem,current.CURRENT)[0]['content'])
        with self.assertRaises(ValueError):new.frozen_interface('mathir')
        with self.assertRaises(ValueError):new.prompt_messages('python_factors',problem,'hybrid_solver_v4')

    def test_case_stream_is_stable_and_excludes_past_cases(self):
        import modebench_level3_python_neutral_v2 as new
        from itertools import islice
        a=list(islice(new._sampler.case_stream(32,set(),1200911,0),4))
        b=list(islice(new._sampler.case_stream(32,set(),1200911,0),2))
        self.assertEqual(a[:2],b)
        excluded={('python_factors',a[0])}
        c=list(islice(new._sampler.case_stream(32,excluded,1200911,0),3))
        self.assertNotIn(a[0],c)
        self.assertEqual(a[1:],c)

if __name__=='__main__':unittest.main()
