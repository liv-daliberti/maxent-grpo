"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import maxent_grpo.rewards as R


def test_canon_math_strips_wrappers_and_spaces():
    canon = R._canon_math
    assert canon(" { 42 } ") == "42"
    assert canon("( 3.0 )") == "3"
    assert canon("-0") == "0"
    assert canon("12.0") == "12"
    assert canon("  \\sqrt  { 4 }  ") == "\\sqrt{4}".replace(" ", "")


def test_pure_accuracy_reward_math_happy_and_mismatch():
    comp_ok = "<think>...</think><answer> 42 </answer>"
    comp_bad = "<think>...</think><answer> 41 </answer>"
    gold = ["42", "42"]
    out = R.pure_accuracy_reward_math([comp_ok, comp_bad], gold)
    assert out == [1.0, 0.0]


def test_pure_accuracy_reward_requires_format_tags():
    # Missing tags yields 0.0
    out = R.pure_accuracy_reward_math(["no tags here"], ["anything"])
    assert out == [0.0]


def test_pure_accuracy_reward_relaxes_format_in_eval():
    comp = "no think but <answer>42</answer>"
    gold = ["42"]
    out_train = R.pure_accuracy_reward_math([comp], gold)
    out_eval = R.pure_accuracy_reward_math([comp], gold, is_eval=True)
    assert out_train == [0.0]
    assert out_eval == [1.0]


"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
