"""Names vulture cannot see being used (passed to `vulture` by `ta dev dead-code`).

Parameters of Protocol methods and `if TYPE_CHECKING:` signature stubs exist
only to declare a third-party call's signature for pyright, a test fake must
accept every parameter of the call it replaces, and imports used only in quoted
annotations are invisible to vulture. Add a name here only for one of those
reasons, never to silence genuinely dead code.
"""

Gemma4TextConfig  # unused import (tiny_audio/asr_modeling.py:41)
Gemma4TextModel  # unused import (tiny_audio/asr_modeling.py:45)
raw_speech  # unused variable (tiny_audio/asr_types.py:76)
return_attention_mask  # unused variable (tiny_audio/asr_types.py:80)
GenerativeDecoder  # unused import (tests/test_asr_modeling.py:33, quoted cast)
timestamp_id  # unused variable (tests/test_alignment.py:32, fake's signature)
config_override  # unused variable (tests/test_eval_dataset_loader.py:90, fake's signature)
