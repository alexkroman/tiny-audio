"""Names vulture cannot see being used (passed to `vulture` by `ta dev dead-code`).

Parameters of Protocol methods and `if TYPE_CHECKING:` signature stubs exist
only to declare a third-party call's signature for pyright, and imports
used only in quoted annotations are invisible to vulture. Add a name here only
for one of those reasons, never to silence genuinely dead code.
"""

function  # unused variable (scripts/train.py:115)
input_columns  # unused variable (scripts/train.py:118)
raw_tokens  # unused variable (tiny_audio/alignment.py:17)
processor_kwargs  # unused variable (tiny_audio/alignment.py:70)
Gemma4TextModel  # unused import (tiny_audio/asr_modeling.py:45)
bpayload  # unused variable (tiny_audio/asr_pipeline.py:16)
skip_special_tokens  # unused variable (tiny_audio/asr_pipeline.py:177)
tokenize  # unused variable (tiny_audio/asr_processing.py:60)
add_generation_prompt  # unused variable (tiny_audio/asr_processing.py:61)
enable_thinking  # unused variable (tiny_audio/asr_processing.py:63)
auto_class  # unused variable (tiny_audio/asr_processing.py:72)
config_class  # unused variable (tiny_audio/asr_processing.py:80)
processor_class  # unused variable (tiny_audio/asr_processing.py:80)
raw_speech  # unused variable (tiny_audio/asr_types.py:90)
return_attention_mask  # unused variable (tiny_audio/asr_types.py:94)
info_or_id  # unused variable (tiny_audio/handler.py:48)
quiet  # unused variable (tiny_audio/handler.py:48)
