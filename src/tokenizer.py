from typing import List, Optional
from transformers import BertTokenizer


class TwinBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, add_cls_tokens=False, **kwargs):
        super(TwinBertTokenizer, self).__init__(vocab_file=vocab_file, **kwargs)
        self.add_cls_tokens = add_cls_tokens

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if self.add_cls_tokens:
            return [self.cls_token_id] + token_ids_0
        return token_ids_0

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if self.add_cls_tokens:
            return [1] + ([0] * len(token_ids_0))
        return [0] * len(token_ids_0)

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if self.add_cls_tokens:
            return [0] * (1 + len(token_ids_0))
        return [0] * len(token_ids_0)
